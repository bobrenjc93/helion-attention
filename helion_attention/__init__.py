"""helion-attention: FlashAttention's API, served by Triton kernels.

The kernels in :mod:`helion_attention.kernels` were generated and autotuned by
`Helion <https://github.com/pytorch/helion>`_, but they are plain Triton by the
time they are checked in: importing this package pulls in ``torch`` and
``triton`` and nothing else.

Every entry point takes a required ``shape`` argument. Registered shapes use an
exact generated specialization; compatible unregistered dense shapes and dense
ALiBi calls use a generic Triton forward kernel. Grad-enabled dense calls
without a generated backward use PyTorch SDPA autograd. The explicit shape
validates these paths and makes specialization introspection independent of
fallback coverage.
"""

from __future__ import annotations

import math

import torch

from ._autograd import attention_autograd
from ._registry import UnsupportedShapeError
from ._registry import available_paged_shapes
from ._registry import available_shapes
from ._registry import available_varlen_shapes
from ._registry import has_backward
from ._registry import has_kernel
from ._registry import has_paged_kernel
from ._registry import has_varlen_kernel
from ._registry import lookup
from ._registry import lookup_backward
from ._registry import lookup_paged
from ._registry import lookup_varlen
from ._sdpa import dense_attention_sdpa
from ._shape import AttnShape
from ._shape import ShapeLike
from ._shape import check_paged_varlen_tensors
from ._shape import check_tensors
from ._shape import check_varlen_tensors
from ._shape import normalize_shape

__all__ = [
    "AttnShape",
    "UnsupportedShapeError",
    "available_paged_shapes",
    "available_shapes",
    "available_varlen_shapes",
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
    "is_shape_supported",
    "is_paged_shape_supported",
    "is_varlen_shape_supported",
]

__version__ = "0.1.0"

_CORE_PAGED_CHUNKED_PREFILL_SHAPE = (
    2,
    200,
    320,
    8,
    2,
    128,
    torch.bfloat16,
)
_CORE_PAGED_KVCACHE_SHAPE = (4, 1, 1024, 8, 2, 128, torch.bfloat16)
_CORE_PAGED_VARLEN_SHAPES = frozenset(
    {_CORE_PAGED_CHUNKED_PREFILL_SHAPE, _CORE_PAGED_KVCACHE_SHAPE}
)
_CORE_PAGED_VARLEN_PAGE_SIZE = 16
_GENERIC_DENSE_MAX_HEAD_DIM = 256
_INT32_MAX = 2**31 - 1
_DIAGNOSTIC_DECODE_PROFILES = frozenset(
    {
        (1, 1, cache_length, 32, 8, 128, torch.bfloat16)
        for cache_length in (1024, 4096, 16384)
    }
)


def _reject_unsupported(
    dropout_p: float,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_attn_probs: bool,
    *,
    allow_alibi: bool = False,
    allow_return_attn_probs: bool = False,
) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not implemented; pass dropout_p=0.0")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("sliding-window attention is not implemented")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not implemented")
    if alibi_slopes is not None and not allow_alibi:
        raise NotImplementedError("ALiBi slopes are not implemented")
    if return_attn_probs and not allow_return_attn_probs:
        raise NotImplementedError("return_attn_probs is not implemented")


def _supports_diagnostic_return(spec: AttnShape) -> bool:
    """Whether ``spec`` has one of the three LSE-capable decode kernels."""
    profile = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    # Causal and non-causal are equivalent for a bottom-right single-token
    # query, and the registry maps both modes to the same checked-in kernel.
    return profile in _DIAGNOSTIC_DECODE_PROFILES and has_kernel(spec)


def _contiguous_tensors_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Whether two validated contiguous tensors share any device memory."""
    first_start = first.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_start = second.data_ptr()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def _validate_kvcache_rotary(
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    q: torch.Tensor,
    spec: AttnShape,
) -> None:
    """Validate the narrow full-head rotary contract for a final-slot append."""
    for name, tensor in (("rotary_cos", rotary_cos), ("rotary_sin", rotary_sin)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use torch.strided layout")
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have shape [seqlen_ro, head_dim / 2], got "
                f"{tuple(tensor.shape)}"
            )

    if rotary_cos.shape != rotary_sin.shape:
        raise ValueError(
            "rotary_cos and rotary_sin must have the same shape, got "
            f"{tuple(rotary_cos.shape)} and {tuple(rotary_sin.shape)}"
        )
    rotary_dim = rotary_cos.shape[1] * 2
    if rotary_dim != spec.head_dim:
        raise NotImplementedError(
            "partial rotary dimensions are not implemented; full-head rotary "
            f"requires rotary_dim={spec.head_dim}, got {rotary_dim}"
        )
    if rotary_cos.shape[0] < spec.seqlen_k:
        raise ValueError(
            "rotary_cos and rotary_sin must contain the final cache position; "
            f"seqlen_ro must be at least {spec.seqlen_k}, got "
            f"{rotary_cos.shape[0]}"
        )

    for name, tensor in (("rotary_cos", rotary_cos), ("rotary_sin", rotary_sin)):
        if tensor.dtype != spec.dtype:
            raise ValueError(
                f"{name} has dtype {tensor.dtype} but q has dtype {spec.dtype}"
            )
        if not tensor.is_cuda:
            raise ValueError(
                f"{name} must be a CUDA tensor, got device {tensor.device}"
            )
        if tensor.device != q.device:
            raise ValueError(
                f"{name} must be on the same CUDA device as q, k_cache, and v_cache"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _apply_interleaved_rotary(
    tensor: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """Rotate adjacent full-head pairs with one fp32-rounded rotary table row."""
    pairs = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
    cos = rotary_cos[position].float()
    sin = rotary_sin[position].float()
    even = pairs[..., 0]
    odd = pairs[..., 1]
    return (
        torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        .flatten(-2)
        .to(dtype=tensor.dtype)
    )


def _check_core_paged_varlen_spec(spec: AttnShape, page_size: int) -> None:
    """Require a paged specialization exposed by the core varlen API."""
    requested = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    if (
        requested not in _CORE_PAGED_VARLEN_SHAPES
        or (
            requested == _CORE_PAGED_CHUNKED_PREFILL_SHAPE
            and not spec.causal
        )
        or page_size != _CORE_PAGED_VARLEN_PAGE_SIZE
    ):
        raise UnsupportedShapeError(
            "flash_attn_varlen_func with block_table currently supports only "
            "these bf16 profiles with page_size=16:\n"
            "    batch=2 seqlen_q=200 seqlen_k=320 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True\n"
            "    batch=4 seqlen_q=1 seqlen_k=1024 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True or causal=False\n"
            f"got:\n    {spec.describe()}, page_size={page_size}"
        )


def _check_core_paged_kvcache_spec(spec: AttnShape, page_size: int) -> None:
    """Require the one paged specialization exposed by the KV-cache API."""
    requested = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    if (
        requested != _CORE_PAGED_KVCACHE_SHAPE
        or page_size != _CORE_PAGED_VARLEN_PAGE_SIZE
    ):
        raise UnsupportedShapeError(
            "flash_attn_with_kvcache with block_table currently supports only:\n"
            "    batch=4 seqlen_q=1 seqlen_k=1024 nheads=8 (GQA 8:2) "
            "head_dim=128 dtype=bf16 page_size=16\n"
            f"got:\n    {spec.describe()}, page_size={page_size}"
        )


def _validate_generic_dense_layout(spec: AttnShape) -> tuple[int, int]:
    """Validate signed-int32 indices and return packed Q/K token totals."""
    if spec.head_dim > _GENERIC_DENSE_MAX_HEAD_DIM:
        raise UnsupportedShapeError(
            "no checked-in dense specialization exists for:\n"
            f"    {spec.describe()}\n"
            "the generic dense fallback supports head_dim <= "
            f"{_GENERIC_DENSE_MAX_HEAD_DIM}. To request a specialization, file "
            "an issue at https://github.com/bobrenjc93/helion-attention/issues"
        )

    total_q = spec.batch * spec.seqlen_q
    total_k = spec.batch * spec.seqlen_k
    if max(total_q, total_k) > _INT32_MAX:
        raise UnsupportedShapeError(
            "no checked-in dense specialization exists for:\n"
            f"    {spec.describe()}\n"
            "the generic dense fallback requires packed token offsets to fit "
            "in int32. To request a specialization, file an issue at "
            "https://github.com/bobrenjc93/helion-attention/issues"
        )

    # The packed kernel forms pointers with signed-int32 element arithmetic.
    # Token offsets alone are insufficient: token * stride can overflow first
    # for a large head count or dimension. Q and output share one layout, as do
    # K and V, so bound the maximum valid relative element offset for each.
    layout_numels = (
        ("Q/output", total_q * spec.nheads_q * spec.head_dim),
        ("K/V", total_k * spec.nheads_kv * spec.head_dim),
    )
    for layout, numel in layout_numels:
        max_element_offset = numel - 1
        if max_element_offset > _INT32_MAX:
            raise UnsupportedShapeError(
                "no checked-in dense specialization exists for:\n"
                f"    {spec.describe()}\n"
                "the generic dense fallback uses signed int32 element offsets, "
                f"but {layout} requires maximum offset {max_element_offset} "
                f"(limit {_INT32_MAX}). To request a specialization, file an "
                "issue at https://github.com/bobrenjc93/helion-attention/issues"
            )
    return total_q, total_k


def _validate_dense_alibi_slopes(
    alibi_slopes: torch.Tensor,
    q: torch.Tensor,
    spec: AttnShape,
) -> None:
    """Validate FlashAttention's dense ALiBi metadata contract."""
    if not isinstance(alibi_slopes, torch.Tensor):
        raise TypeError("alibi_slopes must be a torch.Tensor or None")
    if alibi_slopes.layout != torch.strided:
        raise ValueError("alibi_slopes must use torch.strided layout")
    if alibi_slopes.dtype != torch.float32:
        raise ValueError("alibi_slopes must have dtype torch.float32")
    expected_shapes = ((spec.nheads_q,), (spec.batch, spec.nheads_q))
    if tuple(alibi_slopes.shape) not in expected_shapes:
        raise ValueError(
            "alibi_slopes must have shape [nheads_q] or [batch, nheads_q]; "
            f"expected {expected_shapes[0]} or {expected_shapes[1]}, got "
            f"{tuple(alibi_slopes.shape)}"
        )
    if not alibi_slopes.is_cuda:
        raise ValueError(
            f"alibi_slopes must be a CUDA tensor, got device {alibi_slopes.device}"
        )
    if alibi_slopes.device != q.device:
        raise ValueError("alibi_slopes must be on the same CUDA device as q, k, and v")
    if alibi_slopes.stride(-1) != 1:
        raise ValueError("alibi_slopes must be contiguous in its last dimension")


def _generic_dense_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
    alibi_slopes: torch.Tensor | None,
) -> torch.Tensor:
    """Adapt a validated dense batch to the generic packed Triton runtime."""
    total_q, total_k = _validate_generic_dense_layout(spec)

    # The dense inputs were validated as contiguous, so these views retain the
    # packed runtime's [total, heads, head_dim] layout without copying data.
    packed_q = q.view(total_q, spec.nheads_q, spec.head_dim)
    packed_k = k.view(total_k, spec.nheads_kv, spec.head_dim)
    packed_v = v.view(total_k, spec.nheads_kv, spec.head_dim)
    request_ids = torch.arange(
        spec.batch + 1, device=q.device, dtype=torch.int32
    )
    cu_seqlens_q = request_ids * spec.seqlen_q
    cu_seqlens_k = (
        cu_seqlens_q
        if spec.seqlen_q == spec.seqlen_k
        else request_ids * spec.seqlen_k
    )

    # Keep the Triton dependency lazy for callers that only inspect the
    # specialization manifest.
    from ._paged_attention import packed_attention

    packed_out = packed_attention(
        packed_q,
        packed_k,
        packed_v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=False,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_out, torch.Tensor):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention unexpectedly returned softmax LSE")
    return packed_out.view(
        spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim
    )


def is_shape_supported(
    shape: ShapeLike, dtype: torch.dtype = torch.bfloat16, causal: bool = False
) -> bool:
    """True when an accelerated kernel for this exact shape is checked in.

    This intentionally reports specialization availability, not whether the
    generic dense forward fallback can execute the shape.
    """
    return has_kernel(normalize_shape(shape, dtype, causal))


def is_varlen_shape_supported(
    shape: ShapeLike, dtype: torch.dtype = torch.bfloat16, causal: bool = False
) -> bool:
    """True when a packed-sequence kernel for this maximum shape is checked in."""
    return has_varlen_kernel(normalize_shape(shape, dtype, causal))


def is_paged_shape_supported(
    shape: ShapeLike,
    dtype: torch.dtype = torch.bfloat16,
    causal: bool = False,
    page_size: int = 16,
) -> bool:
    """True when vLLM paged attention has this maximum shape and page size."""
    return has_paged_kernel(normalize_shape(shape, dtype, causal), page_size)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``flash_attn.flash_attn_func``.

    Args:
        q: ``[batch, seqlen_q, nheads_q, head_dim]``, contiguous, fp16 or bf16.
        k: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        v: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        softmax_scale: defaults to ``1 / sqrt(head_dim)``.
        causal: bottom-right causal masking, including unequal sequence lengths.
        alibi_slopes: fp32 CUDA tensor shaped ``[nheads_q]`` or
            ``[batch, nheads_q]``. ALiBi calls are currently forward-only.
        return_attn_probs: for the three shipped bf16 Llama GQA decode
            profiles, return FlashAttention's diagnostic tuple. This is
            available only when no backward is needed and all options other
            than ``softmax_scale`` and the equivalent decode ``causal`` mode
            retain their defaults.
        shape: required. Either an :class:`AttnShape`, a 4-tuple
            ``(batch, seqlen, nheads, head_dim)``, or a 6-tuple
            ``(batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim)``.

    Returns:
        ``[batch, seqlen_q, nheads_q, head_dim]``. With the supported
        ``return_attn_probs=True`` subset, returns ``(out, softmax_lse,
        S_dmask)``. The LSE is fp32 with shape ``[batch, nheads_q, 1]`` and
        ``S_dmask`` is an empty input-dtype tensor, matching FlashAttention
        when dropout is zero.

    Unregistered fp16/bf16 shapes with ``head_dim <= 256`` and all ALiBi calls
    use a generic packed Triton forward kernel. Grad-enabled calls without
    ALiBi or a generated backward use PyTorch SDPA autograd.
    :func:`is_shape_supported` remains ``False`` for unregistered calls because
    it reports checked-in acceleration only.
    """
    _reject_unsupported(
        dropout_p,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
        allow_alibi=True,
        allow_return_attn_probs=True,
    )
    spec = normalize_shape(shape, q.dtype, causal)
    check_tensors(q, k, v, spec)
    if alibi_slopes is not None:
        _validate_dense_alibi_slopes(alibi_slopes, q, spec)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    scale = float(softmax_scale)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v)
    )
    if return_attn_probs:
        if needs_backward:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented for grad-enabled "
                "calls"
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented with ALiBi slopes"
            )
        if deterministic:
            raise NotImplementedError(
                "return_attn_probs=True requires deterministic=False"
            )
        if not _supports_diagnostic_return(spec):
            raise NotImplementedError(
                "return_attn_probs=True is implemented only for the three "
                "shipped batch-1, single-token bf16 Llama GQA decode profiles"
            )
        out, softmax_lse = lookup(spec)(
            q, k, v, scale, return_softmax_lse=True
        )
        return out, softmax_lse, q.new_empty((0,))
    if alibi_slopes is not None and torch.is_grad_enabled() and (
        needs_backward or alibi_slopes.requires_grad
    ):
        raise NotImplementedError(
            "ALiBi backward is not implemented; ALiBi calls are forward-only"
        )
    if needs_backward:
        if has_backward(spec):
            return attention_autograd(q, k, v, scale, spec)
        if deterministic:
            raise NotImplementedError(
                "deterministic=True is not supported by the PyTorch SDPA "
                "autograd fallback"
            )
        # Preserve the generic dense forward's documented shape envelope for
        # unregistered calls even though SDPA itself may accept more shapes.
        if not has_kernel(spec):
            _validate_generic_dense_layout(spec)
        return dense_attention_sdpa(q, k, v, scale, spec)
    if alibi_slopes is None and has_kernel(spec):
        return lookup(spec)(q, k, v, scale)
    return _generic_dense_forward(q, k, v, scale, spec, alibi_slopes)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    *,
    shape: ShapeLike,
) -> torch.Tensor:
    """Drop-in replacement for ``flash_attn.flash_attn_varlen_func``.

    ``q`` is ``[total_q, nheads_q, head_dim]``. Without ``block_table``,
    ``k``/``v`` are ``[total_k, nheads_kv, head_dim]``. With ``block_table``,
    the checked-in bf16 page-size-16 profiles are chunked prefill
    ``(2, 200, 320, 8, 2, 128)`` with ``causal=True`` and decode
    ``(4, 1, 1024, 8, 2, 128)`` with either causal flag. They accept caches
    shaped ``[num_blocks, 16, 2, 128]`` and derive each request's used cache
    length from adjacent ``cu_seqlens_k`` offsets without copying them to the
    host.
    The int32 CUDA cumulative-length tensors contain ``batch + 1`` offsets.
    ``shape`` uses the same forms as :func:`flash_attn_func`, but its sequence
    dimensions are the maximum query and key lengths rather than dense tensor
    dimensions.

    The packed token totals and individual sequence lengths may change between
    calls to the same specialization.  The batch size, maxima, head geometry,
    dtype, and causal mode must continue to match ``shape``.
    """
    del deterministic  # This option affects backward only.
    _reject_unsupported(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    if type(max_seqlen_q) is not int or type(max_seqlen_k) is not int:
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")

    spec = normalize_shape(shape, q.dtype, causal)
    if max_seqlen_q != spec.seqlen_q or max_seqlen_k != spec.seqlen_k:
        raise ValueError(
            "max_seqlen_q/max_seqlen_k must match the maximum sequence lengths "
            f"declared by shape ({spec.seqlen_q}, {spec.seqlen_k}); got "
            f"({max_seqlen_q}, {max_seqlen_k})"
        )

    if block_table is not None:
        page_size = check_paged_varlen_tensors(
            q, k, v, cu_seqlens_q, cu_seqlens_k, block_table, spec
        )
        _check_core_paged_varlen_spec(spec, page_size)
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q, k, v)
        ):
            raise NotImplementedError(
                "flash_attn_varlen_func with block_table is forward-only; "
                "no paged-cache backward kernel is checked in"
            )
        # For the decode profile, a bottom-right-aligned single-token query has
        # equivalent causal and non-causal results. The registry handles that
        # equivalence; chunked prefill still requires its generated causal mode.
        kernel = lookup_paged(spec, page_size)
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(spec.head_dim)
        # Keep sequence metadata on-device. This allocates the four per-request
        # used lengths directly on CUDA and remains safe for graph capture.
        seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        return kernel(
            q,
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            float(softmax_scale),
            causal,
        )

    check_varlen_tensors(q, k, v, cu_seqlens_q, cu_seqlens_k, spec)
    kernel = lookup_varlen(spec)
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v)
    ):
        raise NotImplementedError(
            "flash_attn_varlen_func is currently forward-only; no packed-sequence "
            "backward kernel is checked in"
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    return kernel(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        float(softmax_scale),
        causal,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor:
    """Run varlen self-attention on ``[total, 3, nheads, head_dim]`` QKV."""
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError(
            "qkv must have shape [total, 3, nheads, head_dim], "
            f"got {tuple(qkv.shape)}"
        )
    q, k, v = (qkv[:, index].contiguous() for index in range(3))
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor:
    """Run varlen attention with ``[total_k, 2, nheads_kv, head_dim]`` KV."""
    if kv.ndim != 4 or kv.shape[1] != 2:
        raise ValueError(
            "kv must have shape [total_k, 2, nheads_kv, head_dim], "
            f"got {tuple(kv.shape)}"
        )
    k, v = (kv[:, index].contiguous() for index in range(2))
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )


def _paged_kvcache_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    append_kv: bool,
    cache_seqlens: int | torch.Tensor | None,
    block_table: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
    return_softmax_lse: bool,
    shape: ShapeLike,
) -> torch.Tensor:
    """Adapt the exact read-only paged decode profile to core varlen."""
    if append_kv:
        raise NotImplementedError(
            "paged KV-cache updates are not implemented; paged caches are read-only"
        )
    if return_softmax_lse:
        raise NotImplementedError(
            "return_softmax_lse is not implemented for paged KV caches"
        )

    spec = normalize_shape(shape, q.dtype, causal)
    for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if cache.ndim != 4:
            raise ValueError(
                f"{name} must be a paged KV cache with rank 4, got "
                f"shape {tuple(cache.shape)}"
            )
    if k_cache.shape[1] != v_cache.shape[1]:
        raise ValueError(
            "k_cache and v_cache must have the same page size, got "
            f"{k_cache.shape[1]} and {v_cache.shape[1]}"
        )
    page_size = k_cache.shape[1]
    _check_core_paged_kvcache_spec(spec, page_size)

    expected_q = (spec.batch, 1, spec.nheads_q, spec.head_dim)
    if tuple(q.shape) != expected_q:
        raise ValueError(
            f"q has shape {tuple(q.shape)} but the declared paged KV-cache "
            f"shape requires {expected_q}"
        )
    if not isinstance(cache_seqlens, torch.Tensor):
        raise NotImplementedError(
            "paged KV caches require cache_seqlens as a CUDA int32 tensor "
            "with shape [batch]"
        )
    if cache_seqlens.layout != torch.strided:
        raise ValueError("cache_seqlens must use torch.strided layout")
    if tuple(cache_seqlens.shape) != (spec.batch,):
        raise ValueError(
            f"cache_seqlens must have shape ({spec.batch},), got "
            f"{tuple(cache_seqlens.shape)}"
        )
    if cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must have dtype torch.int32")
    if not cache_seqlens.is_cuda:
        raise ValueError(
            "cache_seqlens must be a CUDA tensor, got device "
            f"{cache_seqlens.device}"
        )
    if cache_seqlens.device != q.device:
        raise ValueError(
            "cache_seqlens must be on the same CUDA device as q, k_cache, "
            "and v_cache"
        )

    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k_cache, v_cache)
    ):
        raise NotImplementedError(
            "paged KV-cache calls do not support autograd; use inference mode "
            "or detach the inputs"
        )

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else float(softmax_scale)
    )
    packed_q = q.squeeze(1)
    cu_seqlens_q = torch.arange(
        spec.batch + 1, device=q.device, dtype=torch.int32
    )
    cu_seqlens_k = torch.cat(
        (
            cache_seqlens.new_zeros(1),
            cache_seqlens.cumsum(dim=0, dtype=torch.int32),
        )
    )
    packed_out = flash_attn_varlen_func(
        packed_q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=scale,
        causal=causal,
        block_table=block_table,
        shape=spec,
    )
    return packed_out.unsqueeze(1)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: torch.Tensor | None = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Read or append one token to a FlashAttention-style KV cache.

    The general supported path is a dense, contiguous cache and exactly one
    query token. As with every entry point in this package, ``shape`` is
    required and describes ``q`` plus the cache:
    ``(batch, 1, cache_len, nheads_q, nheads_kv, head_dim)``.

    One read-only paged specialization is also exposed: bf16
    ``(4, 1, 1024, 8, 2, 128)`` with page-size-16 caches, an int32 CUDA
    ``cache_seqlens`` tensor shaped ``[4]``, and ``block_table``. It routes
    through :func:`flash_attn_varlen_func` and supports ragged logical caches.
    Because this is single-token bottom-right decode, the default
    ``causal=False`` and ``causal=True`` modes are equivalent and both work.

    For dense caches, ``cache_seqlens`` may be omitted or supplied as a Python
    integer equal to the declared cache length for a read-only call. A paired,
    one-token ``k`` and ``v`` update is supported when the Python integer
    ``cache_seqlens`` is exactly one less than that length; the update is copied
    into the final cache slot before attention runs. On this dense path,
    ``return_softmax_lse=True`` returns ``(out, softmax_lse)`` with LSE shape
    ``[batch, nheads_q, 1]`` and fp32 dtype, matching FlashAttention. Cache
    tensors created in inference mode must also be updated in inference mode,
    and an append requires disjoint query, K-cache, and V-cache memory. Dense
    tensor-valued/partial lengths and multi-token updates fail explicitly. A
    paired ``rotary_cos``/``rotary_sin`` table may be supplied only for this
    final-slot append: it must cover the full head, use the default interleaved
    layout, and rotates both ``q`` and the appended ``k`` at ``cache_seqlens``.
    Read-only rotary calls and other paged profiles fail explicitly.
    """
    if (k is None) != (v is None):
        raise ValueError("k and v must be provided together when updating the KV cache")
    append_kv = k is not None
    has_rotary_metadata = rotary_cos is not None or rotary_sin is not None
    if has_rotary_metadata and not append_kv:
        raise NotImplementedError(
            "rotary embeddings are implemented only for a one-token KV-cache append"
        )
    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos and rotary_sin must be provided together")
    apply_rotary = rotary_cos is not None
    if apply_rotary and not rotary_interleaved:
        raise NotImplementedError(
            "non-interleaved rotary embeddings are not implemented; pass "
            "rotary_interleaved=True"
        )
    if cache_batch_idx is not None:
        raise NotImplementedError("cache_batch_idx is not implemented")
    if cache_leftpad is not None:
        raise NotImplementedError("cache_leftpad is not implemented")
    if num_splits != 0:
        raise NotImplementedError(
            "explicit num_splits is not implemented; pass num_splits=0"
        )
    # This flag changes only how rotary pairs are laid out, so it remains
    # irrelevant when rotary_cos/rotary_sin are absent.

    _reject_unsupported(0.0, window_size, softcap, alibi_slopes, False)
    if block_table is not None:
        return _paged_kvcache_forward(
            q,
            k_cache,
            v_cache,
            append_kv=append_kv,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            shape=shape,
        )

    spec = normalize_shape(shape, q.dtype, causal)
    if not spec.is_decode:
        raise NotImplementedError(
            "flash_attn_with_kvcache currently supports only seqlen_q=1 "
            "with a non-empty cache"
        )
    if isinstance(cache_seqlens, torch.Tensor):
        # Reading a CUDA tensor on the host would synchronize and break graph
        # capture, while a device assertion would poison the CUDA context for
        # an ordinary input error. Reject this unsupported dynamic/ragged form
        # recoverably before launching any CUDA work.
        raise NotImplementedError(
            "tensor-valued cache_seqlens are not implemented; pass a Python int"
        )
    if cache_seqlens is not None and type(cache_seqlens) is not int:
        raise TypeError("cache_seqlens must be a Python int, a torch.Tensor, or None")

    if append_kv:
        if cache_seqlens is None:
            raise ValueError(
                "cache_seqlens must be a Python int when updating the KV cache"
            )
        if cache_seqlens < 0 or cache_seqlens >= spec.seqlen_k:
            raise ValueError(
                f"cache_seqlens={cache_seqlens} is out of range for a cache "
                f"of length {spec.seqlen_k}"
            )
        if cache_seqlens + 1 != spec.seqlen_k:
            raise NotImplementedError(
                "only a one-token update that fills the final cache slot is "
                "implemented; cache_seqlens + 1 must equal the cache length "
                "declared by shape"
            )
    elif cache_seqlens is not None and cache_seqlens != spec.seqlen_k:
        raise NotImplementedError(
            "partial or ragged KV caches are not implemented for read-only "
            "calls; cache_seqlens must equal the cache length declared by shape"
        )

    # Dispatch directly after the KV-cache-specific checks. Routing back
    # through flash_attn_func would repeat normalization and validation on
    # every latency-sensitive decode step.
    check_tensors(q, k_cache, v_cache, spec)
    if k_cache.device != q.device or v_cache.device != q.device:
        raise ValueError("q, k_cache, and v_cache must be on the same CUDA device")
    if append_kv:
        assert k is not None and v is not None
        expected_update = (spec.batch, 1, spec.nheads_kv, spec.head_dim)
        for name, tensor in (("k", k), ("v", v)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim == 4 and tensor.shape[1] != 1:
                raise NotImplementedError(
                    "multi-token KV-cache updates are not implemented; "
                    f"{name} must contain exactly one token"
                )
            if tuple(tensor.shape) != expected_update:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)} but a one-token "
                    f"update for the declared shape requires {expected_update}"
                )
            if tensor.dtype != spec.dtype:
                raise ValueError(
                    f"{name} has dtype {tensor.dtype} but shape declares {spec.dtype}"
                )
            if not tensor.is_cuda:
                raise ValueError(
                    f"{name} must be a CUDA tensor, got device {tensor.device}"
                )
            if tensor.device != q.device:
                raise ValueError(
                    "q, k_cache, v_cache, and update k/v must be on the same "
                    "CUDA device"
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous in "
                    "[batch, 1, nheads_kv, head_dim] layout"
                )
        if _contiguous_tensors_overlap(k_cache, v_cache):
            raise ValueError("k_cache and v_cache must not overlap when updating")
        if _contiguous_tensors_overlap(q, k_cache) or _contiguous_tensors_overlap(
            q, v_cache
        ):
            raise ValueError("q must not overlap k_cache or v_cache when updating")
        if not torch.is_inference_mode_enabled() and (
            k_cache.is_inference() or v_cache.is_inference()
        ):
            # PyTorch mutates an inference tensor's data before raising when an
            # in-place operation is attempted outside inference mode. Reject
            # before either copy so K and V cannot become desynchronized.
            raise RuntimeError(
                "KV caches created in torch.inference_mode() must be updated "
                "while torch.inference_mode() is enabled"
            )
        if apply_rotary:
            assert rotary_cos is not None and rotary_sin is not None
            _validate_kvcache_rotary(rotary_cos, rotary_sin, q, spec)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    kernel = lookup(spec)
    scale = float(softmax_scale)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (q, k_cache, v_cache, k, v, rotary_cos, rotary_sin)
        if tensor is not None
    )
    if needs_backward:
        if append_kv:
            raise NotImplementedError("KV-cache updates do not support autograd")
        lookup_backward(spec)
        return attention_autograd(q, k_cache, v_cache, scale, spec)

    # All input, feature, dispatch, and autograd validation must precede both
    # writes so a rejected call cannot leave a half-updated cache.
    q_for_attention = q
    if append_kv:
        assert k is not None and v is not None and cache_seqlens is not None
        cache_tensors = (k_cache, v_cache)
        if apply_rotary:
            assert rotary_cos is not None and rotary_sin is not None
            update_k = _apply_interleaved_rotary(
                k, rotary_cos, rotary_sin, cache_seqlens
            )
            q_for_attention = _apply_interleaved_rotary(
                q, rotary_cos, rotary_sin, cache_seqlens
            )
        else:
            update_k = (
                k.clone()
                if any(_contiguous_tensors_overlap(k, cache) for cache in cache_tensors)
                else k
            )
        update_v = (
            v.clone()
            if any(_contiguous_tensors_overlap(v, cache) for cache in cache_tensors)
            else v
        )
        k_cache[:, cache_seqlens : cache_seqlens + 1].copy_(update_k)
        v_cache[:, cache_seqlens : cache_seqlens + 1].copy_(update_v)
    if return_softmax_lse:
        return kernel(
            q_for_attention,
            k_cache,
            v_cache,
            scale,
            return_softmax_lse=True,
        )
    return kernel(q_for_attention, k_cache, v_cache, scale)


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qkv`` is ``[batch, seqlen, 3, nheads, head_dim]``; see :func:`flash_attn_func`."""
    if qkv.ndim != 5:
        raise ValueError(
            "qkv must be rank 5 with shape "
            "[batch, seqlen, 3, nheads, head_dim]; "
            f"got rank {qkv.ndim} and shape {tuple(qkv.shape)}"
        )
    if qkv.shape[2] != 3:
        raise ValueError(
            "qkv packed axis (dimension 2) must contain exactly 3 entries "
            f"(Q, K, V); got {qkv.shape[2]} in shape {tuple(qkv.shape)}"
        )
    q, k, v = (qkv[:, :, i].contiguous() for i in range(3))
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``kv`` is ``[batch, seqlen_k, 2, nheads_kv, head_dim]``; see :func:`flash_attn_func`."""
    if kv.ndim != 5:
        raise ValueError(
            "kv must be rank 5 with shape "
            "[batch, seqlen_k, 2, nheads_kv, head_dim]; "
            f"got rank {kv.ndim} and shape {tuple(kv.shape)}"
        )
    if kv.shape[2] != 2:
        raise ValueError(
            "kv packed axis (dimension 2) must contain exactly 2 entries "
            f"(K, V); got {kv.shape[2]} in shape {tuple(kv.shape)}"
        )
    k, v = (kv[:, :, i].contiguous() for i in range(2))
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )
