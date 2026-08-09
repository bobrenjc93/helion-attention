"""helion-attention: FlashAttention's API, served by Triton kernels.

The kernels in :mod:`helion_attention.kernels` were generated and autotuned by
`Helion <https://github.com/pytorch/helion>`_, but they are plain Triton by the
time they are checked in: importing this package pulls in ``torch`` and
``triton`` and nothing else.

Every entry point takes a required ``shape`` argument. Registered shapes use an
exact generated specialization, except three evidenced SM90 causal MHA/GQA
fast paths that use direct cuDNN SDPA; compatible unregistered dense shapes,
dense ALiBi calls, ALiBi on both shipped varlen profiles, and diagnostics on the
shipped causal varlen profile use a generic Triton forward kernel. The exposed
page-16 decode cache also uses the generic paged runtime when ALiBi is supplied.
Positive dropout on the shipped encoder-training profile, grad-enabled dense
calls without a generated backward, and the full-length noncausal varlen
encoder profile use PyTorch SDPA autograd. The explicit shape validates these
paths and makes specialization introspection independent of fallback coverage.
"""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

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
from ._sdpa import dense_attention_cudnn_default_scale
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
_CORE_PAGED_KVCACHE_SHAPES = frozenset(
    {_CORE_PAGED_CHUNKED_PREFILL_SHAPE, _CORE_PAGED_KVCACHE_SHAPE}
)
_CORE_PAGED_VARLEN_PAGE_SIZE = 16
_GENERIC_DENSE_MAX_HEAD_DIM = 256
_INT32_MAX = 2**31 - 1
_DROPOUT_SDPA_KEY = "b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal"
_CUDNN_SDPA_FAST_PATH_KEYS = frozenset(
    {
        "b1_sq4096_sk4096_hq32_hkv8_d128_bf16_causal",
        "b2_sq8192_sk8192_hq16_hkv16_d128_bf16_causal",
        "b4_sq4096_sk4096_hq32_hkv32_d128_bf16_causal",
    }
)
_VARLEN_ALIBI_KEYS = frozenset(
    {
        "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_causal",
        "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal",
    }
)
_VARLEN_DIAGNOSTIC_KEY = "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_causal"
_VARLEN_SDPA_BACKWARD_KEY = (
    "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal"
)
_DIAGNOSTIC_DECODE_PROFILES = frozenset(
    {
        (1, 1, cache_length, 32, 8, 128, torch.bfloat16)
        for cache_length in (1024, 4096, 16384)
    }
)


@lru_cache(maxsize=None)
def _is_sm90(device: torch.device) -> bool:
    """Whether ``device`` is exactly an SM90 CUDA GPU."""
    return torch.cuda.get_device_capability(device) == (9, 0)


def _reject_unsupported(
    dropout_p: float,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_attn_probs: bool,
    *,
    allow_dropout: bool = False,
    allow_alibi: bool = False,
    allow_return_attn_probs: bool = False,
) -> float:
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, Real):
        raise TypeError("dropout_p must be a real number")
    probability = float(dropout_p)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout_p must satisfy 0.0 <= dropout_p < 1.0")
    if probability != 0.0 and not allow_dropout:
        raise NotImplementedError("dropout is not implemented; pass dropout_p=0.0")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("sliding-window attention is not implemented")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not implemented")
    if alibi_slopes is not None and not allow_alibi:
        raise NotImplementedError("ALiBi slopes are not implemented")
    if probability != 0.0 and alibi_slopes is not None:
        raise NotImplementedError(
            "dropout combined with ALiBi slopes is not implemented"
        )
    if probability != 0.0 and return_attn_probs:
        raise NotImplementedError(
            "return_attn_probs=True is not implemented with dropout"
        )
    if return_attn_probs and not allow_return_attn_probs:
        raise NotImplementedError("return_attn_probs is not implemented")
    return probability


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
    """Validate the narrow rotary contract for a final-slot append."""
    for name, tensor in (("rotary_cos", rotary_cos), ("rotary_sin", rotary_sin)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use torch.strided layout")
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have shape [seqlen_ro, rotary_dim / 2], got "
                f"{tuple(tensor.shape)}"
            )

    if rotary_cos.shape != rotary_sin.shape:
        raise ValueError(
            "rotary_cos and rotary_sin must have the same shape, got "
            f"{tuple(rotary_cos.shape)} and {tuple(rotary_sin.shape)}"
        )
    rotary_dim = rotary_cos.shape[1] * 2
    supported_rotary_dims = {spec.head_dim}
    if spec.head_dim == 128:
        supported_rotary_dims.add(64)
    if rotary_dim not in supported_rotary_dims:
        expected = " or ".join(str(dim) for dim in sorted(supported_rotary_dims))
        raise NotImplementedError(
            "partial rotary dimensions are implemented only for rotary_dim=64 "
            "with head_dim=128; "
            f"rotary_dim must be {expected}, got {rotary_dim}"
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
    """Rotate adjacent prefix pairs and preserve the unrotated head tail."""
    rotary_dim = rotary_cos.shape[1] * 2
    rotary_prefix = tensor[..., :rotary_dim]
    pairs = rotary_prefix.float().reshape(
        *tensor.shape[:-1], rotary_dim // 2, 2
    )
    cos = rotary_cos[position].float()
    sin = rotary_sin[position].float()
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated_prefix = (
        torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        .flatten(-2)
        .to(dtype=tensor.dtype)
    )
    if rotary_dim == tensor.shape[-1]:
        return rotated_prefix
    return torch.cat((rotated_prefix, tensor[..., rotary_dim:]), dim=-1)


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
    """Require an exact read-only specialization exposed by the KV-cache API."""
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
        requested not in _CORE_PAGED_KVCACHE_SHAPES
        or (
            requested == _CORE_PAGED_CHUNKED_PREFILL_SHAPE
            and not spec.causal
        )
        or page_size != _CORE_PAGED_VARLEN_PAGE_SIZE
    ):
        raise UnsupportedShapeError(
            "flash_attn_with_kvcache with block_table currently supports only "
            "these bf16 profiles with page_size=16:\n"
            "    batch=2 seqlen_q=200 seqlen_k=320 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True\n"
            "    batch=4 seqlen_q=1 seqlen_k=1024 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True or causal=False\n"
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


def _validate_alibi_slopes(
    alibi_slopes: torch.Tensor,
    q: torch.Tensor,
    spec: AttnShape,
) -> None:
    """Validate FlashAttention's dense and varlen ALiBi metadata contract."""
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


def _check_varlen_alibi_spec(spec: AttnShape) -> None:
    """Restrict generic varlen ALiBi dispatch to its validated profiles."""
    requested = f"varlen_{spec.key}"
    if requested not in _VARLEN_ALIBI_KEYS:
        supported = ", ".join(sorted(_VARLEN_ALIBI_KEYS))
        raise NotImplementedError(
            "varlen ALiBi slopes are implemented only for "
            f"{supported}; got {requested}"
        )


def _supports_varlen_diagnostic_return(spec: AttnShape) -> bool:
    """Whether ``spec`` is the one LSE-capable core varlen profile."""
    return (
        f"varlen_{spec.key}" == _VARLEN_DIAGNOSTIC_KEY
        and has_varlen_kernel(spec)
    )


def _has_full_varlen_token_totals(
    q: torch.Tensor,
    k: torch.Tensor,
    spec: AttnShape,
) -> bool:
    """Whether valid packed inputs contain the full declared dense batch.

    Varlen callers are responsible for valid cumulative offsets and lengths no
    larger than the declared maxima. Under that contract, these full totals
    imply that every sequence has its canonical maximum length. Keeping this
    decision shape-only also makes the SDPA autograd path CUDA-graph capturable.
    """
    return (
        q.shape[0] == spec.batch * spec.seqlen_q
        and k.shape[0] == spec.batch * spec.seqlen_k
    )


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


def _generic_varlen_alibi_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
    alibi_slopes: torch.Tensor,
) -> torch.Tensor:
    """Run validated packed ALiBi inputs through the generic Triton runtime."""
    # Keep the Triton dependency lazy for callers that only inspect the
    # specialization manifest.
    from ._paged_attention import packed_attention

    packed_out = packed_attention(
        q,
        k,
        v,
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
    return packed_out


def _generic_varlen_diagnostic_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FA2-compatible diagnostics from the generic packed runtime."""
    # The generated specialization remains the default fast path, but it does
    # not materialize LSE. The generic packed kernel stores LSE directly in
    # FlashAttention's [heads, total_q] layout without unpacking ragged rows.
    from ._paged_attention import packed_attention

    packed_result = packed_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=True,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_result, tuple):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention did not return softmax LSE")
    out, softmax_lse = packed_result
    return out, softmax_lse, q.new_empty((0,))


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
        dropout_p: values in ``(0, 1)`` are supported only for the shipped
            noncausal bf16 ``(8, 512, 512, 16, 16, 64)`` profile.
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
    use a generic packed Triton forward kernel. The exact default-option,
    default-scale, no-backward causal bf16 ``(1, 4096, 4096, 32, 8, 128)``,
    ``(2, 8192, 8192, 16, 16, 128)``, and
    ``(4, 4096, 4096, 32, 32, 128)`` calls use direct cuDNN SDPA on SM90 when
    eligible, falling back to their generated kernels otherwise. Grad-enabled
    calls without ALiBi or a generated backward, plus supported positive-dropout
    calls, use PyTorch SDPA autograd.
    :func:`is_shape_supported` remains ``False`` for unregistered calls because
    it reports checked-in acceleration only.
    """
    dropout = _reject_unsupported(
        dropout_p,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
        allow_dropout=True,
        allow_alibi=True,
        allow_return_attn_probs=True,
    )
    spec = normalize_shape(shape, q.dtype, causal)
    if dropout != 0.0:
        if spec.key != _DROPOUT_SDPA_KEY:
            raise NotImplementedError(
                "dropout is implemented only for the shipped encoder-training "
                f"profile {_DROPOUT_SDPA_KEY}; got {spec.key}"
            )
        if deterministic:
            raise NotImplementedError(
                "deterministic=True is not supported with dropout"
            )
    check_tensors(q, k, v, spec)
    if alibi_slopes is not None:
        _validate_alibi_slopes(alibi_slopes, q, spec)
    default_softmax_scale = softmax_scale is None
    if default_softmax_scale:
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
    if dropout != 0.0:
        return dense_attention_sdpa(q, k, v, scale, spec, dropout)
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
    if (
        default_softmax_scale
        and not deterministic
        and alibi_slopes is None
        and spec.key in _CUDNN_SDPA_FAST_PATH_KEYS
        and _is_sm90(q.device)
    ):
        cudnn_out = dense_attention_cudnn_default_scale(q, k, v)
        if cudnn_out is not None:
            return cudnn_out
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    dtype, and causal mode must continue to match ``shape``. Forward-only
    ALiBi is supported for both causal modes of the
    ``(8, 512, 512, 16, 16, 64)`` bf16 profile, with fp32 slopes shaped
    ``[16]`` or ``[8, 16]``.

    The causal version of that profile also supports
    ``return_attn_probs=True`` when no backward is needed, ``causal=True``, and
    all options other than ``softmax_scale`` retain their defaults. It returns
    ``(out, softmax_lse, S_dmask)`` with fp32 LSE shaped ``[nheads_q, total_q]``
    and an empty bf16 ``S_dmask``, matching FlashAttention 2 when dropout is
    zero.

    The noncausal version supports zero-dropout backward when all eight query
    and key sequences have length 512. That exact full-length layout is
    reshaped to dense BSHD and uses PyTorch SDPA autograd. Ragged, causal,
    deterministic, paged, ALiBi, and diagnostic varlen backward remain
    unsupported. Calls that do not need backward retain the generated packed
    kernel, including full-length calls.
    """
    _reject_unsupported(
        dropout_p,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
        allow_alibi=block_table is None,
        allow_return_attn_probs=True,
    )
    if type(max_seqlen_q) is not int or type(max_seqlen_k) is not int:
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")

    spec = normalize_shape(shape, q.dtype, causal)
    if max_seqlen_q != spec.seqlen_q or max_seqlen_k != spec.seqlen_k:
        raise ValueError(
            "max_seqlen_q/max_seqlen_k must match the maximum sequence lengths "
            f"declared by shape ({spec.seqlen_q}, {spec.seqlen_k}); got "
            f"({max_seqlen_q}, {max_seqlen_k})"
        )

    if return_attn_probs:
        if block_table is not None:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented with block_table"
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented with ALiBi slopes"
            )
        if deterministic:
            raise NotImplementedError(
                "return_attn_probs=True requires deterministic=False"
            )
        if not _supports_varlen_diagnostic_return(spec):
            raise NotImplementedError(
                "return_attn_probs=True is implemented only for "
                f"{_VARLEN_DIAGNOSTIC_KEY}"
            )

    if alibi_slopes is not None:
        _check_varlen_alibi_spec(spec)

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
    if alibi_slopes is not None:
        _validate_alibi_slopes(alibi_slopes, q, spec)
    grad_tensors = (
        (q, k, v, alibi_slopes)
        if alibi_slopes is not None
        else (q, k, v)
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in grad_tensors
    )
    if needs_backward:
        if return_attn_probs:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented for grad-enabled "
                "calls"
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "ALiBi backward is not implemented; varlen ALiBi calls are "
                "forward-only"
            )
        if f"varlen_{spec.key}" != _VARLEN_SDPA_BACKWARD_KEY:
            raise NotImplementedError(
                "varlen backward is implemented only for "
                f"{_VARLEN_SDPA_BACKWARD_KEY} with eight full-length "
                "sequences"
            )
        if deterministic:
            raise NotImplementedError(
                "deterministic=True is not supported by the varlen PyTorch "
                "SDPA autograd fallback"
            )
        if not _has_full_varlen_token_totals(q, k, spec):
            raise NotImplementedError(
                "varlen backward requires all eight query and key sequences "
                "to have the canonical length 512; partial or ragged batches "
                "remain forward-only"
            )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(spec.head_dim)
        dense_q = q.reshape(
            spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim
        )
        dense_k = k.reshape(
            spec.batch, spec.seqlen_k, spec.nheads_kv, spec.head_dim
        )
        dense_v = v.reshape(
            spec.batch, spec.seqlen_k, spec.nheads_kv, spec.head_dim
        )
        dense_out = dense_attention_sdpa(
            dense_q, dense_k, dense_v, float(softmax_scale), spec
        )
        return dense_out.reshape(q.shape)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    scale = float(softmax_scale)
    if return_attn_probs:
        return _generic_varlen_diagnostic_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            spec,
        )
    if alibi_slopes is not None:
        return _generic_varlen_alibi_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            spec,
            alibi_slopes,
        )
    # Keep ordinary slope-free calls on the generated specialization.
    kernel = lookup_varlen(spec)
    return kernel(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    alibi_slopes: torch.Tensor | None,
    return_softmax_lse: bool,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Adapt exact read-only dense-query paged profiles to core varlen."""
    if append_kv:
        raise NotImplementedError(
            "paged KV-cache updates are not implemented; paged caches are read-only"
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
    requested = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    if return_softmax_lse and requested != _CORE_PAGED_KVCACHE_SHAPE:
        raise NotImplementedError(
            "return_softmax_lse=True for paged KV caches is implemented only "
            "for the bf16 page-size-16 batch=4 seqlen_q=1 seqlen_k=1024 "
            "nheads=8 (GQA 8:2) head_dim=128 decode profile"
        )
    if alibi_slopes is not None:
        if requested != _CORE_PAGED_KVCACHE_SHAPE:
            raise NotImplementedError(
                "paged KV-cache ALiBi is implemented only for the bf16 "
                "page-size-16 batch=4 seqlen_q=1 seqlen_k=1024 nheads=8 "
                "(GQA 8:2) head_dim=128 decode profile"
            )
        if return_softmax_lse:
            raise NotImplementedError(
                "return_softmax_lse=True is not implemented with paged "
                "KV-cache ALiBi"
            )

    expected_q = (
        spec.batch,
        spec.seqlen_q,
        spec.nheads_q,
        spec.head_dim,
    )
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
    if alibi_slopes is not None:
        _validate_alibi_slopes(alibi_slopes, q, spec)

    grad_tensors = (
        (q, k_cache, v_cache, alibi_slopes)
        if alibi_slopes is not None
        else (q, k_cache, v_cache)
    )
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in grad_tensors
    ):
        if alibi_slopes is not None:
            raise NotImplementedError(
                "ALiBi backward is not implemented; paged KV-cache ALiBi "
                "calls are forward-only"
            )
        raise NotImplementedError(
            "paged KV-cache calls do not support autograd; use inference mode "
            "or detach the inputs"
        )

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else float(softmax_scale)
    )
    packed_q = q.flatten(0, 1)
    cu_seqlens_q = torch.arange(
        0,
        (spec.batch + 1) * spec.seqlen_q,
        spec.seqlen_q,
        device=q.device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.cat(
        (
            cache_seqlens.new_zeros(1),
            cache_seqlens.cumsum(dim=0, dtype=torch.int32),
        )
    )
    if return_softmax_lse or alibi_slopes is not None:
        # The generated paged specialization intentionally stays on the lean
        # slope-free output-only contract. Reuse the vLLM-compatible
        # single-launch runtime for ALiBi or the diagnostic return it already
        # computes.
        check_paged_varlen_tensors(
            packed_q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            spec,
        )
        from ._paged_attention import paged_attention

        # The generic kernel indexes per-request lengths as a flat array and
        # therefore requires unit-stride storage. Preserve the public adapter's
        # support for valid strided metadata by normalizing only this argument.
        seqused_k = cache_seqlens.contiguous()
        result = paged_attention(
            packed_q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q=spec.seqlen_q,
            max_seqlen_k=spec.seqlen_k,
            dynamic_max_seqlen_q=None,
            dynamic_max_seqlen_k=None,
            softmax_scale=scale,
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
            return_softmax_lse=return_softmax_lse,
            shift_fa2_lse=False,
            fa_version=2,
        )
        if return_softmax_lse:
            if not isinstance(result, tuple):  # pragma: no cover - contract guard
                raise RuntimeError("generic paged attention did not return softmax LSE")
            packed_out, packed_lse = result
            softmax_lse = (
                packed_lse.reshape(spec.nheads_q, spec.batch, spec.seqlen_q)
                .permute(1, 0, 2)
                .contiguous()
            )
            return packed_out.reshape(expected_q), softmax_lse
        if not isinstance(result, torch.Tensor):  # pragma: no cover - contract guard
            raise RuntimeError(
                "generic paged attention unexpectedly returned softmax LSE"
            )
        return result.reshape(expected_q)

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
    return packed_out.reshape(expected_q)


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
    """Read from, or append one token to, a FlashAttention-style KV cache.

    The general supported path is a dense, contiguous cache and exactly one
    query token. As with every entry point in this package, ``shape`` is
    required and describes ``q`` plus the cache:
    ``(batch, 1, cache_len, nheads_q, nheads_kv, head_dim)``.

    Two read-only paged specializations are also exposed with page-size-16
    caches, an int32 CUDA ``cache_seqlens`` tensor shaped ``[batch]``, and
    ``block_table``. Bf16 ``(2, 200, 320, 8, 2, 128)`` supports only
    ``causal=True`` and uses bottom-right causal alignment for chunked prefill.
    Bf16 ``(4, 1, 1024, 8, 2, 128)`` supports both causal modes, which are
    equivalent for single-token bottom-right decode. Slope-free output-only
    calls route through :func:`flash_attn_varlen_func`; both profiles support
    ragged logical caches. The decode profile additionally accepts forward-only
    fp32 ALiBi slopes shaped ``[8]`` or ``[4, 8]`` through the generic paged
    runtime.

    For dense caches, ``cache_seqlens`` may be omitted or supplied as a Python
    integer equal to the declared cache length for a read-only call. A paired,
    one-token ``k`` and ``v`` update is supported when the Python integer
    ``cache_seqlens`` is exactly one less than that length; the update is copied
    into the final cache slot before attention runs. On this dense path,
    ``return_softmax_lse=True`` returns ``(out, softmax_lse)`` with LSE shape
    ``[batch, nheads_q, 1]`` and fp32 dtype, matching FlashAttention. The paged
    decode profile supports the same return through the generic single-launch
    paged runtime; slope-free output-only calls retain the generated
    specialization. ALiBi cannot be combined with that LSE return. Cache tensors
    created in inference mode must also be updated in inference mode, and an
    append requires disjoint query, K-cache, and V-cache memory. Dense
    tensor-valued/partial lengths and multi-token updates fail explicitly. A
    paired ``rotary_cos``/``rotary_sin`` table may be supplied only for this
    final-slot append: it may cover the full head or the first 64 dimensions of
    a 128-dimensional head, must use the default interleaved layout, and rotates
    both ``q`` and the appended ``k`` at ``cache_seqlens``. Read-only rotary
    calls and other paged profiles fail explicitly. Paged updates, rotary, and
    autograd are unsupported for both paged profiles. Paged ALiBi is unsupported
    for chunked prefill, updates, LSE returns, other profiles, and other page
    sizes. Paged softmax LSE is unsupported for chunked prefill, other profiles,
    and other page sizes.
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

    _reject_unsupported(
        0.0,
        window_size,
        softcap,
        alibi_slopes,
        False,
        allow_alibi=block_table is not None,
    )
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
            alibi_slopes=alibi_slopes,
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
