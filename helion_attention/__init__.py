"""helion-attention: FlashAttention's API, served by checked-in Triton kernels.

The kernels in :mod:`helion_attention.kernels` were generated and autotuned by
`Helion <https://github.com/pytorch/helion>`_, but they are plain Triton by the
time they are checked in: importing this package pulls in ``torch`` and
``triton`` and nothing else.

Every entry point takes a required ``shape`` argument. Helion only wins against
FlashAttention when a kernel is specialized to one exact problem size, so the
shape is part of the call contract rather than something discovered from the
tensors at runtime.
"""

from __future__ import annotations

import math

import torch

from ._autograd import attention_autograd
from ._registry import UnsupportedShapeError
from ._registry import available_paged_shapes
from ._registry import available_shapes
from ._registry import available_varlen_shapes
from ._registry import has_kernel
from ._registry import has_paged_kernel
from ._registry import has_varlen_kernel
from ._registry import lookup
from ._registry import lookup_backward
from ._registry import lookup_varlen
from ._shape import AttnShape
from ._shape import ShapeLike
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


def _reject_unsupported(
    dropout_p: float,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_attn_probs: bool,
) -> None:
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not implemented; pass dropout_p=0.0")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("sliding-window attention is not implemented")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not implemented")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi slopes are not implemented")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not implemented")


def _contiguous_tensors_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Whether two validated contiguous tensors share any device memory."""
    first_start = first.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_start = second.data_ptr()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def is_shape_supported(
    shape: ShapeLike, dtype: torch.dtype = torch.bfloat16, causal: bool = False
) -> bool:
    """True when a kernel for this exact shape is checked in."""
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
) -> torch.Tensor:
    """Drop-in replacement for ``flash_attn.flash_attn_func``.

    Args:
        q: ``[batch, seqlen_q, nheads_q, head_dim]``, contiguous, fp16 or bf16.
        k: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        v: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        softmax_scale: defaults to ``1 / sqrt(head_dim)``.
        causal: bottom-right causal masking, including unequal sequence lengths.
        shape: required. Either an :class:`AttnShape`, a 4-tuple
            ``(batch, seqlen, nheads, head_dim)``, or a 6-tuple
            ``(batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim)``.

    Returns:
        ``[batch, seqlen_q, nheads_q, head_dim]``.

    Raises:
        UnsupportedShapeError: no kernel is checked in for this shape.
    """
    _reject_unsupported(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    spec = normalize_shape(shape, q.dtype, causal)
    check_tensors(q, k, v, spec)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    kernel = lookup(spec)
    scale = float(softmax_scale)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v)
    )
    if needs_backward:
        # Resolve this before launching the forward so unsupported training
        # shapes fail at the call site rather than later during loss.backward().
        lookup_backward(spec)
        return attention_autograd(q, k, v, scale, spec)
    return kernel(q, k, v, scale)


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

    ``q`` is ``[total_q, nheads_q, head_dim]`` and ``k``/``v`` are
    ``[total_k, nheads_kv, head_dim]``.  The int32 CUDA cumulative-length
    tensors contain ``batch + 1`` offsets.  ``shape`` uses the same forms as
    :func:`flash_attn_func`, but its sequence dimensions are the maximum query
    and key lengths rather than dense tensor dimensions.

    The packed token totals and individual sequence lengths may change between
    calls to the same specialization.  The batch size, maxima, head geometry,
    dtype, and causal mode must continue to match ``shape``.
    """
    del deterministic  # This option affects backward only.
    if block_table is not None:
        raise NotImplementedError("paged KV block tables are not implemented")
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
    """Read or append one token to a dense FlashAttention-style KV cache.

    The supported path is a dense, contiguous cache and exactly one query
    token. As with every entry point in this package, ``shape`` is required and
    describes ``q`` plus the cache:
    ``(batch, 1, cache_len, nheads_q, nheads_kv, head_dim)``.

    ``cache_seqlens`` may be omitted or supplied as a Python integer equal to
    the declared cache length for a read-only call. A paired, one-token ``k``
    and ``v`` update is supported when the Python integer ``cache_seqlens`` is
    exactly one less than that length; the update is copied into the final
    cache slot before attention runs. If ``return_softmax_lse=True``, the result
    is ``(out, softmax_lse)`` with LSE shape ``[batch, nheads_q, 1]`` and fp32
    dtype, matching FlashAttention. Cache tensors created in inference mode
    must also be updated in inference mode, and an append requires disjoint
    query, K-cache, and V-cache memory. Tensor-valued/ragged lengths,
    multi-token updates, rotary embeddings, and paged caches fail explicitly.
    """
    if (k is None) != (v is None):
        raise ValueError("k and v must be provided together when updating the KV cache")
    append_kv = k is not None
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError(
            "rotary embeddings in the KV-cache entry point are not implemented"
        )
    if cache_batch_idx is not None:
        raise NotImplementedError("cache_batch_idx is not implemented")
    if cache_leftpad is not None:
        raise NotImplementedError("cache_leftpad is not implemented")
    if block_table is not None:
        raise NotImplementedError("paged KV caches are not implemented")
    if num_splits != 0:
        raise NotImplementedError(
            "explicit num_splits is not implemented; pass num_splits=0"
        )
    # This flag changes only how rotary pairs are laid out, so it is irrelevant
    # when rotary_cos/rotary_sin are absent. Keep it in the compatible signature.
    del rotary_interleaved

    _reject_unsupported(0.0, window_size, softcap, alibi_slopes, False)
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
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    kernel = lookup(spec)
    scale = float(softmax_scale)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (q, k_cache, v_cache, k, v)
        if tensor is not None
    )
    if needs_backward:
        if append_kv:
            raise NotImplementedError("KV-cache updates do not support autograd")
        lookup_backward(spec)
        return attention_autograd(q, k_cache, v_cache, scale, spec)

    # All input, feature, dispatch, and autograd validation must precede both
    # writes so a rejected call cannot leave a half-updated cache.
    if append_kv:
        assert k is not None and v is not None and cache_seqlens is not None
        cache_tensors = (k_cache, v_cache)
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
            q,
            k_cache,
            v_cache,
            scale,
            return_softmax_lse=True,
        )
    return kernel(q, k_cache, v_cache, scale)


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
) -> torch.Tensor:
    """``qkv`` is ``[batch, seqlen, 3, nheads, head_dim]``; see :func:`flash_attn_func`."""
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
) -> torch.Tensor:
    """``kv`` is ``[batch, seqlen_k, 2, nheads_kv, head_dim]``; see :func:`flash_attn_func`."""
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
