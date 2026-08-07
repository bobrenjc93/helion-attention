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

from ._registry import UnsupportedShapeError
from ._registry import available_shapes
from ._registry import lookup
from ._shape import AttnShape
from ._shape import ShapeLike
from ._shape import check_tensors
from ._shape import normalize_shape

__all__ = [
    "AttnShape",
    "UnsupportedShapeError",
    "available_shapes",
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "is_shape_supported",
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


def is_shape_supported(
    shape: ShapeLike, dtype: torch.dtype = torch.bfloat16, causal: bool = False
) -> bool:
    """True when a kernel for this exact shape is checked in."""
    try:
        lookup(normalize_shape(shape, dtype, causal))
    except UnsupportedShapeError:
        return False
    return True


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
    """Drop-in replacement for ``flash_attn.flash_attn_func`` (forward only).

    Args:
        q: ``[batch, seqlen_q, nheads_q, head_dim]``, contiguous, fp16 or bf16.
        k: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        v: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        softmax_scale: defaults to ``1 / sqrt(head_dim)``.
        causal: bottom-of-the-diagonal masking; requires ``seqlen_q == seqlen_k``.
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
    if causal and spec.seqlen_q != spec.seqlen_k:
        raise NotImplementedError(
            "causal attention with seqlen_q != seqlen_k is not implemented"
        )
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise NotImplementedError(
            "helion-attention ships forward-only kernels; run under torch.no_grad() "
            "or detach the inputs"
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    return lookup(spec)(q, k, v, float(softmax_scale))


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
