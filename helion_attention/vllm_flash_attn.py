"""Compatibility surface for :mod:`vllm.vllm_flash_attn`.

vLLM calls FlashAttention through a varlen API that does not carry
``helion_attention``'s explicit ``shape`` argument.  This module infers the
specialization from the packed tensors and maximum sequence lengths.  Calls
matching a checked-in packed-varlen or paged-cache kernel use it, other packed
and paged CUDA calls use a generic single-launch Triton kernel, and remaining
calls use a correctness fallback implemented with PyTorch operations.

Both generic paths keep sequence metadata on-device and carry tiled online
softmax state rather than materializing a quadratic attention matrix.  The
PyTorch path deliberately favors coverage and correctness over performance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from . import flash_attn_varlen_func as _specialized_varlen_func
from ._registry import has_paged_kernel
from ._registry import has_varlen_kernel
from ._registry import lookup_paged
from ._shape import AttnShape

__all__ = [
    "compile_flash_attn_varlen_func_from_specs",
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]


# vLLM treats None as "this backend does not need an ahead-of-time compile
# pass".  Generated Helion kernels are already checked in, so that is exactly
# the contract we want.
compile_flash_attn_varlen_func_from_specs = None

_SUPPORTED_FA_VERSIONS = frozenset({2, 3})
_FP8_DTYPES: frozenset[torch.dtype] = frozenset(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if hasattr(torch, name)
)
_GENERIC_CUDA_DTYPES: frozenset[torch.dtype] = _FP8_DTYPES | {
    torch.float16,
    torch.bfloat16,
}

# Bound the largest temporary tensors used by the correctness fallback.  The
# online-softmax state is carried across key tiles, so these constants affect
# launch count but not numerical semantics.
_QUERY_TILE_SIZE = 16
_KEY_TILE_SIZE = 128


def _paged_attention(*args: Any, **kwargs: Any) -> Any:
    """Import Triton only when a CUDA paged call reaches the fast path."""
    from ._paged_attention import paged_attention

    return paged_attention(*args, **kwargs)


def _packed_attention(*args: Any, **kwargs: Any) -> Any:
    """Import Triton only when a CUDA packed call reaches the fast path."""
    from ._paged_attention import packed_attention

    return packed_attention(*args, **kwargs)


def is_fa_version_supported(fa_version: int) -> bool:
    """Whether this adapter can serve calls labelled with an FA version."""
    return fa_version in _SUPPORTED_FA_VERSIONS


def fa_version_unsupported_reason(fa_version: int) -> str | None:
    """Return ``None`` for supported versions and a diagnostic otherwise."""
    if is_fa_version_supported(fa_version):
        return None
    return (
        "helion-attention's vLLM compatibility module supports FA versions "
        f"2 and 3, not {fa_version}"
    )


def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    cache_seqlens: torch.Tensor,
    qkv_dtype: torch.dtype = torch.bfloat16,
    headdim_v: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k_new: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    page_size: int | None = None,
    max_seqlen_k_new: int = 0,
    causal: bool = False,
    window_size: Sequence[int] | None = None,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: bool | None = None,
    sm_margin: int = 0,
    **kwargs: Any,
) -> None:
    """Return no FA3 scheduler state.

    Scheduler metadata is only an optimization hint to the upstream FA3
    kernel.  Both checked-in Helion kernels and the PyTorch fallback select
    their own execution strategy, and :func:`flash_attn_varlen_func` accepts
    ``None`` for this argument.
    """
    return None


def _normalize_window(window_size: Sequence[int] | None) -> tuple[int, int]:
    if window_size is None:
        return (-1, -1)
    if len(window_size) != 2:
        raise ValueError("window_size must contain exactly two integers")
    left, right = window_size
    if type(left) is not int or type(right) is not int:
        raise TypeError("window_size must contain exactly two integers")
    return left, right


def _normalize_fa_window(
    window_size: tuple[int, int],
    *,
    fa_version: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[int, int]:
    """Apply the selected FlashAttention version's global-window sentinels."""
    left, right = window_size
    if fa_version == 2:
        left_limit = right_limit = max_seqlen_k
    else:
        # FA3's Hopper API uses the last valid position on each corresponding
        # axis, rather than FA2's key-length threshold for both endpoints.
        left_limit = max_seqlen_k - 1
        right_limit = max_seqlen_q - 1
    return (
        -1 if left >= left_limit else left,
        -1 if right >= right_limit else right,
    )


def _normalize_maximum(
    value: int | torch.Tensor,
    name: str,
    *,
    capture_fallback: int,
) -> int:
    """Accept vLLM's Python integers and zero-dimensional integer tensors."""
    if type(value) is int:
        result = value
    elif isinstance(value, torch.Tensor) and value.ndim == 0:
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} tensor must have dtype torch.int32 or torch.int64")
        if value.is_cuda and torch.cuda.is_current_stream_capturing():
            # CUDA graphs cannot synchronize a device scalar back to Python.
            # Static storage bounds are conservative launch bounds; actual
            # per-request lengths remain in device-side metadata.
            result = capture_fallback
        else:
            result = int(value.detach().item())
    else:
        raise TypeError(f"{name} must be a Python integer or scalar integer tensor")
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def _validate_cumulative(
    name: str,
    cumulative: torch.Tensor,
    *,
    device: torch.device,
    expected_size: int,
) -> None:
    """Validate only static metadata properties, without synchronizing CUDA."""
    if not isinstance(cumulative, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if cumulative.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if cumulative.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if cumulative.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if not cumulative.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if cumulative.numel() != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} offsets, got {cumulative.numel()}"
        )


def _validate_lengths(
    lengths: torch.Tensor,
    *,
    name: str = "seqused_k",
    batch: int,
    device: torch.device,
) -> None:
    if not isinstance(lengths, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(lengths.shape) != (batch,):
        raise ValueError(
            f"{name} must have shape ({batch},), got {tuple(lengths.shape)}"
        )
    if lengths.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if lengths.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if not lengths.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _head_values_for_requests(
    value: torch.Tensor | None,
    *,
    name: str,
    request_ids: torch.Tensor,
    batch: int,
    target_heads: int,
    repeat_from: int | None = None,
) -> torch.Tensor | None:
    """Select scalar-per-head values without reading request IDs on the host."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if value.device != request_ids.device:
        raise ValueError(f"{name} must be on device {request_ids.device}")

    if value.ndim == 0:
        selected = value.reshape(1, 1).expand(request_ids.numel(), -1)
    elif value.ndim == 1:
        selected = value[None].expand(request_ids.numel(), -1)
    elif value.ndim == 2 and value.shape[0] in (1, batch):
        selected = (
            value.expand(request_ids.numel(), -1)
            if value.shape[0] == 1
            else value.index_select(0, request_ids)
        )
    else:
        raise ValueError(
            f"{name} must be scalar, [heads], or [batch, heads]; got "
            f"{tuple(value.shape)}"
        )

    count = selected.shape[1]
    if count == 1:
        return selected.float().expand(-1, target_heads)
    if count == target_heads:
        return selected.float()
    if repeat_from is not None and count == repeat_from and target_heads % count == 0:
        return selected.float().repeat_interleave(target_heads // count, dim=1)
    raise ValueError(
        f"{name} supplies {count} values, but this call needs {target_heads} heads"
    )


def _copy_or_return(result: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return result
    out.copy_(result)
    return out


def _gather_kv_tile(
    k: torch.Tensor,
    v: torch.Tensor,
    request_ids: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    block_table: torch.Tensor | None,
    key_offsets: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one fixed-size key tile for every packed query row."""
    if block_table is not None:
        page_size = k.shape[1]
        logical_blocks = torch.div(key_positions, page_size, rounding_mode="floor")
        physical_blocks = block_table[
            request_ids[:, None], logical_blocks[None, :]
        ].long()
        physical_blocks = physical_blocks.clamp(0, k.shape[0] - 1)
        page_offsets = (key_positions % page_size)[None, :].expand(
            request_ids.numel(), -1
        )
        return (
            k[physical_blocks, page_offsets],
            v[physical_blocks, page_offsets],
        )

    if k.ndim == 4:
        dense_positions = key_positions.clamp(0, k.shape[1] - 1)
        return (
            k[request_ids[:, None], dense_positions[None, :]],
            v[request_ids[:, None], dense_positions[None, :]],
        )

    assert key_offsets is not None
    packed_positions = key_offsets.index_select(0, request_ids)[:, None]
    packed_positions = packed_positions + key_positions[None, :]
    packed_positions = packed_positions.clamp(0, k.shape[0] - 1)
    return k[packed_positions], v[packed_positions]


def _try_specialized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    block_table: torch.Tensor | None,
    return_softmax_lse: bool,
    out: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    s_aux: torch.Tensor | None,
    q_v: torch.Tensor | None,
    cp_world_size: int,
    cp_tot_seqused_k: torch.Tensor | None,
) -> torch.Tensor | None:
    """Use a checked-in kernel when every semantic and layout constraint fits."""
    if (
        cu_seqlens_k is None
        or seqused_k is not None
        or block_table is not None
        or k.ndim != 3
        or v.shape[-1] != q.shape[-1]
        or window_size != (-1, -1)
        or softcap > 0.0
        or alibi_slopes is not None
        or return_softmax_lse
        or q_v is not None
        or cp_world_size != 1
        or cp_tot_seqused_k is not None
        or s_aux is not None
        # vLLM supplies expanded scale buffers for every cache dtype.  The
        # buffers are inert for fp16/bf16 and only participate in FP8 attention.
        or (
            q.dtype in _FP8_DTYPES
            and any(item is not None for item in (q_descale, k_descale, v_descale))
        )
        or not q.is_cuda
        or not all(tensor.is_contiguous() for tensor in (q, k, v))
        or (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)))
    ):
        return None

    try:
        spec = AttnShape(
            batch=cu_seqlens_q.numel() - 1,
            seqlen_q=max_seqlen_q,
            seqlen_k=max_seqlen_k,
            nheads_q=q.shape[1],
            nheads_kv=k.shape[1],
            head_dim=q.shape[2],
            dtype=q.dtype,
            causal=causal,
        )
    except ValueError:
        return None
    if not has_varlen_kernel(spec):
        return None

    result = _specialized_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        shape=spec,
    )
    return _copy_or_return(result, out)


def _try_paged_specialized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_softmax_lse: bool,
    out: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    s_aux: torch.Tensor | None,
    q_v: torch.Tensor | None,
    cp_world_size: int,
    cp_tot_seqused_k: torch.Tensor | None,
    dynamic_max_seqlen_q: torch.Tensor | None,
    dynamic_max_seqlen_k: torch.Tensor | None,
) -> torch.Tensor | None:
    """Use a generated block-table kernel when its lean contract fits."""
    if (
        k.ndim != 4
        or v.shape[-1] != q.shape[-1]
        or window_size != (-1, -1)
        or softcap > 0.0
        or alibi_slopes is not None
        or return_softmax_lse
        or q_v is not None
        or cp_world_size != 1
        or cp_tot_seqused_k is not None
        or dynamic_max_seqlen_q is not None
        or dynamic_max_seqlen_k is not None
        or s_aux is not None
        # vLLM passes expanded Q/K/V scale buffers for every cache dtype.  They
        # are inert for fp16/bf16 and only participate in FP8 attention.
        or (
            q.dtype in _FP8_DTYPES
            and any(item is not None for item in (q_descale, k_descale, v_descale))
        )
        or block_table.dtype != torch.int32
        or not q.is_cuda
        or (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)))
    ):
        return None

    try:
        spec = AttnShape(
            batch=cu_seqlens_q.numel() - 1,
            seqlen_q=max_seqlen_q,
            seqlen_k=max_seqlen_k,
            nheads_q=q.shape[1],
            nheads_kv=k.shape[2],
            head_dim=q.shape[2],
            dtype=q.dtype,
            causal=causal,
        )
    except ValueError:
        return None
    page_size = k.shape[1]
    if not has_paged_kernel(spec, page_size):
        return None

    kernel = lookup_paged(spec, page_size)
    result = kernel(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return _copy_or_return(result, out)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int | torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int | torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    q_v: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Sequence[int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    out: torch.Tensor | None = None,
    scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    num_splits: int = 0,
    output_scale: torch.Tensor | None = None,
    fa_version: int = 2,
    s_aux: torch.Tensor | None = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: torch.Tensor | None = None,
    mask_mod: object | None = None,
    block_sparse_tensors: object | None = None,
    aux_tensors: Sequence[torch.Tensor] | None = None,
    aux_tensor_leading_dims: Sequence[int] | None = None,
    dynamic_causal: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run vLLM's packed or paged variable-length attention call.

    ``q`` uses packed ``[total_q, heads_q, head_dim]`` layout.  Non-paged
    ``k``/``v`` use the matching packed layout and ``cu_seqlens_k``.  Paged
    caches use ``[blocks, page_size, heads_kv, head_dim]`` together with
    ``block_table`` and ``seqused_k``.

    The generic implementations cover FlashAttention's bottom-right mask
    alignment, GQA/MQA, MLA's additional QV score term, context-parallel key
    positions, local windows, softcap, ALiBi, FP8 descales, attention sinks,
    optional LSE output, and ``out=`` semantics.
    """
    if not is_fa_version_supported(fa_version):
        reason = fa_version_unsupported_reason(fa_version)
        raise ValueError(f"unsupported fa_version={fa_version}: {reason}")
    if deterministic:
        raise NotImplementedError("deterministic=True is not supported")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs=True is not supported")
    if output_scale is not None:
        raise NotImplementedError("output_scale requires unsupported FA4")
    if (
        dynamic_causal is not None
        or mask_mod is not None
        or block_sparse_tensors is not None
        or aux_tensors is not None
        or aux_tensor_leading_dims is not None
    ):
        raise NotImplementedError(
            "dynamic_causal, mask_mod, block_sparse_tensors, aux_tensors, and "
            "aux_tensor_leading_dims require unsupported FA4"
        )
    if float(dropout_p) != 0.0:
        raise NotImplementedError(
            "helion-attention's vLLM adapter only supports dropout_p=0.0"
        )
    if type(cp_world_size) is not int or type(cp_rank) is not int:
        raise TypeError("cp_world_size and cp_rank must be Python integers")
    if cp_world_size <= 0:
        raise ValueError("cp_world_size must be positive")
    if cp_rank < 0 or cp_rank >= cp_world_size:
        raise ValueError("cp_rank must be in [0, cp_world_size)")
    if cp_world_size > 1 and fa_version != 3:
        raise NotImplementedError("context parallelism is only supported with FA3")
    if cp_world_size > 1 and cp_tot_seqused_k is None:
        raise ValueError(
            "cp_tot_seqused_k is required when cp_world_size is greater than one"
        )
    del scheduler_metadata, num_splits  # Optimization selectors only.

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if q.ndim != 3:
        raise ValueError(
            f"q must have shape [total_q, heads, dim], got {tuple(q.shape)}"
        )
    if k.ndim not in (3, 4) or v.ndim != k.ndim:
        raise ValueError("k and v must both use packed rank-3 or cache rank-4 layout")
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError("k and v must have matching token and head dimensions")
    if q.shape[1] == 0 or k.shape[-2] == 0:
        raise ValueError("q, k, and v must have at least one attention head")
    if q.shape[-1] == 0 or v.shape[-1] == 0:
        raise ValueError("q, k, and v head dimensions must be nonzero")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if q.shape[1] % k.shape[-2] != 0:
        raise ValueError("the number of query heads must be divisible by KV heads")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same device")
    if q_v is not None:
        if fa_version != 3:
            raise NotImplementedError("q_v is only supported with FA3")
        if not isinstance(q_v, torch.Tensor):
            raise TypeError("q_v must be a torch.Tensor or None")
        expected_q_v_shape = (q.shape[0], q.shape[1], v.shape[-1])
        if tuple(q_v.shape) != expected_q_v_shape:
            raise ValueError(
                f"q_v must have shape {expected_q_v_shape}, got {tuple(q_v.shape)}"
            )
        if q_v.device != q.device or q_v.dtype != q.dtype:
            raise ValueError("q_v must have the same device and dtype as q")
    original_max_seqlen_q = max_seqlen_q
    original_max_seqlen_k = max_seqlen_k
    max_k_storage = k.shape[0]
    if k.ndim == 4:
        max_k_storage = k.shape[1]
        if (
            isinstance(block_table, torch.Tensor)
            and block_table.ndim == 2
        ):
            max_k_storage = block_table.shape[1] * k.shape[1]
    max_seqlen_q = _normalize_maximum(
        max_seqlen_q, "max_seqlen_q", capture_fallback=q.shape[0]
    )
    max_seqlen_k = _normalize_maximum(
        max_seqlen_k, "max_seqlen_k", capture_fallback=max_k_storage
    )
    if not isinstance(cu_seqlens_q, torch.Tensor) or cu_seqlens_q.ndim != 1:
        raise ValueError("cu_seqlens_q must be a one-dimensional tensor")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.device != q.device:
        raise ValueError("cu_seqlens_q must be an int32 tensor on q's device")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must contain at least two offsets")

    batch = cu_seqlens_q.numel() - 1
    expected_out_shape = (q.shape[0], q.shape[1], v.shape[-1])
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor or None")
        if tuple(out.shape) != expected_out_shape:
            raise ValueError(
                f"out must have shape {expected_out_shape}, got {tuple(out.shape)}"
            )
        if out.device != q.device:
            raise ValueError("out must be on the same device as q")

    raw_window = _normalize_window(window_size)
    real_window = _normalize_fa_window(
        raw_window,
        fa_version=fa_version,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    dynamic_max_seqlen_q = (
        original_max_seqlen_q
        if isinstance(original_max_seqlen_q, torch.Tensor)
        and original_max_seqlen_q.is_cuda
        and original_max_seqlen_q.device == q.device
        else None
    )
    dynamic_max_seqlen_k = (
        original_max_seqlen_k
        if isinstance(original_max_seqlen_k, torch.Tensor)
        and original_max_seqlen_k.is_cuda
        and original_max_seqlen_k.device == q.device
        else None
    )
    kernel_window = (
        raw_window
        if dynamic_max_seqlen_q is not None or dynamic_max_seqlen_k is not None
        else real_window
    )
    if cp_world_size > 1 and real_window != (-1, -1):
        raise NotImplementedError(
            "local attention is not supported with context parallelism"
        )
    scale = None if softmax_scale is None else float(softmax_scale)
    cap = float(softcap)

    specialized = _try_specialized(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        bool(causal),
        real_window,
        cap,
        alibi_slopes,
        block_table,
        bool(return_softmax_lse),
        out,
        seqused_k,
        q_descale,
        k_descale,
        v_descale,
        s_aux,
        q_v,
        cp_world_size,
        cp_tot_seqused_k,
    )
    if specialized is not None:
        return specialized

    _validate_cumulative(
        "cu_seqlens_q", cu_seqlens_q, device=q.device, expected_size=batch + 1
    )
    if cu_seqlens_k is not None and seqused_k is not None:
        raise ValueError("cu_seqlens_k and seqused_k cannot both be provided")
    if cu_seqlens_k is None and seqused_k is None:
        raise ValueError("either cu_seqlens_k or seqused_k must be provided")
    if block_table is not None and seqused_k is None:
        raise ValueError("seqused_k is required with a block_table")

    key_offsets: torch.Tensor | None = None
    if cu_seqlens_k is not None:
        if k.ndim != 3 or block_table is not None:
            raise ValueError("cu_seqlens_k is only valid with non-paged rank-3 K/V")
        _validate_cumulative(
            "cu_seqlens_k", cu_seqlens_k, device=q.device, expected_size=batch + 1
        )
        key_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        key_offsets = cu_seqlens_k[:-1]
    else:
        assert seqused_k is not None
        _validate_lengths(seqused_k, batch=batch, device=q.device)
        key_lengths = seqused_k
        if k.ndim == 3:
            key_offsets = torch.cat(
                (torch.zeros_like(seqused_k[:1]), torch.cumsum(seqused_k, dim=0)[:-1])
            )

    if cp_tot_seqused_k is not None:
        _validate_lengths(
            cp_tot_seqused_k,
            name="cp_tot_seqused_k",
            batch=batch,
            device=q.device,
        )
    total_key_lengths = (
        key_lengths if cp_tot_seqused_k is None else cp_tot_seqused_k
    )

    if block_table is not None:
        if k.ndim != 4:
            raise ValueError(
                "paged K/V must have [blocks, page_size, heads, dim] layout"
            )
        if block_table.ndim != 2 or block_table.shape[0] < batch:
            raise ValueError("block_table must have one row per request")
        if block_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("block_table must have dtype torch.int32 or torch.int64")
        if block_table.device != q.device:
            raise ValueError("block_table must be on the same device as q")
        if max_seqlen_k > block_table.shape[1] * k.shape[1]:
            raise ValueError("block_table does not have capacity for max_seqlen_k")
        if max_seqlen_k > 0 and (k.shape[0] == 0 or k.shape[1] == 0):
            raise ValueError("paged K/V cache cannot be empty when max_seqlen_k > 0")
    elif k.ndim == 4 and k.shape[0] < batch:
        raise ValueError("dense rank-4 K/V must have one cache row per request")
    elif max_seqlen_k > 0 and (
        (k.ndim == 4 and k.shape[1] == 0) or (k.ndim == 3 and k.shape[0] == 0)
    ):
        raise ValueError("K/V storage cannot be empty when max_seqlen_k > 0")

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if block_table is not None:
        assert seqused_k is not None
        paged_specialized = _try_paged_specialized(
            q,
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            bool(causal),
            real_window,
            cap,
            alibi_slopes,
            bool(return_softmax_lse),
            out,
            q_descale,
            k_descale,
            v_descale,
            s_aux,
            q_v,
            cp_world_size,
            cp_tot_seqused_k,
            dynamic_max_seqlen_q,
            dynamic_max_seqlen_k,
        )
        if paged_specialized is not None:
            return paged_specialized

    differentiable_inputs = (q, k, v) if q_v is None else (q, k, v, q_v)
    use_generic_cuda = (
        q.is_cuda
        and q.dtype in _GENERIC_CUDA_DTYPES
        and not (
            torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in differentiable_inputs)
        )
    )
    left_window, right_window = real_window
    fa2_one_sided_alibi = (
        fa_version == 2
        and alibi_slopes is not None
        and (causal or right_window == 0)
    )
    if not fa2_one_sided_alibi:
        shift_fa2_lse = False
    elif left_window < 0:
        shift_fa2_lse = True
    elif dynamic_max_seqlen_k is not None:
        # The kernel must compare the real scalar on-device. During graph
        # capture the normalized launch bound may be the larger cache capacity.
        shift_fa2_lse = True
    else:
        shift_fa2_lse = left_window >= max_seqlen_k

    if use_generic_cuda and block_table is not None:
        assert seqused_k is not None
        return _paged_attention(
            q,
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dynamic_max_seqlen_q=dynamic_max_seqlen_q,
            dynamic_max_seqlen_k=dynamic_max_seqlen_k,
            softmax_scale=scale,
            causal=bool(causal),
            window_size=kernel_window,
            softcap=cap,
            alibi_slopes=alibi_slopes,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=s_aux,
            q_v=q_v,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqused_k=cp_tot_seqused_k,
            out=out,
            return_softmax_lse=bool(return_softmax_lse),
            shift_fa2_lse=shift_fa2_lse,
            fa_version=fa_version,
        )

    if use_generic_cuda and cu_seqlens_k is not None and k.ndim == 3:
        return _packed_attention(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dynamic_max_seqlen_q=dynamic_max_seqlen_q,
            dynamic_max_seqlen_k=dynamic_max_seqlen_k,
            softmax_scale=scale,
            causal=bool(causal),
            window_size=kernel_window,
            softcap=cap,
            alibi_slopes=alibi_slopes,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=s_aux,
            q_v=q_v,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqused_k=cp_tot_seqused_k,
            out=out,
            return_softmax_lse=bool(return_softmax_lse),
            shift_fa2_lse=shift_fa2_lse,
            fa_version=fa_version,
        )

    output_dtype = torch.bfloat16 if out is None and q.dtype in _FP8_DTYPES else q.dtype
    if out is not None:
        output_dtype = out.dtype
    outputs: list[torch.Tensor] = []
    lse_parts: list[torch.Tensor] = []
    left_window, right_window = real_window
    nheads_q = q.shape[1]
    nheads_kv = k.shape[-2]
    group_size = nheads_q // nheads_kv

    for q_start in range(0, q.shape[0], _QUERY_TILE_SIZE):
        q_stop = min(q_start + _QUERY_TILE_SIZE, q.shape[0])
        token_positions = torch.arange(q_start, q_stop, device=q.device)
        request_ids = torch.sum(
            token_positions[:, None] >= cu_seqlens_q[None, 1:], dim=1
        ).long()
        query_starts = cu_seqlens_q.index_select(0, request_ids)
        query_stops = cu_seqlens_q.index_select(0, request_ids + 1)
        query_lengths = query_stops - query_starts
        local_query_positions = token_positions - query_starts
        tile_key_lengths = key_lengths.index_select(0, request_ids)
        tile_total_key_lengths = total_key_lengths.index_select(0, request_ids)
        aligned_query_positions = (
            local_query_positions + tile_total_key_lengths - query_lengths
        )

        query_f = (
            q[q_start:q_stop]
            .float()
            .reshape(q_stop - q_start, nheads_kv, group_size, q.shape[-1])
        )
        query_v_f = None
        if q_v is not None:
            query_v_f = (
                q_v[q_start:q_stop]
                .float()
                .reshape(q_stop - q_start, nheads_kv, group_size, v.shape[-1])
            )
        query_descale = _head_values_for_requests(
            q_descale,
            name="q_descale",
            request_ids=request_ids,
            batch=batch,
            target_heads=nheads_q,
            repeat_from=nheads_kv,
        )
        key_descale = _head_values_for_requests(
            k_descale,
            name="k_descale",
            request_ids=request_ids,
            batch=batch,
            target_heads=nheads_kv,
        )
        value_descale = _head_values_for_requests(
            v_descale,
            name="v_descale",
            request_ids=request_ids,
            batch=batch,
            target_heads=nheads_kv,
        )
        slopes = _head_values_for_requests(
            alibi_slopes,
            name="alibi_slopes",
            request_ids=request_ids,
            batch=batch,
            target_heads=nheads_q,
        )
        sink = _head_values_for_requests(
            s_aux,
            name="s_aux",
            request_ids=request_ids,
            batch=batch,
            target_heads=nheads_q,
        )
        if query_descale is not None:
            query_f = query_f * query_descale.reshape(
                q_stop - q_start, nheads_kv, group_size, 1
            )
            if query_v_f is not None:
                query_v_f = query_v_f * query_descale.reshape(
                    q_stop - q_start, nheads_kv, group_size, 1
                )

        state_shape = (q_stop - q_start, nheads_kv, group_size)
        if sink is None:
            running_max = torch.full(
                state_shape, float("-inf"), device=q.device, dtype=torch.float32
            )
            running_sum = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
        else:
            running_max = sink.reshape(state_shape)
            running_sum = torch.ones(state_shape, device=q.device, dtype=torch.float32)
        accumulator = torch.zeros(
            (*state_shape, v.shape[-1]), device=q.device, dtype=torch.float32
        )

        for key_start in range(0, max_seqlen_k, _KEY_TILE_SIZE):
            key_stop = min(key_start + _KEY_TILE_SIZE, max_seqlen_k)
            key_positions = torch.arange(key_start, key_stop, device=q.device)
            key_tile, value_tile = _gather_kv_tile(
                k,
                v,
                request_ids,
                key_positions,
                block_table=block_table,
                key_offsets=key_offsets,
            )
            key_f = key_tile.float()
            value_f = value_tile.float()
            if key_descale is not None:
                key_f = key_f * key_descale[:, None, :, None]
            if value_descale is not None:
                value_f = value_f * value_descale[:, None, :, None]

            scores = torch.einsum("qhgd,qkhd->qhgk", query_f, key_f)
            if query_v_f is not None:
                scores = scores + torch.einsum(
                    "qhgd,qkhd->qhgk", query_v_f, value_f
                )
            scores = scores * scale
            if cap > 0.0:
                scores = cap * torch.tanh(scores / cap)

            columns = key_positions[None, :]
            global_columns = columns * cp_world_size + cp_rank
            keep = columns < tile_key_lengths[:, None]
            keep &= global_columns < tile_total_key_lengths[:, None]
            if causal:
                keep &= global_columns <= aligned_query_positions[:, None]
            if left_window >= 0:
                keep &= (
                    global_columns >= aligned_query_positions[:, None] - left_window
                )
            if right_window >= 0:
                keep &= (
                    global_columns <= aligned_query_positions[:, None] + right_window
                )

            # A zero attention weight does not neutralize NaN/Inf under IEEE
            # multiplication.  Clear masked values before the contraction so
            # unused ragged-cache slots cannot poison otherwise valid rows.
            value_f = value_f.masked_fill(~keep[:, :, None, None], 0.0)

            if slopes is not None:
                distance = (
                    aligned_query_positions[:, None] - global_columns
                ).abs().float()
                scores = (
                    scores
                    - slopes.reshape(q_stop - q_start, nheads_kv, group_size, 1)
                    * distance[:, None, None, :]
                )
            scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))

            tile_max = scores.amax(dim=-1)
            next_max = torch.maximum(running_max, tile_max)
            old_weight = torch.exp(running_max - next_max)
            old_weight = torch.nan_to_num(old_weight, nan=0.0)
            tile_weights = torch.exp(scores - next_max[..., None])
            tile_weights = torch.nan_to_num(tile_weights, nan=0.0)
            accumulator = accumulator * old_weight[..., None] + torch.einsum(
                "qhgk,qkhd->qhgd", tile_weights, value_f
            )
            running_sum = old_weight * running_sum + tile_weights.sum(dim=-1)
            running_max = next_max

        has_mass = running_sum > 0
        safe_sum = torch.where(has_mass, running_sum, torch.ones_like(running_sum))
        result_tile = accumulator / safe_sum[..., None]
        result_tile = result_tile.reshape(q_stop - q_start, nheads_q, v.shape[-1])
        lse_tile = torch.where(
            has_mass,
            running_max + torch.log(safe_sum),
            torch.full_like(
                running_max, float("inf") if fa_version == 2 else float("-inf")
            ),
        )
        fa2_global_left = left_window < 0 or left_window >= max_seqlen_k
        fa2_one_sided_alibi = fa2_global_left and (causal or right_window == 0)
        if fa_version == 2 and fa2_one_sided_alibi and slopes is not None:
            # FA2's global one-sided ALiBi kernel drops the row-constant
            # ``-slope * aligned_query_position`` while accumulating LSE.  It
            # has no effect on softmax probabilities, but callers merging
            # split attention states rely on this position-shifted LSE.
            lse_tile = lse_tile + slopes.reshape(state_shape) * (
                aligned_query_positions[:, None, None].float()
            )
        outputs.append(result_tile.to(output_dtype))
        lse_parts.append(lse_tile.reshape(q_stop - q_start, nheads_q).transpose(0, 1))

    if outputs:
        result = torch.cat(outputs, dim=0)
        softmax_lse = torch.cat(lse_parts, dim=1)
    else:
        result = torch.empty(expected_out_shape, device=q.device, dtype=output_dtype)
        softmax_lse = torch.empty((nheads_q, 0), device=q.device, dtype=torch.float32)
    result = _copy_or_return(result, out)
    if return_softmax_lse:
        return result, softmax_lse
    return result
