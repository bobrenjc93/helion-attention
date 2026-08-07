"""Compatibility surface for :mod:`vllm.vllm_flash_attn`.

vLLM calls FlashAttention through a varlen API that does not carry
``helion_attention``'s explicit ``shape`` argument.  This module infers the
specialization from the packed tensors and maximum sequence lengths.  Calls
matching a checked-in varlen kernel use it; all other calls use a correctness
fallback implemented with PyTorch operations.

The fallback deliberately favors coverage and correctness over performance.
In particular, it supports vLLM's paged KV-cache layout and synchronizes once
to read the per-request sequence metadata before evaluating each request.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from . import flash_attn_varlen_func as _specialized_varlen_func
from ._registry import has_varlen_kernel
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


def _check_cumulative(
    name: str,
    cumulative: torch.Tensor,
    *,
    device: torch.device,
    expected_size: int | None,
    total: int,
) -> list[int]:
    if not isinstance(cumulative, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if cumulative.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if cumulative.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if cumulative.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if expected_size is not None and cumulative.numel() != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} offsets, got {cumulative.numel()}"
        )

    offsets = [int(value) for value in cumulative.detach().cpu().tolist()]
    if len(offsets) < 2:
        raise ValueError(f"{name} must contain at least two offsets")
    if offsets[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(stop < start for start, stop in zip(offsets, offsets[1:])):
        raise ValueError(f"{name} must be monotonically nondecreasing")
    if offsets[-1] != total:
        raise ValueError(
            f"{name} ends at {offsets[-1]}, but the packed tensor has {total} tokens"
        )
    return offsets


def _check_lengths(
    seqused_k: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
) -> list[int]:
    if not isinstance(seqused_k, torch.Tensor):
        raise TypeError("seqused_k must be a torch.Tensor")
    if tuple(seqused_k.shape) != (batch,):
        raise ValueError(
            f"seqused_k must have shape ({batch},), got {tuple(seqused_k.shape)}"
        )
    if seqused_k.dtype != torch.int32:
        raise ValueError("seqused_k must have dtype torch.int32")
    if seqused_k.device != device:
        raise ValueError(f"seqused_k must be on device {device}")
    lengths = [int(value) for value in seqused_k.detach().cpu().tolist()]
    if any(length < 0 for length in lengths):
        raise ValueError("seqused_k cannot contain negative lengths")
    return lengths


def _head_values(
    value: torch.Tensor | None,
    *,
    name: str,
    request: int,
    batch: int,
    target_heads: int,
    repeat_from: int | None = None,
) -> torch.Tensor | None:
    """Select one request's scalar-per-head values and expand GQA values."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")

    if value.ndim == 0:
        selected = value.reshape(1)
    elif value.ndim == 1:
        selected = value
    elif value.ndim == 2 and value.shape[0] in (1, batch):
        selected = value[0 if value.shape[0] == 1 else request]
    else:
        raise ValueError(
            f"{name} must be scalar, [heads], or [batch, heads]; got "
            f"{tuple(value.shape)}"
        )

    count = selected.numel()
    if count == 1:
        return selected.float().expand(target_heads)
    if count == target_heads:
        return selected.float()
    if repeat_from is not None and count == repeat_from and target_heads % count == 0:
        return selected.float().repeat_interleave(target_heads // count)
    raise ValueError(
        f"{name} supplies {count} values, but this call needs {target_heads} heads"
    )


def _copy_or_return(result: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return result
    out.copy_(result)
    return out


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
) -> torch.Tensor | None:
    """Use a checked-in kernel when every semantic and layout constraint fits."""
    if (
        cu_seqlens_k is None
        or seqused_k is not None
        or block_table is not None
        or k.ndim != 3
        or window_size != (-1, -1)
        or softcap > 0.0
        or alibi_slopes is not None
        or return_softmax_lse
        or any(item is not None for item in (q_descale, k_descale, v_descale, s_aux))
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


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Sequence[int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    out: torch.Tensor | None = None,
    scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    num_splits: int = 0,
    fa_version: int = 3,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run vLLM's packed or paged variable-length attention call.

    ``q`` uses packed ``[total_q, heads_q, head_dim]`` layout.  Non-paged
    ``k``/``v`` use the matching packed layout and ``cu_seqlens_k``.  Paged
    caches use ``[blocks, page_size, heads_kv, head_dim]`` together with
    ``block_table`` and ``seqused_k``.

    The fallback implements FlashAttention's bottom-right mask alignment,
    GQA/MQA, local windows, softcap, ALiBi, FP8 descales, attention sinks,
    optional LSE output, and ``out=`` semantics.
    """
    del scheduler_metadata, num_splits, fa_version  # Optimization selectors only.

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
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if q.shape[1] % k.shape[-2] != 0:
        raise ValueError("the number of query heads must be divisible by KV heads")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same device")
    if type(max_seqlen_q) is not int or type(max_seqlen_k) is not int:
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("maximum sequence lengths cannot be negative")
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

    real_window = _normalize_window(window_size)
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
    )
    if specialized is not None:
        return specialized

    q_offsets = _check_cumulative(
        "cu_seqlens_q",
        cu_seqlens_q,
        device=q.device,
        expected_size=batch + 1,
        total=q.shape[0],
    )
    query_lengths = [stop - start for start, stop in zip(q_offsets, q_offsets[1:])]
    if max(query_lengths, default=0) > max_seqlen_q:
        raise ValueError("max_seqlen_q is smaller than an actual query sequence")

    if cu_seqlens_k is not None and seqused_k is not None:
        raise ValueError("cu_seqlens_k and seqused_k cannot both be provided")
    if cu_seqlens_k is None and seqused_k is None:
        raise ValueError("either cu_seqlens_k or seqused_k must be provided")
    if block_table is not None and seqused_k is None:
        raise ValueError("seqused_k is required with a block_table")

    k_offsets: list[int] | None = None
    if cu_seqlens_k is not None:
        if k.ndim != 3 or block_table is not None:
            raise ValueError("cu_seqlens_k is only valid with non-paged rank-3 K/V")
        k_offsets = _check_cumulative(
            "cu_seqlens_k",
            cu_seqlens_k,
            device=q.device,
            expected_size=batch + 1,
            total=k.shape[0],
        )
        key_lengths = [stop - start for start, stop in zip(k_offsets, k_offsets[1:])]
    else:
        assert seqused_k is not None
        key_lengths = _check_lengths(seqused_k, batch=batch, device=q.device)

    if max(key_lengths, default=0) > max_seqlen_k:
        raise ValueError("max_seqlen_k is smaller than an actual key sequence")

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
    elif k.ndim == 4 and k.shape[0] < batch:
        raise ValueError("dense rank-4 K/V must have one cache row per request")
    elif k.ndim == 3 and k_offsets is None and sum(key_lengths) != k.shape[0]:
        raise ValueError(
            "rank-3 K/V with seqused_k must be tightly packed to sum(seqused_k)"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    output_dtype = q.dtype if out is None else out.dtype
    outputs: list[torch.Tensor] = []
    lse_parts: list[torch.Tensor] = []
    packed_k_start = 0
    page_size = k.shape[1] if k.ndim == 4 else 0
    left_window, right_window = real_window

    for request, (q_start, q_stop, key_length) in enumerate(
        zip(q_offsets, q_offsets[1:], key_lengths)
    ):
        query = q[q_start:q_stop]
        if block_table is not None:
            blocks_needed = (key_length + page_size - 1) // page_size
            if blocks_needed > block_table.shape[1]:
                raise ValueError(
                    "block_table does not contain enough pages for seqused_k"
                )
            block_ids = block_table[request, :blocks_needed].to(torch.long)
            key = k.index_select(0, block_ids).flatten(0, 1)[:key_length]
            value = v.index_select(0, block_ids).flatten(0, 1)[:key_length]
        elif k.ndim == 4:
            if key_length > k.shape[1]:
                raise ValueError("seqused_k exceeds the dense cache capacity")
            key = k[request, :key_length]
            value = v[request, :key_length]
        elif k_offsets is not None:
            key = k[k_offsets[request] : k_offsets[request + 1]]
            value = v[k_offsets[request] : k_offsets[request + 1]]
        else:
            key = k[packed_k_start : packed_k_start + key_length]
            value = v[packed_k_start : packed_k_start + key_length]
            packed_k_start += key_length

        nheads_q = q.shape[1]
        nheads_kv = k.shape[-2]
        query_f = query.float().transpose(0, 1)
        key_f = key.float().transpose(0, 1)
        value_f = value.float().transpose(0, 1)

        query_descale = _head_values(
            q_descale,
            name="q_descale",
            request=request,
            batch=batch,
            target_heads=nheads_q,
            repeat_from=nheads_kv,
        )
        key_descale = _head_values(
            k_descale,
            name="k_descale",
            request=request,
            batch=batch,
            target_heads=nheads_kv,
        )
        value_descale = _head_values(
            v_descale,
            name="v_descale",
            request=request,
            batch=batch,
            target_heads=nheads_kv,
        )
        if query_descale is not None:
            query_f = query_f * query_descale[:, None, None]
        if key_descale is not None:
            key_f = key_f * key_descale[:, None, None]
        if value_descale is not None:
            value_f = value_f * value_descale[:, None, None]

        group_size = nheads_q // nheads_kv
        if group_size != 1:
            key_f = key_f.repeat_interleave(group_size, dim=0)
            value_f = value_f.repeat_interleave(group_size, dim=0)

        scores = torch.matmul(query_f, key_f.transpose(-1, -2)) * scale
        if cap > 0.0:
            scores = cap * torch.tanh(scores / cap)

        seqlen_q = query.shape[0]
        rows = torch.arange(seqlen_q, device=q.device)[:, None]
        columns = torch.arange(key_length, device=q.device)[None, :]
        aligned_rows = rows + key_length - seqlen_q

        slopes = _head_values(
            alibi_slopes,
            name="alibi_slopes",
            request=request,
            batch=batch,
            target_heads=nheads_q,
        )
        if slopes is not None:
            distance = (aligned_rows - columns).abs().float()
            scores = scores - slopes[:, None, None] * distance[None]

        keep = torch.ones((seqlen_q, key_length), device=q.device, dtype=torch.bool)
        if causal:
            keep &= columns <= aligned_rows
        if left_window >= 0:
            keep &= columns >= aligned_rows - left_window
        if right_window >= 0:
            keep &= columns <= aligned_rows + right_window
        scores = scores.masked_fill(~keep[None], float("-inf"))

        sink = _head_values(
            s_aux,
            name="s_aux",
            request=request,
            batch=batch,
            target_heads=nheads_q,
        )
        if sink is not None:
            sink_scores = sink[:, None, None].expand(-1, seqlen_q, 1)
            scores_for_softmax = torch.cat((sink_scores, scores), dim=-1)
            lse = torch.logsumexp(scores_for_softmax, dim=-1)
            probabilities = torch.softmax(scores_for_softmax, dim=-1)[..., 1:]
        else:
            lse = torch.logsumexp(scores, dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        result = torch.matmul(probabilities, value_f).transpose(0, 1)
        outputs.append(result.to(output_dtype))
        lse_parts.append(lse)

    result = torch.cat(outputs, dim=0)
    result = _copy_or_return(result, out)
    if return_softmax_lse:
        return result, torch.cat(lse_parts, dim=1)
    return result
