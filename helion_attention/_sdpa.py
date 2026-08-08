"""PyTorch SDPA helpers shared by runtime fallbacks and benchmarks."""

from __future__ import annotations

from typing import TypeVar

import torch
from torch.nn.attention.bias import causal_lower_right

from ._shape import AttnShape

_Mask = TypeVar("_Mask")


def sdpa_causal_options(
    spec: AttnShape, causal_mask: _Mask | None
) -> tuple[_Mask | None, bool]:
    """Choose fused-SDPA mask arguments without changing causal semantics."""
    is_causal = spec.causal and spec.seqlen_q == spec.seqlen_k
    mask = (
        causal_mask
        if spec.causal and not (is_causal or spec.is_decode)
        else None
    )
    return mask, is_causal


def dense_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run a validated dense call through PyTorch's native SDPA autograd."""
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    enable_gqa = spec.nheads_q != spec.nheads_kv

    # FlashAttention returns the input dtype even under cross-dtype autocast.
    # SDPA is autocast-eligible, so keep its native fp16/bf16 contract explicit.
    with torch.autocast(device_type=q.device.type, enabled=False):
        if spec.causal and spec.seqlen_q > spec.seqlen_k:
            # Bottom-right alignment leaves the first Sq-Sk query rows fully
            # masked. PyTorch's lower-right bias warns that those rows may be
            # NaN on some backends. The remaining Sk rows are exactly ordinary
            # equal-length causal attention, so compute that fused subproblem
            # and prepend constant zeros without allocating an Sq x Sk mask.
            masked_rows = spec.seqlen_q - spec.seqlen_k
            visible_out = torch.nn.functional.scaled_dot_product_attention(
                query[:, :, masked_rows:],
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
                scale=softmax_scale,
                enable_gqa=enable_gqa,
            )
            out = torch.cat(
                (torch.zeros_like(query[:, :, :masked_rows]), visible_out),
                dim=2,
            )
        else:
            causal_bias = (
                causal_lower_right(spec.seqlen_q, spec.seqlen_k)
                if spec.causal
                and spec.seqlen_q != spec.seqlen_k
                and not spec.is_decode
                else None
            )
            causal_bias, is_causal = sdpa_causal_options(spec, causal_bias)
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_bias,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=softmax_scale,
                enable_gqa=enable_gqa,
            )
    return out.transpose(1, 2).contiguous()


def _varlen_layout(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build capture-safe lengths, validity, and packed indices on CUDA."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    positions = torch.arange(max_seqlen, device=cu_seqlens.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    indices = cu_seqlens[:-1].to(torch.int64).unsqueeze(1) + positions.unsqueeze(0)
    indices = torch.where(valid, indices, torch.full_like(indices, total_tokens))
    return lengths, valid, indices


def _pad_varlen_tensor(
    tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather packed tokens into a fixed shape with a graph-connected sentinel."""
    # The empty slice is identically zero even when tensor contains NaN/Inf,
    # while retaining a zero-gradient edge to an empty input tensor.
    dependency = tensor.reshape(-1)[:0].sum()
    sentinel = tensor.new_zeros((1, *tensor.shape[1:])) + dependency
    padded = torch.cat((tensor, sentinel), dim=0)
    return padded.index_select(0, indices.flatten()).view(
        *indices.shape, *tensor.shape[1:]
    )


def varlen_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run packed requests through capture-safe, independently masked SDPA."""
    lengths_q, valid_q, indices_q = _varlen_layout(
        cu_seqlens_q, q.shape[0], spec.seqlen_q
    )
    lengths_k, valid_k, indices_k = _varlen_layout(
        cu_seqlens_k, k.shape[0], spec.seqlen_k
    )
    query = _pad_varlen_tensor(q, indices_q)
    key = _pad_varlen_tensor(k, indices_k)
    value = _pad_varlen_tensor(v, indices_k)

    query_positions = torch.arange(spec.seqlen_q, device=q.device)
    key_positions = torch.arange(spec.seqlen_k, device=q.device)
    mask = valid_q.unsqueeze(2) & valid_k.unsqueeze(1)
    if spec.causal:
        causal_mask = key_positions.view(1, 1, -1) <= (
            query_positions.view(1, -1, 1)
            + lengths_k.view(-1, 1, 1)
            - lengths_q.view(-1, 1, 1)
        )
        mask = mask & causal_mask

    visible_queries = mask.any(dim=-1)
    used_requests = mask.flatten(1).any(dim=-1)

    # Fully masked query rows and requests with no query tokens must not read
    # poisoned inputs. torch.where retains the autograd edge and returns a
    # connected zero gradient without evaluating non-finite values times zero.
    query = torch.where(
        visible_queries[:, :, None, None], query, torch.zeros_like(query)
    )
    key = torch.where(
        used_requests[:, None, None, None], key, torch.zeros_like(key)
    )
    value = torch.where(
        used_requests[:, None, None, None], value, torch.zeros_like(value)
    )

    with torch.autocast(device_type=q.device.type, enabled=False):
        out = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask.unsqueeze(1),
            dropout_p=0.0,
            scale=softmax_scale,
            enable_gqa=spec.nheads_q != spec.nheads_kv,
        )
    out = out.transpose(1, 2)
    out = torch.where(
        visible_queries[:, :, None, None], out, torch.zeros_like(out)
    )
    out = torch.where(valid_q[:, :, None, None], out, torch.zeros_like(out))

    # Valid query indices are unique. Padded rows all accumulate zero into the
    # extra sentinel row, which is dropped to restore packed THD layout.
    packed = q.new_zeros((q.shape[0] + 1, *q.shape[1:]))
    packed = packed.index_add(0, indices_q.flatten(), out.flatten(0, 1))
    return packed[:-1]
