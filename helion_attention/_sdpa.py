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


def _varlen_lengths(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_seqlen: int,
    name: str,
) -> list[int]:
    """Copy backward-only offsets to the host and validate their lengths."""
    offsets = cu_seqlens.tolist()
    if offsets[0] != 0:
        raise ValueError(f"{name} must start at 0, got {offsets[0]}")
    if offsets[-1] != total_tokens:
        raise ValueError(
            f"{name} must end at the packed token count {total_tokens}, "
            f"got {offsets[-1]}"
        )

    lengths = []
    for request, (start, end) in enumerate(zip(offsets, offsets[1:])):
        length = end - start
        if length < 0:
            raise ValueError(
                f"{name} must be nondecreasing; request {request} has offsets "
                f"({start}, {end})"
            )
        if length > max_seqlen:
            raise ValueError(
                f"{name} request {request} has length {length}, greater than "
                f"the declared maximum {max_seqlen}"
            )
        lengths.append(length)
    return lengths


def varlen_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run packed requests independently through native SDPA autograd."""
    lengths_q = _varlen_lengths(
        cu_seqlens_q, q.shape[0], spec.seqlen_q, "cu_seqlens_q"
    )
    lengths_k = _varlen_lengths(
        cu_seqlens_k, k.shape[0], spec.seqlen_k, "cu_seqlens_k"
    )

    outputs = []
    for query, key, value in zip(
        torch.split(q, lengths_q),
        torch.split(k, lengths_k),
        torch.split(v, lengths_k),
    ):
        seqlen_q = query.shape[0]
        seqlen_k = key.shape[0]
        if seqlen_q == 0 or seqlen_k == 0:
            # SDPA defines empty-key attention as zero. Keep all three inputs
            # in the graph so requested gradients are empty/zero, not unused.
            dependency = (key.sum() + value.sum()).to(query.dtype) * 0
            outputs.append(query * 0 + dependency)
            continue

        request_spec = AttnShape(
            batch=1,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            nheads_q=spec.nheads_q,
            nheads_kv=spec.nheads_kv,
            head_dim=spec.head_dim,
            dtype=spec.dtype,
            causal=spec.causal,
        )
        outputs.append(
            dense_attention_sdpa(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                softmax_scale,
                request_spec,
            ).squeeze(0)
        )
    return torch.cat(outputs)
