"""PyTorch SDPA helpers shared by runtime fallbacks and benchmarks."""

from __future__ import annotations

from typing import TypeVar

import torch

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
    causal_mask = None
    if spec.causal and spec.seqlen_q != spec.seqlen_k and not spec.is_decode:
        query_index = torch.arange(spec.seqlen_q, device=q.device)[:, None]
        key_index = torch.arange(spec.seqlen_k, device=q.device)[None, :]
        causal_mask = (
            key_index <= query_index + spec.seqlen_k - spec.seqlen_q
        )
    causal_mask, is_causal = sdpa_causal_options(spec, causal_mask)

    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=softmax_scale,
        enable_gqa=spec.nheads_q != spec.nheads_kv,
    )
    return out.transpose(1, 2).contiguous()
