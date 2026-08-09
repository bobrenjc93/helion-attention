"""PyTorch SDPA helpers shared by runtime fallbacks and benchmarks."""

from __future__ import annotations

from typing import TypeVar

import torch
from torch.nn.attention.bias import causal_lower_right

from ._shape import AttnShape

_Mask = TypeVar("_Mask")

DENSE_ALIBI_BACKWARD_KEY = "b1_sq64_sk320_hq8_hkv2_d128_bf16_causal"
_DENSE_ALIBI_BIAS_ELEMENTS = 1 * 8 * 64 * 320


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


def _dense_alibi_bias(
    alibi_slopes: torch.Tensor,
    q: torch.Tensor,
    spec: AttnShape,
) -> torch.Tensor:
    """Materialize the one bounded additive bias supported for autograd."""
    if spec.key != DENSE_ALIBI_BACKWARD_KEY:
        raise NotImplementedError(
            "the additive-bias ALiBi SDPA fallback is implemented only for "
            f"{DENSE_ALIBI_BACKWARD_KEY}; got {spec.key}"
        )
    if alibi_slopes.requires_grad:
        raise NotImplementedError(
            "ALiBi backward does not implement slope gradients"
        )

    slopes = (
        alibi_slopes.unsqueeze(0)
        if alibi_slopes.ndim == 1
        else alibi_slopes
    )
    row = torch.arange(spec.seqlen_q, device=q.device)[:, None]
    col = torch.arange(spec.seqlen_k, device=q.device)[None, :]
    aligned_row = row + spec.seqlen_k - spec.seqlen_q
    distance = torch.abs(aligned_row - col)
    causal_mask = col <= aligned_row

    # SDPA requires a floating bias to have the query dtype. Compute the ALiBi
    # values from the fp32 slopes first, then cast the exact bounded tensor.
    bias = (
        -slopes[:, :, None, None] * distance[None, None]
    ).to(dtype=q.dtype)
    if bias.numel() != _DENSE_ALIBI_BIAS_ELEMENTS:  # pragma: no cover - guard
        raise RuntimeError(
            "the bounded ALiBi bias unexpectedly contained "
            f"{bias.numel()} elements"
        )
    return bias.masked_fill_(~causal_mask[None, None], float("-inf"))


def dense_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
    dropout_p: float = 0.0,
    *,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a validated dense call through PyTorch's native SDPA autograd."""
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    enable_gqa = spec.nheads_q != spec.nheads_kv

    # FlashAttention returns the input dtype even under cross-dtype autocast.
    # SDPA is autocast-eligible, so keep its native fp16/bf16 contract explicit.
    with torch.autocast(device_type=q.device.type, enabled=False):
        if alibi_slopes is not None:
            if dropout_p != 0.0:
                raise NotImplementedError(
                    "dropout combined with ALiBi slopes is not implemented"
                )
            additive_bias = _dense_alibi_bias(alibi_slopes, q, spec)
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=additive_bias,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
                enable_gqa=enable_gqa,
            )
        elif spec.causal and spec.seqlen_q > spec.seqlen_k:
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
                dropout_p=dropout_p,
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
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=softmax_scale,
                enable_gqa=enable_gqa,
            )
    return out.transpose(1, 2).contiguous()
