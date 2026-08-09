"""PyTorch SDPA helpers shared by runtime fallbacks and benchmarks."""

from __future__ import annotations

from typing import TypeVar

import torch
from torch.nn.attention.bias import causal_lower_right

from ._shape import AttnShape

_Mask = TypeVar("_Mask")


def dense_attention_cudnn_default_scale(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor | None:
    """Run direct cuDNN SDPA, or return ``None`` when it is not usable."""
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)

    # sdpa_kernel mutates process-wide backend flags and races with concurrent
    # callers. Ask PyTorch whether this exact call is supported, then invoke
    # the cuDNN operator directly without changing backend-selection state.
    params_type = getattr(torch.backends.cuda, "SDPAParams", None)
    can_use_cudnn = getattr(
        torch.backends.cuda, "can_use_cudnn_attention", None
    )
    cudnn_sdp_enabled = getattr(
        torch.backends.cuda, "cudnn_sdp_enabled", None
    )
    if (
        params_type is None
        or can_use_cudnn is None
        or cudnn_sdp_enabled is None
        or not cudnn_sdp_enabled()
        or not torch.backends.cudnn.enabled
        or not torch.backends.cudnn.is_available()
        or torch.backends.cudnn.deterministic
        or torch.are_deterministic_algorithms_enabled()
    ):
        return None
    try:
        # PyTorch's eligibility probe consults the current CUDA device rather
        # than deriving all device properties from the tensors. Preserve the
        # caller's device while making both the probe and launch tensor-local.
        with torch.cuda.device(q.device):
            params = params_type(query, key, value, None, 0.0, True, False)
            if not can_use_cudnn(params):
                return None
            cudnn_attention = (
                torch.ops.aten._scaled_dot_product_cudnn_attention.default
            )
            with torch.autocast(device_type=q.device.type, enabled=False):
                out = cudnn_attention(
                    query,
                    key,
                    value,
                    None,
                    False,
                    0.0,
                    True,
                    False,
                )[0]
    except torch.cuda.OutOfMemoryError:
        raise
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        # Capability checks cannot cover missing operators in older builds or
        # every cuDNN graph-construction/runtime rejection. The checked-in
        # generated kernel remains the compatibility path for those cases.
        return None
    return out.transpose(1, 2).contiguous()


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
    dropout_p: float = 0.0,
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
