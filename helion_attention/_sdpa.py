"""PyTorch SDPA helpers shared by runtime fallbacks and benchmarks."""

from __future__ import annotations

from typing import TypeVar

import torch
from torch.nn.attention.bias import causal_lower_right

from ._shape import AttnShape

_Mask = TypeVar("_Mask")

DENSE_ALIBI_BACKWARD_KEY = "b1_sq64_sk64_hq32_hkv8_d128_bf16_causal"
_DENSE_ALIBI_BIAS_ELEMENTS = 1 * 32 * 64 * 64

# Unlike the public SDPA dispatcher, this composite autograd operator always
# runs the math implementation and never consults or mutates process-wide CUDA
# backend flags. PyTorch 2.5+, the package minimum, exposes it on every backend.
_MATH_ATTENTION_FORWARD = (
    torch.ops.aten._scaled_dot_product_attention_math.default
)

try:
    # This is the BSHD Flash operator underlying PyTorch's fused SDPA path. It
    # avoids four Python-level transpose/view calls around the BHSD SDPA wrapper.
    _FLASH_ATTENTION_FORWARD = (
        torch.ops.aten._flash_attention_forward.default
    )
except AttributeError:  # pragma: no cover - depends on the PyTorch build
    _FLASH_ATTENTION_FORWARD = None

# Eligibility is invariant for each validated head dimension on a given device
# and PyTorch build. Cache successful probes only; backend enablement and global
# determinism remain live per-call checks below.
_FLASH_CAPABLE_PROFILES: set[tuple[int | None, int, object, object]] = set()


def dense_attention_flash_default_scale(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor | None:
    """Run direct Flash SDPA, or return ``None`` when it is not usable."""
    # Selecting a backend through sdpa_kernel mutates process-wide flags and
    # adds dispatcher overhead to this latency-sensitive shape. Probe the exact
    # validated layout once, then invoke PyTorch's BSHD Flash operator directly.
    params_type = getattr(torch.backends.cuda, "SDPAParams", None)
    can_use_flash = getattr(
        torch.backends.cuda, "can_use_flash_attention", None
    )
    flash_sdp_enabled = getattr(
        torch.backends.cuda, "flash_sdp_enabled", None
    )
    if (
        params_type is None
        or can_use_flash is None
        or flash_sdp_enabled is None
        or _FLASH_ATTENTION_FORWARD is None
        or not flash_sdp_enabled()
        or torch.are_deterministic_algorithms_enabled()
    ):
        return None
    try:
        capability_key = (
            q.device.index,
            q.shape[-1],
            params_type,
            can_use_flash,
        )
        if capability_key not in _FLASH_CAPABLE_PROFILES:
            query = q.transpose(1, 2)
            key = k.transpose(1, 2)
            value = v.transpose(1, 2)
            # The probe consults the current CUDA device. Temporarily select
            # the tensor device only on this cold capability-check path; the
            # operator itself has its own tensor-derived device guard.
            with torch.cuda.device(q.device):
                params = params_type(
                    query, key, value, None, 0.0, False, False
                )
                if not can_use_flash(params):
                    return None
            _FLASH_CAPABLE_PROFILES.add(capability_key)

        flash_attention = _FLASH_ATTENTION_FORWARD
        if torch.is_autocast_enabled(q.device.type):
            # The raw operator participates in autocast. FlashAttention's API
            # instead preserves the validated input dtype.
            with torch.autocast(device_type=q.device.type, enabled=False):
                out = flash_attention(
                    q,
                    k,
                    v,
                    None,
                    None,
                    q.shape[1],
                    k.shape[1],
                    0.0,
                    False,
                    False,
                )[0]
        else:
            out = flash_attention(
                q,
                k,
                v,
                None,
                None,
                q.shape[1],
                k.shape[1],
                0.0,
                False,
                False,
            )[0]
    except torch.cuda.OutOfMemoryError:
        raise
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        # Older PyTorch builds may lack the raw operator, and a successful
        # capability probe cannot rule out every launch-time rejection. The
        # generated specialization remains the compatibility fallback.
        return None
    return out


def dense_attention_cudnn_default_scale(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor | None:
    """Run direct cuDNN SDPA, or return ``None`` when it is not usable."""
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    enable_gqa = query.shape[1] != key.shape[1]

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
            params = params_type(
                query, key, value, None, 0.0, True, enable_gqa
            )
            if not can_use_cudnn(params):
                return None
            cudnn_attention = (
                torch.ops.aten._scaled_dot_product_cudnn_attention.default
            )
            if torch.is_autocast_enabled(q.device.type):
                # The raw operator is autocast-eligible. Preserve the input
                # dtype under an active cross-dtype autocast context, without
                # paying for a redundant context in the latency-critical
                # default case.
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
            else:
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

    # PyTorch requires a floating SDPA bias to match the query dtype. Form the
    # values from fp32 slopes first, then cast only this exact bounded tensor.
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
    symmetric_window_radius: int | None = None,
    causal_window_left: int | None = None,
) -> torch.Tensor:
    """Run a validated dense call through PyTorch's native SDPA autograd.

    ``symmetric_window_radius`` is reserved for the bounded equal-length,
    noncausal varlen bridge validated by the public dispatcher.
    ``causal_window_left`` is reserved for the equal-length causal Llama
    bridge with a zero-token right window.
    ``alibi_slopes`` is reserved for the exact bounded dense ALiBi profile.
    """
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    enable_gqa = spec.nheads_q != spec.nheads_kv

    if (
        symmetric_window_radius is not None
        and causal_window_left is not None
    ):
        raise ValueError("only one SDPA window mode may be selected")
    if alibi_slopes is not None and (
        symmetric_window_radius is not None or causal_window_left is not None
    ):
        raise ValueError("ALiBi SDPA cannot be combined with a window mode")

    window_mask = None
    if symmetric_window_radius is not None:
        if spec.causal or spec.seqlen_q != spec.seqlen_k:
            raise ValueError(
                "symmetric-window SDPA requires equal-length noncausal "
                "self-attention"
            )
        if symmetric_window_radius < 0:
            raise ValueError("symmetric_window_radius must be non-negative")
        positions = torch.arange(spec.seqlen_q, device=q.device)
        window_mask = (
            positions[:, None] - positions[None, :]
        ).abs() <= symmetric_window_radius
    elif causal_window_left is not None:
        if not spec.causal or spec.seqlen_q != spec.seqlen_k:
            raise ValueError(
                "causal left-window SDPA requires equal-length causal "
                "self-attention"
            )
        if causal_window_left < 0:
            raise ValueError("causal_window_left must be non-negative")
        positions = torch.arange(spec.seqlen_q, device=q.device)
        relative_position = positions[:, None] - positions[None, :]
        window_mask = (relative_position >= 0) & (
            relative_position <= causal_window_left
        )

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
        elif window_mask is not None:
            if dropout_p != 0.0:
                raise ValueError("windowed SDPA requires dropout_p=0.0")
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=window_mask,
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


def dense_attention_math_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run deterministic math SDPA without changing global backend flags."""
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    enable_gqa = spec.nheads_q != spec.nheads_kv

    # The validated deterministic profiles are equal-length MHA. Pass their
    # causal semantics explicitly while preserving FlashAttention's input dtype
    # under any active cross-dtype autocast context.
    with torch.autocast(device_type=q.device.type, enabled=False):
        out = _MATH_ATTENTION_FORWARD(
            query,
            key,
            value,
            None,
            0.0,
            spec.causal,
            None,
            scale=softmax_scale,
            enable_gqa=enable_gqa,
        )[0]
    return out.transpose(1, 2).contiguous()
