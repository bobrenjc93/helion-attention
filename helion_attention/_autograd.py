"""Autograd bridges for generated attention forwards."""

from __future__ import annotations

from typing import Any

import torch

from ._registry import lookup
from ._registry import lookup_backward
from ._sdpa import dense_attention_cudnn
from ._sdpa import dense_attention_sdpa
from ._shape import AttnShape


class _GeneratedForwardWithSDPAGradients(torch.autograd.Function):
    """Expose generated values while differentiating an SDPA surrogate."""

    @staticmethod
    def forward(
        ctx: Any,
        generated: torch.Tensor,
        sdpa: torch.Tensor,
    ) -> torch.Tensor:
        del ctx, sdpa
        # Returning an independent tensor avoids the custom-Function view
        # restrictions while preserving every generated bf16 value exactly.
        return generated.clone()

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        del ctx
        return None, grad_out


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float,
        spec: AttnShape,
    ) -> torch.Tensor:
        out = lookup(spec)(q, k, v, softmax_scale)
        ctx.save_for_backward(q, k, v)
        ctx.softmax_scale = softmax_scale
        ctx.spec = spec
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        q, k, v = ctx.saved_tensors
        dq, dk, dv = lookup_backward(ctx.spec)(
            q,
            k,
            v,
            grad_out.contiguous(),
            ctx.softmax_scale,
        )
        needs_q, needs_k, needs_v = ctx.needs_input_grad[:3]
        return (
            dq if needs_q else None,
            dk if needs_k else None,
            dv if needs_v else None,
            None,
            None,
        )


def attention_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run a generated forward and attach its matching generated backward."""
    return _Attention.apply(q, k, v, softmax_scale, spec)


def attention_with_sdpa_gradients(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run the generated forward while routing gradients through SDPA."""
    with torch.no_grad():
        generated = lookup(spec)(q, k, v, softmax_scale)
    sdpa = dense_attention_cudnn(
        q, k, v, softmax_scale, is_causal=spec.causal
    )
    if sdpa is None:
        sdpa = dense_attention_sdpa(q, k, v, softmax_scale, spec)
    return _GeneratedForwardWithSDPAGradients.apply(generated, sdpa)
