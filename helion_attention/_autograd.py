"""Autograd bridges for generated attention forwards."""

from __future__ import annotations

from typing import Any

import torch

from ._registry import lookup
from ._registry import lookup_backward
from ._sdpa import dense_attention_flash_default_scale
from ._shape import AttnShape


class _GeneratedForwardWithFlashGradients(torch.autograd.Function):
    """Expose generated values while differentiating a Flash surrogate."""

    @staticmethod
    def forward(
        ctx: Any,
        generated: torch.Tensor,
        flash: torch.Tensor,
    ) -> torch.Tensor:
        del ctx, flash
        # Returning an independent tensor avoids the custom-Function view
        # restrictions while preserving every generated bf16 value exactly.
        return generated.clone()

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        del ctx
        return None, grad_out


class _AttentionDiagnostics(torch.autograd.Function):
    """Expose diagnostic tensors while differentiating only the output."""

    @staticmethod
    def forward(
        ctx: Any,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        s_dmask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del ctx
        # FlashAttention exposes all three tensors as differentiable outputs,
        # but its backward ignores grad_softmax_lse and grad_S_dmask. Cloning
        # avoids custom-Function view restrictions while retaining that exact
        # contract for diagnostics produced under no_grad.
        return out.clone(), softmax_lse.clone(), s_dmask.clone()

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
        grad_softmax_lse: torch.Tensor,
        grad_s_dmask: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        del ctx, grad_softmax_lse, grad_s_dmask
        return grad_out, None, None


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


def attention_with_flash_gradients(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Use raw BSHD Flash gradients, falling back to generated backward."""
    flash = dense_attention_flash_default_scale(q, k, v)
    if flash is None or not flash.requires_grad:
        return attention_autograd(q, k, v, softmax_scale, spec)
    with torch.no_grad():
        generated = lookup(spec)(q, k, v, softmax_scale)
    return _GeneratedForwardWithFlashGradients.apply(generated, flash)


def attention_diagnostics(
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    s_dmask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach FA2-compatible zero-gradient diagnostic outputs to ``out``."""
    return _AttentionDiagnostics.apply(out, softmax_lse, s_dmask)
