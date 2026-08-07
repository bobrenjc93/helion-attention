"""Small PyTorch SDPA policy helpers used by benchmark tooling."""

from __future__ import annotations

from typing import TypeVar

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
