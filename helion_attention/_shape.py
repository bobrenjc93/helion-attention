"""Shape specification for shape-specialized attention kernels.

Helion only beats FlashAttention when the kernel is specialized to one exact
problem size, so ``shape`` is a required argument of the public API instead of
something inferred from the tensors. Passing it explicitly also means a missing
kernel is reported before any tensor is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


@dataclass(frozen=True)
class AttnShape:
    """One fully specialized attention problem.

    Layout follows FlashAttention: ``q`` is ``[batch, seqlen_q, nheads_q, head_dim]``
    and ``k``/``v`` are ``[batch, seqlen_k, nheads_kv, head_dim]``.
    """

    batch: int
    seqlen_q: int
    seqlen_k: int
    nheads_q: int
    nheads_kv: int
    head_dim: int
    dtype: torch.dtype
    causal: bool

    def __post_init__(self) -> None:
        for field in ("batch", "seqlen_q", "seqlen_k", "nheads_q", "nheads_kv", "head_dim"):
            value = getattr(self, field)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"AttnShape.{field} must be a positive int, got {value!r}")
        if self.nheads_q % self.nheads_kv != 0:
            raise ValueError(
                f"nheads_q ({self.nheads_q}) must be divisible by nheads_kv ({self.nheads_kv})"
            )
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(
                f"unsupported dtype {self.dtype}; supported: {sorted(_DTYPE_NAMES.values())}"
            )

    @property
    def dtype_name(self) -> str:
        return _DTYPE_NAMES[self.dtype]

    @property
    def key(self) -> str:
        """Stable identifier used as the generated kernel module name."""
        return (
            f"b{self.batch}"
            f"_sq{self.seqlen_q}_sk{self.seqlen_k}"
            f"_hq{self.nheads_q}_hkv{self.nheads_kv}"
            f"_d{self.head_dim}"
            f"_{self.dtype_name}"
            f"_{'causal' if self.causal else 'noncausal'}"
        )

    def describe(self) -> str:
        gqa = "" if self.nheads_q == self.nheads_kv else f" (GQA {self.nheads_q}:{self.nheads_kv})"
        return (
            f"batch={self.batch} seqlen_q={self.seqlen_q} seqlen_k={self.seqlen_k} "
            f"nheads={self.nheads_q}{gqa} head_dim={self.head_dim} "
            f"dtype={self.dtype_name} causal={self.causal}"
        )


ShapeLike = Union[AttnShape, tuple[int, ...], torch.Size]


def normalize_shape(shape: ShapeLike, dtype: torch.dtype, causal: bool) -> AttnShape:
    """Turn the user-supplied ``shape`` argument into an :class:`AttnShape`.

    A 4-tuple ``(batch, seqlen, nheads, head_dim)`` is the common self-attention
    case. A 6-tuple additionally spells out ``seqlen_k`` and ``nheads_kv`` for
    cross-attention and grouped-query attention.
    """
    if isinstance(shape, AttnShape):
        if shape.dtype != dtype or shape.causal != causal:
            raise ValueError(
                f"shape says dtype={shape.dtype}/causal={shape.causal} but the call "
                f"passed dtype={dtype}/causal={causal}"
            )
        return shape
    values = tuple(int(v) for v in shape)
    if len(values) == 4:
        batch, seqlen, nheads, head_dim = values
        return AttnShape(batch, seqlen, seqlen, nheads, nheads, head_dim, dtype, causal)
    if len(values) == 6:
        batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim = values
        return AttnShape(
            batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, causal
        )
    raise ValueError(
        "shape must be an AttnShape, a 4-tuple (batch, seqlen, nheads, head_dim), "
        f"or a 6-tuple (batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim); got {shape!r}"
    )


def check_tensors(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: AttnShape
) -> None:
    """Fail loudly when the tensors disagree with the declared shape."""
    expected_q = (spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim)
    expected_kv = (spec.batch, spec.seqlen_k, spec.nheads_kv, spec.head_dim)
    for name, tensor, expected in (("q", q, expected_q), ("k", k, expected_kv), ("v", v, expected_kv)):
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)} but the declared shape implies {expected}"
            )
        if tensor.dtype != spec.dtype:
            raise ValueError(f"{name} has dtype {tensor.dtype} but shape declares {spec.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous in [batch, seqlen, nheads, head_dim] layout; "
                "the generated kernels bake in the strides of that layout"
            )
