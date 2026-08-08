"""Shape specification for shape-specialized attention kernels.

Helion only beats FlashAttention when the kernel is specialized to one exact
problem size, so ``shape`` is a required argument of the public API instead of
something inferred from the tensors. The same declaration validates calls that
use the generic dense fallback when no specialization is checked in.
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
    def is_decode(self) -> bool:
        """Whether this is single-token attention over an existing KV cache."""
        return self.seqlen_q == 1 and self.seqlen_k > 1

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
    device = q.device
    for name, tensor, expected in (("q", q, expected_q), ("k", k, expected_kv), ("v", v, expected_kv)):
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)} but the declared shape implies {expected}"
            )
        if tensor.dtype != spec.dtype:
            raise ValueError(f"{name} has dtype {tensor.dtype} but shape declares {spec.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {tensor.device}")
        if tensor.device != device:
            raise ValueError("q, k, and v must be on the same CUDA device")
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous in [batch, seqlen, nheads, head_dim] layout; "
                "the generated kernels bake in the strides of that layout"
            )


def check_varlen_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    spec: AttnShape,
) -> None:
    """Validate FlashAttention's packed ``[total, heads, dim]`` inputs.

    The cumulative values stay device-resident: checking them on the host would
    synchronize every invocation and make CUDA graph capture impossible.  As in
    FlashAttention, callers are responsible for supplying monotonic offsets
    whose final values equal the corresponding packed tensor lengths.
    """
    expected_trailing = {
        "q": (spec.nheads_q, spec.head_dim),
        "k": (spec.nheads_kv, spec.head_dim),
        "v": (spec.nheads_kv, spec.head_dim),
    }
    total_names = {"q": "total_q", "k": "total_k", "v": "total_k"}
    tensors = {"q": q, "k": k, "v": v}
    device = q.device
    for name, tensor in tensors.items():
        trailing = expected_trailing[name]
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != trailing:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)} but the declared varlen "
                f"shape requires [{total_names[name]}, {trailing[0]}, {trailing[1]}]"
            )
        if tensor.dtype != spec.dtype:
            raise ValueError(f"{name} has dtype {tensor.dtype} but shape declares {spec.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {tensor.device}")
        if tensor.device != device:
            raise ValueError("q, k, and v must be on the same CUDA device")
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous in [total, heads, head_dim] layout"
            )
    if k.shape[0] != v.shape[0]:
        raise ValueError(
            f"k and v must contain the same number of packed tokens, got "
            f"{k.shape[0]} and {v.shape[0]}"
        )

    expected_cu_shape = (spec.batch + 1,)
    for name, cu_seqlens in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tuple(cu_seqlens.shape) != expected_cu_shape:
            raise ValueError(
                f"{name} has shape {tuple(cu_seqlens.shape)} but batch={spec.batch} "
                f"requires {expected_cu_shape}"
            )
        if cu_seqlens.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32")
        if not cu_seqlens.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {cu_seqlens.device}")
        if cu_seqlens.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as q, k, and v")
        if not cu_seqlens.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def check_paged_varlen_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    spec: AttnShape,
) -> int:
    """Validate packed queries and block-table-addressed paged K/V caches.

    The generated paged kernels take explicit strides for queries, caches, and
    block tables, so those tensors need not be contiguous.  Cumulative lengths
    remain device-resident, just as they do on the contiguous varlen path.

    Returns:
        The cache page size, used to select the exact generated specialization.
    """
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "block_table": block_table,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use torch.strided layout")

    expected_q_trailing = (spec.nheads_q, spec.head_dim)
    if q.ndim != 3 or tuple(q.shape[1:]) != expected_q_trailing:
        raise ValueError(
            f"q has shape {tuple(q.shape)} but the declared paged varlen shape "
            f"requires [total_q, {spec.nheads_q}, {spec.head_dim}]"
        )
    if q.shape[0] > spec.batch * spec.seqlen_q:
        raise ValueError(
            f"q contains {q.shape[0]} packed tokens, more than batch={spec.batch} "
            f"and max_seqlen_q={spec.seqlen_q} allow"
        )
    if q.dtype != spec.dtype:
        raise ValueError(f"q has dtype {q.dtype} but shape declares {spec.dtype}")
    if not q.is_cuda:
        raise ValueError(f"q must be a CUDA tensor, got device {q.device}")

    expected_cache_tail = (spec.nheads_kv, spec.head_dim)
    for name, cache in (("k", k), ("v", v)):
        if cache.ndim != 4 or tuple(cache.shape[2:]) != expected_cache_tail:
            raise ValueError(
                f"{name} must be a paged KV cache with shape "
                f"[num_blocks, page_size, {spec.nheads_kv}, {spec.head_dim}], "
                f"got {tuple(cache.shape)}"
            )
        if cache.dtype != spec.dtype:
            raise ValueError(
                f"{name} has dtype {cache.dtype} but shape declares {spec.dtype}"
            )
        if not cache.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {cache.device}")
        if cache.device != q.device:
            raise ValueError("q, k, and v must be on the same CUDA device")
    if k.shape != v.shape:
        raise ValueError(
            f"k and v paged caches must have the same shape, got "
            f"{tuple(k.shape)} and {tuple(v.shape)}"
        )
    if k.shape[0] == 0 or k.shape[1] == 0:
        raise ValueError("paged K/V caches must contain at least one non-empty page")

    expected_cu_shape = (spec.batch + 1,)
    for name, cu_seqlens in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tuple(cu_seqlens.shape) != expected_cu_shape:
            raise ValueError(
                f"{name} has shape {tuple(cu_seqlens.shape)} but batch={spec.batch} "
                f"requires {expected_cu_shape}"
            )
        if cu_seqlens.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32")
        if not cu_seqlens.is_cuda:
            raise ValueError(
                f"{name} must be a CUDA tensor, got device {cu_seqlens.device}"
            )
        if cu_seqlens.device != q.device:
            raise ValueError(
                f"{name} must be on the same CUDA device as q, k, and v"
            )
        if not cu_seqlens.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    page_size = k.shape[1]
    if block_table.ndim != 2 or block_table.shape[0] != spec.batch:
        raise ValueError(
            "block_table must have shape [batch, max_num_blocks_per_seq] "
            f"with batch={spec.batch}, got {tuple(block_table.shape)}"
        )
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must have dtype torch.int32")
    if not block_table.is_cuda:
        raise ValueError(
            f"block_table must be a CUDA tensor, got device {block_table.device}"
        )
    if block_table.device != q.device:
        raise ValueError("block_table must be on the same CUDA device as q, k, and v")
    required_blocks = (spec.seqlen_k + page_size - 1) // page_size
    if block_table.shape[1] < required_blocks:
        raise ValueError(
            f"block_table has capacity for {block_table.shape[1] * page_size} "
            f"tokens but max_seqlen_k={spec.seqlen_k} requires {required_blocks} "
            "logical pages"
        )
    return page_size
