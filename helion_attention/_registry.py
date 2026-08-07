"""Lookup from a fully specialized shape to a checked-in generated kernel."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Callable
from typing import Protocol

import torch

from ._shape import AttnShape

KERNELS_PACKAGE = "helion_attention.kernels"
_MANIFEST_PATH = Path(__file__).parent / "kernels" / "manifest.json"

AttnKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]
AttnBackwardKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
VarlenAttnKernel = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        float,
        bool,
    ],
    torch.Tensor,
]


class KernelModule(Protocol):
    """Structural type of a generated kernel module."""

    KERNEL_SPEC: dict[str, object]
    attention: AttnKernel


class UnsupportedShapeError(NotImplementedError):
    """Raised when no kernel has been generated for the requested shape."""


@lru_cache(maxsize=1)
def _manifest_payload() -> dict[str, list[dict[str, object]]]:
    if not _MANIFEST_PATH.exists():
        return {"kernels": [], "varlen_kernels": []}
    with _MANIFEST_PATH.open() as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _manifest() -> dict[str, dict[str, object]]:
    entries = _manifest_payload().get("kernels", [])
    return {entry["key"]: entry for entry in entries}


@lru_cache(maxsize=1)
def _varlen_manifest() -> dict[str, dict[str, object]]:
    entries = _manifest_payload().get("varlen_kernels", [])
    return {entry["key"]: entry for entry in entries}


def available_shapes() -> list[dict[str, object]]:
    """Every shape this build ships a kernel for, in manifest order."""
    return list(_manifest().values())


def available_varlen_shapes() -> list[dict[str, object]]:
    """Every packed-sequence specialization shipped by this build."""
    return list(_varlen_manifest().values())


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def spec_from_manifest_entry(entry: dict[str, object]) -> AttnShape:
    """Rebuild an :class:`AttnShape` from a manifest (or ``KERNEL_SPEC``) entry."""
    return AttnShape(
        batch=int(entry["batch"]),
        seqlen_q=int(entry["seqlen_q"]),
        seqlen_k=int(entry["seqlen_k"]),
        nheads_q=int(entry["nheads_q"]),
        nheads_kv=int(entry["nheads_kv"]),
        head_dim=int(entry["head_dim"]),
        dtype=_DTYPES[str(entry["dtype"])],
        causal=bool(entry["causal"]),
    )


@lru_cache(maxsize=None)
def _load(key: str) -> AttnKernel:
    module = importlib.import_module(f"{KERNELS_PACKAGE}.{key}")
    return module.attention


@lru_cache(maxsize=None)
def _load_backward(key: str) -> AttnBackwardKernel:
    module = importlib.import_module(f"{KERNELS_PACKAGE}.{key}_backward")
    return module.attention_backward


@lru_cache(maxsize=None)
def _load_varlen(key: str) -> VarlenAttnKernel:
    module = importlib.import_module(f"{KERNELS_PACKAGE}.{key}")
    return module.attention_varlen


def _entry_for_spec(spec: AttnShape) -> dict[str, object] | None:
    entry = _manifest().get(spec.key)
    if entry is None and spec.is_decode:
        # With a single bottom-right-aligned query, causal and non-causal both
        # expose the entire cache. Reuse the one generated specialization so
        # the FlashAttention-compatible default (causal=False) also works.
        equivalent = replace(spec, causal=not spec.causal)
        entry = _manifest().get(equivalent.key)
    return entry


def _nearest(spec: AttnShape, limit: int = 8) -> list[str]:
    """Shapes that differ from ``spec`` in the fewest fields, for the error text."""
    fields = ("batch", "seqlen_q", "seqlen_k", "nheads_q", "nheads_kv", "head_dim")
    scored = []
    for entry in _manifest().values():
        distance = sum(entry[field] != getattr(spec, field) for field in fields)
        distance += entry["dtype"] != spec.dtype_name
        distance += entry["causal"] != spec.causal
        scored.append((distance, entry["description"]))
    scored.sort(key=lambda item: item[0])
    return [description for _, description in scored[:limit]]


def lookup(spec: AttnShape) -> AttnKernel:
    """Return the generated kernel for ``spec`` or explain how to get one."""
    entry = _entry_for_spec(spec)
    if entry is None:
        nearest = _nearest(spec)
        listing = "\n".join(f"    {item}" for item in nearest)
        raise UnsupportedShapeError(
            f"no helion-attention kernel is checked in for:\n    {spec.describe()}\n"
            f"closest available shapes:\n{listing}\n"
            "Kernels are generated and autotuned per shape. Please file an issue at "
            "https://github.com/bobrenjc93/helion-attention/issues with the shape "
            "above and it can be generated and added."
        )
    return _load(str(entry["key"]))


def lookup_varlen(spec: AttnShape) -> VarlenAttnKernel:
    """Return the generated packed-sequence kernel for ``spec``."""
    key = f"varlen_{spec.key}"
    entry = _varlen_manifest().get(key)
    if entry is None:
        fields = (
            "batch",
            "seqlen_q",
            "seqlen_k",
            "nheads_q",
            "nheads_kv",
            "head_dim",
        )
        scored = []
        for candidate in _varlen_manifest().values():
            distance = sum(
                candidate[field] != getattr(spec, field) for field in fields
            )
            distance += candidate["dtype"] != spec.dtype_name
            distance += candidate["causal"] != spec.causal
            scored.append((distance, candidate["description"]))
        scored.sort(key=lambda item: item[0])
        listing = "\n".join(f"    {description}" for _, description in scored[:8])
        raise UnsupportedShapeError(
            "no helion-attention varlen kernel is checked in for:\n"
            f"    {spec.describe()}\n"
            f"closest available varlen shapes:\n{listing or '    (none)'}\n"
            "Packed kernels are generated and autotuned per maximum shape. "
            "Please file an issue at "
            "https://github.com/bobrenjc93/helion-attention/issues with the "
            "shape above and it can be generated and added."
        )
    return _load_varlen(str(entry["key"]))


def has_backward(spec: AttnShape) -> bool:
    """Whether this forward specialization also ships generated gradients."""
    entry = _entry_for_spec(spec)
    return entry is not None and bool(entry.get("backward", False))


def lookup_backward(spec: AttnShape) -> AttnBackwardKernel:
    """Return the generated backward kernel for ``spec``."""
    entry = _entry_for_spec(spec)
    if entry is None or not entry.get("backward", False):
        supported = [
            str(item["description"])
            for item in _manifest().values()
            if item.get("backward", False)
        ]
        listing = "\n".join(f"    {item}" for item in supported)
        raise NotImplementedError(
            "no helion-attention backward kernel is checked in for:\n"
            f"    {spec.describe()}\n"
            f"training-enabled shapes:\n{listing or '    (none)'}"
        )
    return _load_backward(str(entry["key"]))
