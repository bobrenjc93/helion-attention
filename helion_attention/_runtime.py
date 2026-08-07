"""Runtime helpers the generated kernels call into.

These are the handful of host-side utilities Helion's generated code normally
imports from ``helion.runtime``. Vendoring them keeps the generated kernels
byte-for-byte close to what Helion emitted while removing the import.
"""

from __future__ import annotations

import contextvars

import torch
import triton


def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    """Scratch allocator for Triton's ``make_tensor_descriptor`` (TMA) path."""
    current_target = triton.runtime.driver.active.get_current_target()
    if current_target is None:
        raise RuntimeError("no active Triton target; is a CUDA device selected?")
    return torch.empty(size, device=current_target.backend, dtype=torch.int8)


def set_triton_allocator() -> None:
    """Install ``_alloc_fn`` in this context unless one is already configured."""
    try:
        from triton import set_allocator
        from triton.runtime._allocation import NullAllocator
        from triton.runtime._allocation import _allocator
    except ImportError:
        return
    if isinstance(_allocator, contextvars.ContextVar):
        existing = _allocator.get()
    else:  # older Triton
        existing = _allocator
    if isinstance(existing, NullAllocator):
        set_allocator(_alloc_fn)


def get_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    """Grid size for the persistent kernels: one program per SM."""
    if device.type != "cuda":
        raise RuntimeError(f"helion-attention kernels require CUDA, got {device.type}")
    available = torch.cuda.get_device_properties(device.index).multi_processor_count
    if reserved_sms <= 0:
        return available
    return max(available - reserved_sms, 1)
