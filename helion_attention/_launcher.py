"""Minimal Triton kernel launcher used by the checked-in generated kernels.

Helion emits code that calls a ``_launcher(kernel, grid, *args, ...)`` callable.
Vendoring the launcher here is what lets the generated kernels run with only
``torch`` and ``triton`` installed, with no import of Helion at runtime.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol

import torch

from ._runtime import set_triton_allocator


class TritonKernel(Protocol):
    """Structural type for the ``triton.jit`` wrapper object we launch."""

    def run(self, *args: object, **kwargs: object) -> object: ...


def default_launcher(
    triton_kernel: TritonKernel,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: Any,
) -> object:
    """Launch ``triton_kernel`` on the device of its CUDA tensor arguments."""
    run_kwargs: dict[str, object] = {
        "grid": grid,
        "warmup": False,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        **kwargs,
    }
    if ptx_options is not None:
        run_kwargs["ptx_options"] = ptx_options
    try:
        device = next(
            arg.device
            for arg in args
            if isinstance(arg, torch.Tensor) and arg.is_cuda
        )
    except StopIteration as exc:
        raise ValueError("Triton kernel launch requires a CUDA tensor argument") from exc

    # Triton's allocator is context-local, so imports performed by a setup
    # thread do not configure worker threads. Install it in the invocation
    # context, while the input device is current for allocation and launch.
    with torch.cuda.device(device):
        set_triton_allocator()
        return triton_kernel.run(*args, **run_kwargs)
