"""Minimal Triton kernel launcher used by the checked-in generated kernels.

Helion emits code that calls a ``_launcher(kernel, grid, *args, ...)`` callable.
Vendoring the launcher here is what lets the generated kernels run with only
``torch`` and ``triton`` installed, with no import of Helion at runtime.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol


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
    """Launch ``triton_kernel`` immediately on the current CUDA stream."""
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
    return triton_kernel.run(*args, **run_kwargs)
