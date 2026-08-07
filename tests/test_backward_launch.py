"""Portable launch constraints for the generated cooperative backward kernel."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

MODULE = (
    "helion_attention.kernels."
    "b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal_backward"
)
REGISTERS_PER_SM = {"sm80": 65_536, "sm86": 65_536, "sm89": 65_536}
MAX_REGISTERS_PER_THREAD = 255


@pytest.mark.parametrize(
    ("architecture", "num_sms"),
    [("sm80", 108), ("sm86", 84), ("sm89", 128)],
)
def test_cooperative_grid_uses_one_cta_per_sm(
    architecture: str,
    num_sms: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One resident CTA per SM is safe across Ampere and Ada resource limits."""
    module = importlib.import_module(MODULE)
    launches: list[tuple[tuple[int, ...], dict[str, Any]]] = []

    def capture_launch(
        _kernel: object, grid: tuple[int, ...], *_args: object, **kwargs: Any
    ) -> None:
        launches.append((grid, kwargs))

    monkeypatch.setattr(module, "get_num_sm", lambda _device: num_sms)
    shape = (8, 512, 16, 64)
    q = torch.empty(shape, device="meta", dtype=torch.bfloat16)

    module.attention_backward(q, q, q, q, 1.0, _launcher=capture_launch)

    assert len(launches) == 1, architecture
    grid, options = launches[0]
    assert grid == (num_sms,), architecture
    assert options["launch_cooperative_grid"] is True, architecture
    threads_per_cta = int(options["num_warps"]) * 32
    assert (
        MAX_REGISTERS_PER_THREAD * threads_per_cta
        <= REGISTERS_PER_SM[architecture]
    ), architecture
