"""Portable launch constraints for the generated cooperative backward kernel."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "helion_attention"
    / "kernels"
    / "b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal_backward.py"
)
REGISTERS_PER_SM = {"sm80": 65_536, "sm86": 65_536, "sm89": 65_536}
MAX_REGISTERS_PER_THREAD = 255


def _backward_launch() -> ast.Call:
    tree = ast.parse(MODULE_PATH.read_text(), filename=str(MODULE_PATH))
    wrappers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "attention_backward"
    ]
    assert len(wrappers) == 1
    launches = [
        node
        for node in ast.walk(wrappers[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(launches) == 1
    return launches[0]


def test_backward_module_certifies_deterministic_execution() -> None:
    tree = ast.parse(MODULE_PATH.read_text(), filename=str(MODULE_PATH))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "KERNEL_SPEC"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    spec = ast.literal_eval(assignments[0].value)
    assert spec["backward_deterministic"] is True

    atomic_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("atomic_")
    ]
    assert not atomic_calls


@pytest.mark.parametrize("architecture", ["sm80", "sm86", "sm89"])
def test_cooperative_grid_uses_one_cta_per_sm(architecture: str) -> None:
    """One resident CTA per SM is safe across Ampere and Ada resource limits."""
    launch = _backward_launch()
    grid = launch.args[1]
    assert isinstance(grid, ast.Tuple), architecture
    assert len(grid.elts) == 1, architecture
    assert isinstance(grid.elts[0], ast.Name), architecture
    assert grid.elts[0].id == "_NUM_SM", architecture

    options = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in launch.keywords
        if keyword.arg is not None
    }
    assert options["launch_cooperative_grid"] is True, architecture
    threads_per_cta = int(options["num_warps"]) * 32
    assert (
        MAX_REGISTERS_PER_THREAD * threads_per_cta
        <= REGISTERS_PER_SM[architecture]
    ), architecture
