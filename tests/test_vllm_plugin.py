"""Integration coverage for the vLLM general-plugin swap."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_vllm_package_preserved_and_spawn_worker_patched() -> None:
    repository = Path(__file__).resolve().parent.parent
    smoke = repository / "tests" / "fixtures" / "vllm_swap_smoke.py"
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(repository), pythonpath) if item
    )
    environment.pop("HELION_ATTENTION_VLLM", None)
    environment["VLLM_PLUGINS"] = "existing_plugin"

    result = subprocess.run(
        [sys.executable, str(smoke)],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "vLLM package-preserving spawn smoke passed" in result.stdout


@pytest.mark.parametrize(
    ("allowlist", "expectation"),
    [
        pytest.param(
            "existing_plugin,helion_attention", "patched", id="enabled"
        ),
        pytest.param("existing_plugin", "unpatched", id="filtered-out"),
    ],
)
def test_ray_style_worker_respects_propagated_vllm_plugins(
    allowlist: str, expectation: str
) -> None:
    repository = Path(__file__).resolve().parent.parent
    smoke = repository / "tests" / "fixtures" / "vllm_ray_worker_smoke.py"
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(repository), pythonpath) if item
    )
    environment.pop("HELION_ATTENTION_VLLM", None)
    environment["VLLM_PLUGINS"] = allowlist

    result = subprocess.run(
        [sys.executable, str(smoke), expectation],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Ray-style worker plugin check passed" in result.stdout
