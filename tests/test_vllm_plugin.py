"""Integration coverage for the vLLM general-plugin swap."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_vllm_package_preserved_and_spawn_worker_patched() -> None:
    repository = Path(__file__).resolve().parent.parent
    smoke = repository / "tests" / "fixtures" / "vllm_swap_smoke.py"
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(repository), pythonpath) if item
    )

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
