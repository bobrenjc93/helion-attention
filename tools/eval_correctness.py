"""Burner evaluation: does every checked-in kernel still compute attention?

Prints one JSON object with a 0-100 score. Run directly:
    python tools/eval_correctness.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(argv: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600
    )
    return completed.returncode, (completed.stdout + completed.stderr)[-4000:]


def main() -> int:
    verify_code, verify_output = run([sys.executable, "tools/verify.py"])
    test_code, test_output = run([sys.executable, "-m", "pytest", "-q", "tests"])

    kernels = 0
    manifest = REPO_ROOT / "helion_attention" / "kernels" / "manifest.json"
    if manifest.exists():
        kernels = len(json.loads(manifest.read_text())["kernels"])

    score = 0
    if kernels:
        score += 20
    if verify_code == 0:
        score += 40
    if test_code == 0:
        score += 40

    suggestions = []
    if verify_code != 0:
        suggestions.append("tools/verify.py reports a kernel that disagrees with fp32 SDPA")
    if test_code != 0:
        suggestions.append("pytest is failing; see the evidence for the failing cases")
    if not kernels:
        suggestions.append("no kernels are checked in; run tools/generate_all.py")

    json.dump(
        {
            "score": score,
            "summary": (
                f"{kernels} kernels checked in; "
                f"verify {'passed' if verify_code == 0 else 'FAILED'}; "
                f"pytest {'passed' if test_code == 0 else 'FAILED'}"
            ),
            "evidence": [
                f"$ tools/verify.py\n{verify_output}",
                f"$ pytest -q tests\n{test_output}",
            ],
            "suggestions": suggestions or ["correctness is clean"],
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
