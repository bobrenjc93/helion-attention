"""Burner evaluation: how much of the shape space is covered, and how fast is it?

The score rewards two things equally: shipping kernels for more shapes, and
each kernel beating FlashAttention on the shape it was specialized for.

    python tools/eval_benchmark.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_TARGET = 24  # kernels at which the coverage half of the score saturates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one round, fewer shapes")
    args = parser.parse_args()

    argv = [sys.executable, "benchmarks/bench.py", "--json", "--rounds", "1" if args.quick else "3"]
    completed = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=5400)
    if completed.returncode != 0:
        json.dump(
            {
                "score": 0,
                "summary": "benchmark harness failed to run",
                "evidence": (completed.stdout + completed.stderr)[-4000:],
                "suggestions": "fix benchmarks/bench.py so the benchmark can run at all",
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    report = json.loads(completed.stdout[completed.stdout.index("{") :])
    speedups: list[float] = []
    rows: list[str] = []
    losses: list[str] = []
    for row in report["results"]:
        impls = row["implementations"]
        helion = impls.get("helion-attention")
        flash = impls.get("flash-attn") or impls.get("sdpa-flash")
        if helion is None or flash is None:
            continue
        speedup = flash["us"] / helion["us"]
        speedups.append(speedup)
        rows.append(
            f"{row['key']}: helion {helion['us']:.0f}us vs flash {flash['us']:.0f}us "
            f"({speedup:.2f}x, {helion['tflops']:.0f} TFLOP/s)"
        )
        if speedup < 1.0:
            losses.append(f"{row['key']} ({speedup:.2f}x)")

    if not speedups:
        json.dump(
            {
                "score": 0,
                "summary": "no comparable measurements",
                "evidence": completed.stdout[-4000:],
                "suggestions": "check that flash-attn is installed in the benchmark environment",
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    geomean = math.exp(statistics.fmean(math.log(value) for value in speedups))
    coverage = min(1.0, len(speedups) / COVERAGE_TARGET)
    speed = min(1.0, max(0.0, (geomean - 0.5) / 1.5))
    score = round(100 * (0.5 * coverage + 0.5 * speed))

    suggestions = []
    if losses:
        suggestions.append("shapes still slower than FlashAttention: " + ", ".join(losses))
    if coverage < 1.0:
        suggestions.append(
            f"only {len(speedups)} of {COVERAGE_TARGET} target shapes are covered; "
            "add shapes to tools/shapes.py and autotune them"
        )

    json.dump(
        {
            "score": score,
            "summary": (
                f"{len(speedups)} kernels, geomean {geomean:.2f}x vs FlashAttention, "
                f"{len(losses)} slower than FlashAttention"
            ),
            "evidence": "\n".join(rows),
            "suggestions": " ".join(suggestions) or "every kernel beats FlashAttention",
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
