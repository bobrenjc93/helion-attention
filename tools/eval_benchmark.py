"""Burner evaluation: how much of the shape space is covered, and how fast is it?

Only kernels that beat FlashAttention count toward coverage, so adding a shape
that loses to the library we are replacing cannot buy a higher score.

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
                "evidence": [(completed.stdout + completed.stderr)[-4000:]],
                "suggestions": ["fix benchmarks/bench.py so the benchmark can run at all"],
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
                "evidence": [completed.stdout[-4000:]],
                "suggestions": ["check that flash-attn is installed in the benchmark environment"],
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    geomean = math.exp(statistics.fmean(math.log(value) for value in speedups))
    # Coverage counts only kernels that actually beat FlashAttention. A kernel
    # that loses on the shape it was specialized for is not coverage; it is a
    # slower answer than the library it replaces, so it earns nothing here and
    # still drags the geomean down.
    wins = sum(1 for value in speedups if value >= 1.0)
    coverage = min(1.0, wins / COVERAGE_TARGET)
    speed = min(1.0, max(0.0, (geomean - 0.5) / 1.5))
    score = round(100 * (0.6 * coverage + 0.4 * speed))

    suggestions = []
    if losses:
        suggestions.append(
            "these shapes are slower than FlashAttention and count for nothing until "
            "they are fixed; prefer making them faster over adding new shapes: "
            + ", ".join(losses)
        )
    if coverage < 1.0:
        suggestions.append(
            f"only {wins} of {COVERAGE_TARGET} target shapes beat FlashAttention "
            f"({len(speedups)} kernels are checked in)"
        )

    json.dump(
        {
            "score": score,
            "summary": (
                f"{wins}/{len(speedups)} kernels beat FlashAttention, "
                f"geomean {geomean:.2f}x, {len(losses)} slower than FlashAttention"
            ),
            "evidence": rows[:8],
            "suggestions": suggestions or ["every kernel beats FlashAttention"],
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
