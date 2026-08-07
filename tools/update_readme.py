"""Refresh the generated sections of README.md from the manifest and benchmarks.

    python benchmarks/bench.py --json > docs/benchmarks.json
    python tools/update_readme.py

The shape table and the benchmark table live between HTML comment markers so
that the prose around them is never touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
MANIFEST = REPO_ROOT / "helion_attention" / "kernels" / "manifest.json"
BENCHMARKS = REPO_ROOT / "docs" / "benchmarks.json"


def replace_section(text: str, name: str, body: str) -> str:
    start = f"<!-- {name}:START -->"
    end = f"<!-- {name}:END -->"
    if start not in text or end not in text:
        raise SystemExit(f"README.md is missing the {name} markers")
    head = text.split(start)[0]
    tail = text.split(end)[1]
    return f"{head}{start}\n{body}\n{end}{tail}"


def shape_table() -> str:
    kernels = json.loads(MANIFEST.read_text())["kernels"]
    lines = [
        "| batch | seqlen q | seqlen k | heads q | heads kv | head dim | dtype | causal | note |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for entry in kernels:
        lines.append(
            f"| {entry['batch']} | {entry['seqlen_q']} | {entry['seqlen_k']} "
            f"| {entry['nheads_q']} | {entry['nheads_kv']} | {entry['head_dim']} "
            f"| {entry['dtype']} | {'yes' if entry['causal'] else 'no'} "
            f"| {entry.get('note', '')} |"
        )
    lines.append("")
    lines.append(f"{len(kernels)} kernels.")
    return "\n".join(lines)


def benchmark_table() -> str:
    if not BENCHMARKS.exists():
        return "_No benchmark run recorded yet._"
    report = json.loads(BENCHMARKS.read_text())
    names = ["helion-attention", "flash-attn", "sdpa-flash", "sdpa-cudnn"]
    lines = [
        f"Measured on {report['device']} (torch {report['torch']}, triton {report['triton']}). "
        "Times are the median of interleaved rounds; lower is better.",
        "",
        "| shape | "
        + " | ".join(f"{name} µs" for name in names)
        + " | helion TFLOP/s | speedup vs flash-attn |",
        "| --- | " + " | ".join("---:" for _ in names) + " | ---: | ---: |",
    ]
    speedups = []
    for row in report["results"]:
        impls = row["implementations"]
        cells = [
            "n/a" if impls.get(name) is None else f"{impls[name]['us']:.0f}" for name in names
        ]
        helion = impls.get("helion-attention")
        flash = impls.get("flash-attn")
        if helion is not None and flash is not None:
            speedup = flash["us"] / helion["us"]
            speedups.append(speedup)
            speedup_cell = f"**{speedup:.2f}x**" if speedup >= 1 else f"{speedup:.2f}x"
        else:
            speedup_cell = "n/a"
        tflops = "n/a" if helion is None else f"{helion['tflops']:.0f}"
        lines.append(
            f"| {row['description']} | " + " | ".join(cells) + f" | {tflops} | {speedup_cell} |"
        )
    if speedups:
        geomean = __import__("math").exp(sum(map(__import__("math").log, speedups)) / len(speedups))
        lines.append("")
        lines.append(
            f"Geomean speedup over flash-attn across all {len(speedups)} shapes: **{geomean:.2f}x**."
        )
    return "\n".join(lines)


def main() -> int:
    text = README.read_text()
    text = replace_section(text, "SHAPES", shape_table())
    text = replace_section(text, "BENCHMARKS", benchmark_table())
    README.write_text(text)
    print("README.md updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
