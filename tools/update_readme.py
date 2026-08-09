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
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.timing import timing_methodology  # noqa: E402

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


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds == 0:
        return "0 s"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _format_configs(configs: list[dict[str, object]]) -> str:
    if not configs:
        return "unknown (legacy artifact)"
    rendered = []
    for item in configs:
        config = item["config"]
        assert isinstance(config, dict)
        arguments = ", ".join(
            f"{name}={value!r}" for name, value in sorted(config.items())
        )
        label = f"{item['kernel']}: " if len(configs) > 1 else ""
        rendered.append(f"{label}`helion.Config({arguments})`")
    return "<br>".join(rendered)


def _format_rejected_search(provenance: dict[str, object]) -> str:
    rejected = provenance.get("rejected_search")
    if not isinstance(rejected, dict):
        return "—"
    configs = rejected["configs"]
    assert isinstance(configs, list)
    return (
        f"Helion {rejected['helion_version']}, "
        f"{_format_duration(float(rejected['autotuning_wall_time_seconds']))}; "
        f"candidate {float(rejected['candidate_measured_time_ms']):.6f} ms, "
        f"incumbent {float(rejected['incumbent_measured_time_ms']):.6f} ms"
        f"<br>{_format_configs(configs)}"
    )


def _format_selection(provenance: dict[str, object]) -> str:
    selection = str(provenance["selection"])
    if "artifact_origin_selection" not in provenance:
        return selection
    origin = provenance["artifact_origin_selection"]
    return f"{selection} (origin: {origin or 'unknown'})"


def autotuning_table() -> str:
    """Render generation provenance for every checked-in kernel module."""
    manifest = json.loads(MANIFEST.read_text())
    lines = [
        "| kernel | Helion | selection | tuning wall time | measured time | chosen config | rejected search |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for section in ("kernels", "varlen_kernels", "paged_kernels"):
        for entry in manifest.get(section, []):
            rows = [
                (
                    str(entry["key"]),
                    entry["autotuning_provenance"],
                    f"helion_attention/kernels/{entry['key']}.py",
                )
            ]
            if entry.get("backward"):
                rows.append(
                    (
                        f"{entry['key']}_backward",
                        entry["backward_autotuning_provenance"],
                        f"helion_attention/kernels/{entry['key']}_backward.py",
                    )
                )
            for name, provenance, module_path in rows:
                assert isinstance(provenance, dict)
                configs = provenance["configs"]
                assert isinstance(configs, list)
                wall_time = provenance.get("autotuning_wall_time_seconds")
                lines.append(
                    f"| [`{name}`]({module_path}) "
                    f"| {provenance.get('helion_version') or 'unknown'} "
                    f"| {_format_selection(provenance)} "
                    f"| {_format_duration(None if wall_time is None else float(wall_time))} "
                    f"| {float(provenance['measured_time_ms']):.6f} ms "
                    f"| {_format_configs(configs)} "
                    f"| {_format_rejected_search(provenance)} |"
                )
    return "\n".join(lines)


def benchmark_table() -> str:
    if not BENCHMARKS.exists():
        return "_No benchmark run recorded yet._"
    report = json.loads(BENCHMARKS.read_text())
    names = [
        "helion-attention",
        "flash-attn-3",
        "flash-attn",
        "sdpa-flash",
        "sdpa-cudnn",
    ]
    flash_versions = []
    if report.get("flash_attn"):
        flash_versions.append(f"FA2 {report['flash_attn']}")
    if report.get("flash_attn_3"):
        flash_versions.append(f"FA3 {report['flash_attn_3']}")
    version_suffix = f", {', '.join(flash_versions)}" if flash_versions else ""
    methodology = timing_methodology()
    lines = [
        f"Measured on {report['device']} (torch {report['torch']}, "
        f"triton {report['triton']}{version_suffix}). "
        f"{methodology}",
        "",
    ]
    kinds = {row.get("kind") for row in report["results"]}
    notes = []
    if "paged" in kinds:
        notes.append(
            "Paged rows use identical logical caches with each implementation's native "
            "page size: 16 for Helion and FA3, and FA2's minimum of 256. FA2 "
            "paged decode uses `flash_attn_with_kvcache`; chunked prefill uses "
            "`flash_attn_varlen_func`. FA3 uses `flash_attn_with_kvcache`."
        )
    if "backward" in kinds:
        notes.append(
            "Backward rows time dQ/dK/dV only; their forward graphs are prepared outside "
            "the timed region, and TFLOP/s uses the standard 2.5x-forward operation estimate."
        )
    if notes:
        lines.extend([" ".join(notes), ""])
    lines.extend(
        [
            "| shape | "
            + " | ".join(f"{name} µs" for name in names)
            + " | helion TFLOP/s | comparison vs flash-attn-3 |",
            "| --- | " + " | ".join("---:" for _ in names) + " | ---: | ---: |",
        ]
    )
    flash_3_speedups = []
    faster_than_flash_3 = 0
    slower_than_flash_3 = 0
    best_baseline_speedups = []
    for row in report["results"]:
        impls = row["implementations"]
        cells = [
            "n/a" if impls.get(name) is None else f"{impls[name]['us']:.0f}" for name in names
        ]
        helion = impls.get("helion-attention")
        flash_3 = impls.get("flash-attn-3")
        if helion is not None and flash_3 is not None:
            speedup = flash_3["us"] / helion["us"]
            flash_3_speedups.append(speedup)
            if speedup >= 1:
                faster_than_flash_3 += 1
                speedup_cell = f"**{speedup:.2f}x faster**"
            else:
                slower_than_flash_3 += 1
                speedup_cell = f"**{1 / speedup:.2f}x slower**"
        else:
            speedup_cell = "n/a"
        baselines = [impls[name] for name in names[1:] if impls.get(name) is not None]
        if helion is not None and baselines:
            fastest_baseline = min(baselines, key=lambda value: value["us"])
            best_baseline_speedups.append(fastest_baseline["us"] / helion["us"])
        tflops = "n/a" if helion is None else f"{helion['tflops']:.0f}"
        lines.append(
            f"| {row['description']} | " + " | ".join(cells) + f" | {tflops} | {speedup_cell} |"
        )
    if flash_3_speedups:
        geomean = __import__("math").exp(
            sum(map(__import__("math").log, flash_3_speedups))
            / len(flash_3_speedups)
        )
        lines.append("")
        lines.append(
            f"Against FlashAttention 3, Helion is faster on {faster_than_flash_3} "
            f"kernel workloads and slower on {slower_than_flash_3} kernel workloads."
        )
        lines.append("")
        lines.append(
            "Geomean speedup over FlashAttention 3 across all "
            f"{len(flash_3_speedups)} comparable kernel workloads: "
            f"**{geomean:.2f}x**."
        )
    if best_baseline_speedups:
        wins = sum(speedup >= 1 for speedup in best_baseline_speedups)
        lines.append("")
        lines.append(
            f"Helion is the fastest measured implementation on {wins} of "
            f"{len(best_baseline_speedups)} workloads; FA2, FA3, or a PyTorch "
            "SDPA backend is faster on the remainder."
        )
    return "\n".join(lines)


def main() -> int:
    text = README.read_text()
    text = replace_section(text, "SHAPES", shape_table())
    text = replace_section(text, "AUTOTUNING", autotuning_table())
    text = replace_section(text, "BENCHMARKS", benchmark_table())
    README.write_text(text)
    print("README.md updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
