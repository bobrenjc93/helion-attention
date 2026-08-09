"""Dependency-light Markdown rendering for benchmark reports."""

from __future__ import annotations

from benchmarks.timing import report_timing_methodology

FLASH_ATTN_PAGE_SIZE = 256
IMPLEMENTATION_NAMES = [
    "helion-attention",
    "flash-attn-3",
    "flash-attn",
    "sdpa-flash",
    "sdpa-cudnn",
]


def render_markdown(report: dict[str, object]) -> str:
    names = IMPLEMENTATION_NAMES
    flash_versions = []
    if report.get("flash_attn"):
        flash_versions.append(f"FA2 {report['flash_attn']}")
    if report.get("flash_attn_3"):
        flash_versions.append(f"FA3 {report['flash_attn_3']}")
    version_suffix = f", {', '.join(flash_versions)}" if flash_versions else ""
    lines = [
        f"Measured on {report['device']}, torch {report['torch']}, "
        f"triton {report['triton']}{version_suffix}. "
        f"{report_timing_methodology(report)}",
        "",
    ]
    kinds = {row.get("kind") for row in report["results"]}  # type: ignore[index]
    notes = []
    if "paged" in kinds:
        notes.append(
            "Paged rows use identical logical caches with native page sizes "
            f"(Helion and FA3: 16; FA2: {FLASH_ATTN_PAGE_SIZE}); FA2 paged "
            "decode uses flash_attn_with_kvcache and chunked prefill uses "
            "flash_attn_varlen_func; FA3 uses flash_attn_with_kvcache."
        )
    if "backward" in kinds:
        notes.append(
            "Backward rows time dQ/dK/dV only, excluding forward setup; "
            "TFLOP/s uses a 2.5x-forward operation estimate."
        )
    if notes:
        lines.extend([" ".join(notes), ""])
    lines.extend(
        [
            "| shape | "
            + " | ".join(f"{n} (µs)" for n in names)
            + " | helion vs flash-attn-3 |",
            "| --- | " + " | ".join("---:" for _ in names) + " | ---: |",
        ]
    )
    for row in report["results"]:  # type: ignore[index]
        impls = row["implementations"]
        cells = []
        for name in names:
            entry = impls.get(name)
            cells.append("n/a" if entry is None else f"{entry['us']:.0f}")
        helion = impls.get("helion-attention")
        flash_3 = impls.get("flash-attn-3")
        if helion is not None and flash_3 is not None:
            speedup = flash_3["us"] / helion["us"]
            comparison = (
                f"{speedup:.2f}x faster"
                if speedup >= 1
                else f"{1 / speedup:.2f}x slower"
            )
        else:
            comparison = "n/a"
        lines.append(
            f"| {row['description']} | "
            + " | ".join(cells)
            + f" | {comparison} |"
        )
    return "\n".join(lines) + "\n"
