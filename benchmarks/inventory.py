"""Dependency-light discovery of every checked-in generated kernel artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

BenchmarkKind = Literal["forward", "backward", "varlen", "paged"]

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "helion_attention" / "kernels" / "manifest.json"


def benchmark_key(entry: dict[str, object], kind: BenchmarkKind) -> str:
    key = str(entry["key"])
    return f"{key}_backward" if kind == "backward" else key


def benchmark_entries(
    only: str = "", manifest_path: Path = MANIFEST
) -> list[tuple[dict[str, object], BenchmarkKind]]:
    """Return every generated artifact in manifest order without runtime imports."""
    manifest = json.loads(manifest_path.read_text())
    entries: list[tuple[dict[str, object], BenchmarkKind]] = []
    for entry in manifest.get("kernels", []):
        entries.append((entry, "forward"))
        if entry.get("backward", False):
            entries.append((entry, "backward"))
    entries.extend(
        (entry, "varlen") for entry in manifest.get("varlen_kernels", [])
    )
    entries.extend((entry, "paged") for entry in manifest.get("paged_kernels", []))
    return [
        (entry, kind)
        for entry, kind in entries
        if only in benchmark_key(entry, kind)
    ]
