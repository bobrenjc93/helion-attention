"""Benchmark wiring regressions that do not require a GPU."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from benchmarks.inventory import benchmark_entries
from benchmarks.inventory import benchmark_key
from benchmarks.render import render_markdown
from benchmarks.timing import timing_metadata
from helion_attention._sdpa import sdpa_causal_options
from helion_attention._shape import AttnShape

REPO_ROOT = Path(__file__).parents[1]
UPDATE_README_PATH = Path(__file__).parents[1] / "tools" / "update_readme.py"
UPDATE_README_SPEC = importlib.util.spec_from_file_location(
    "helion_attention_update_readme", UPDATE_README_PATH
)
assert UPDATE_README_SPEC is not None
assert UPDATE_README_SPEC.loader is not None
update_readme = importlib.util.module_from_spec(UPDATE_README_SPEC)
UPDATE_README_SPEC.loader.exec_module(update_readme)


def test_decode_omits_all_true_causal_mask_for_fused_sdpa() -> None:
    decode = AttnShape(
        batch=1,
        seqlen_q=1,
        seqlen_k=4096,
        nheads_q=32,
        nheads_kv=8,
        head_dim=128,
        dtype=torch.bfloat16,
        causal=True,
    )
    decode_mask = torch.ones((1, 4096), dtype=torch.bool)

    mask, is_causal = sdpa_causal_options(decode, decode_mask)
    assert mask is None
    assert not is_causal

    chunked_prefill = replace(decode, seqlen_q=64, seqlen_k=320)
    chunked_mask = torch.ones((64, 320), dtype=torch.bool)
    mask, is_causal = sdpa_causal_options(chunked_prefill, chunked_mask)
    assert mask is chunked_mask
    assert not is_causal

    equal_length = replace(decode, seqlen_q=64, seqlen_k=64)
    equal_mask = torch.ones((64, 64), dtype=torch.bool)
    mask, is_causal = sdpa_causal_options(equal_length, equal_mask)
    assert mask is None
    assert is_causal

    noncausal = replace(chunked_prefill, causal=False)
    mask, is_causal = sdpa_causal_options(noncausal, None)
    assert mask is None
    assert not is_causal


def test_readme_markdown_reports_fa3_baseline_and_legacy_timing_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "device": "test GPU",
        "torch": "test torch",
        "triton": "test triton",
        "flash_attn": "2.test",
        "flash_attn_3": "3.test",
        "results": [
            {
                "description": "faster shape",
                "implementations": {
                    "helion-attention": {"us": 1.0, "tflops": 2.0},
                    "flash-attn-3": {"us": 2.0},
                    "flash-attn": {"us": 3.0},
                },
            },
            {
                "description": "slower shape",
                "implementations": {
                    "helion-attention": {"us": 4.0, "tflops": 0.5},
                    "flash-attn-3": {"us": 2.0},
                    "flash-attn": {"us": 3.0},
                },
            },
        ],
    }

    class BenchmarkReport:
        @staticmethod
        def exists() -> bool:
            return True

        @staticmethod
        def read_text() -> str:
            return json.dumps(report)

    monkeypatch.setattr(update_readme, "BENCHMARKS", BenchmarkReport())

    table = update_readme.benchmark_table()

    assert "FA2 2.test, FA3 3.test" in table
    assert "median of three interleaved rounds" in table
    assert "50 ms warmup and 200 ms measurement" in table
    assert "flash-attn-3 µs" in table
    assert "2.00x faster" in table
    assert "2.00x slower" in table
    assert "Against FlashAttention 3" in table
    assert "fastest measured implementation on 1 of 2 workloads" in table


def test_timing_metadata_records_nondefault_rounds() -> None:
    assert timing_metadata(1) == {
        "rounds": 1,
        "warmup_ms": 50,
        "measurement_ms": 200,
    }


def test_standalone_markdown_reports_recorded_timing() -> None:
    table = render_markdown(
        {
            "device": "test GPU",
            "torch": "test torch",
            "triton": "test triton",
            "timing": {
                "rounds": 1,
                "warmup_ms": 75,
                "measurement_ms": 300,
            },
            "results": [],
        }
    )

    assert "median of one interleaved round" in table
    assert "75 ms warmup and 300 ms measurement" in table
    assert "three interleaved rounds" not in table


def test_readme_renderer_reports_recorded_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "device": "test GPU",
        "torch": "test torch",
        "triton": "test triton",
        "timing": {
            "rounds": 1,
            "warmup_ms": 75,
            "measurement_ms": 300,
        },
        "results": [],
    }

    class BenchmarkReport:
        @staticmethod
        def exists() -> bool:
            return True

        @staticmethod
        def read_text() -> str:
            return json.dumps(report)

    monkeypatch.setattr(update_readme, "BENCHMARKS", BenchmarkReport())

    table = update_readme.benchmark_table()

    assert "median of one interleaved round" in table
    assert "75 ms warmup and 300 ms measurement" in table
    assert "three interleaved rounds" not in table


def test_readme_regeneration_preserves_benchmark_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = update_readme.README.read_text()

    class InMemoryReadme:
        text = original

        @classmethod
        def read_text(cls) -> str:
            return cls.text

        @classmethod
        def write_text(cls, text: str) -> None:
            cls.text = text

    def section(text: str, start: str, end: str) -> str:
        return text.split(start, 1)[1].split(end, 1)[0]

    benchmark_start = "<!-- BENCHMARKS:START -->"
    benchmark_end = "<!-- BENCHMARKS:END -->"
    original_benchmark_rows = [
        line
        for line in section(original, benchmark_start, benchmark_end).splitlines()
        if line.startswith("|")
    ]
    monkeypatch.setattr(update_readme, "README", InMemoryReadme)

    assert update_readme.main() == 0

    regenerated = InMemoryReadme.text
    regenerated_benchmark_rows = [
        line
        for line in section(regenerated, benchmark_start, benchmark_end).splitlines()
        if line.startswith("|")
    ]
    assert regenerated_benchmark_rows == original_benchmark_rows


def test_readme_autotuning_table_covers_every_generated_module() -> None:
    manifest = json.loads(update_readme.MANIFEST.read_text())
    expected = []
    for section in ("kernels", "varlen_kernels", "paged_kernels"):
        for entry in manifest[section]:
            expected.append(str(entry["key"]))
            if entry.get("backward"):
                expected.append(f"{entry['key']}_backward")

    table = update_readme.autotuning_table()

    for key in expected:
        assert table.count(f"[`{key}`]") == 1
    assert table.count(" ms |") == len(expected)
    assert "helion.Config(" in table
    assert "| fixed | 0 s |" in table


def test_discovery_and_report_cover_every_checked_in_kernel() -> None:
    manifest = json.loads(
        (REPO_ROOT / "helion_attention" / "kernels" / "manifest.json").read_text()
    )
    expected = []
    for entry in manifest["kernels"]:
        expected.append(entry["key"])
        if entry.get("backward", False):
            expected.append(f"{entry['key']}_backward")
    for section in ("varlen_kernels", "paged_kernels"):
        expected.extend(entry["key"] for entry in manifest[section])

    discovered = benchmark_entries()
    discovered_keys = [benchmark_key(entry, kind) for entry, kind in discovered]
    artifact_keys = {
        path.stem
        for path in (REPO_ROOT / "helion_attention" / "kernels").glob("*.py")
        if path.name != "__init__.py"
    }
    report = json.loads((REPO_ROOT / "docs" / "benchmarks.json").read_text())
    report_by_key = {row["key"]: row for row in report["results"]}

    assert report["timing"] == timing_metadata()
    assert discovered_keys == expected
    assert set(discovered_keys) == artifact_keys
    assert [row["key"] for row in report["results"]] == expected
    assert [kind for _, kind in benchmark_entries("paged")] == [
        "paged",
        "paged",
    ]
    assert [kind for _, kind in benchmark_entries("backward")] == [
        "backward"
    ]
    assert (
        report_by_key["paged_b2_sq200_sk320_hq8_hkv2_d128_bf16_causal_ps16"][
            "flash_attn_api"
        ]
        == "flash_attn_varlen_func"
    )
    assert (
        report_by_key["paged_b4_sq1_sk1024_hq8_hkv2_d128_bf16_causal_ps16"][
            "flash_attn_api"
        ]
        == "flash_attn_with_kvcache"
    )
    assert all("flash_attn_3_api" in row for row in report["results"])
