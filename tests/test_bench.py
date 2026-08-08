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


def test_markdown_reports_fa3_and_fastest_baseline_plainly(
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
    assert "flash-attn-3 µs" in table
    assert "2.00x faster" in table
    assert "2.00x slower" in table
    assert "Against FlashAttention 3" in table
    assert "fastest measured implementation on 1 of 2 workloads" in table


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
