"""Benchmark wiring regressions that do not require a GPU."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import torch

from helion_attention._shape import AttnShape

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_PATH = REPO_ROOT / "benchmarks" / "bench.py"
SPEC = importlib.util.spec_from_file_location("helion_attention_bench", BENCH_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


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

    mask, is_causal = bench.sdpa_causal_options(decode, decode_mask)
    assert mask is None
    assert not is_causal

    chunked_prefill = replace(decode, seqlen_q=64, seqlen_k=320)
    chunked_mask = torch.ones((64, 320), dtype=torch.bool)
    mask, is_causal = bench.sdpa_causal_options(chunked_prefill, chunked_mask)
    assert mask is chunked_mask
    assert not is_causal

    equal_length = replace(decode, seqlen_q=64, seqlen_k=64)
    equal_mask = torch.ones((64, 64), dtype=torch.bool)
    mask, is_causal = bench.sdpa_causal_options(equal_length, equal_mask)
    assert mask is None
    assert is_causal

    noncausal = replace(chunked_prefill, causal=False)
    mask, is_causal = bench.sdpa_causal_options(noncausal, None)
    assert mask is None
    assert not is_causal
