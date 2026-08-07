"""Benchmark every checked-in kernel against FlashAttention and PyTorch SDPA.

Usage:
    python benchmarks/bench.py                 # markdown table on stdout
    python benchmarks/bench.py --json          # machine readable
    python benchmarks/bench.py --markdown docs/benchmarks.md

Implementations are measured round-robin over several rounds and the median of
each implementation's rounds is reported, so a clock drift partway through the
run cannot make whichever kernel ran first look good.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import triton

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import helion_attention  # noqa: E402
from helion_attention._registry import available_shapes  # noqa: E402
from helion_attention._registry import available_varlen_shapes  # noqa: E402
from helion_attention._registry import spec_from_manifest_entry  # noqa: E402
from helion_attention._shape import AttnShape  # noqa: E402

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except ImportError:  # pragma: no cover - benchmark-only dependency
    _flash_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:  # pragma: no cover - benchmark-only dependency
    _flash_attn_varlen_func = None

try:
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
except ImportError:  # pragma: no cover - benchmark-only dependency
    _flash_attn_with_kvcache = None

from torch.nn.attention import SDPBackend  # noqa: E402
from torch.nn.attention import sdpa_kernel  # noqa: E402


def flops(spec: AttnShape) -> float:
    total = (
        4.0
        * spec.batch
        * spec.nheads_q
        * spec.seqlen_q
        * spec.seqlen_k
        * spec.head_dim
    )
    return total * 0.5 if spec.causal and not spec.is_decode else total


def build_candidates(
    spec: AttnShape,
) -> tuple[dict[str, Callable[[], torch.Tensor]], torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)

    def rand(seqlen: int, nheads: int) -> torch.Tensor:
        return torch.randn(
            (spec.batch, seqlen, nheads, spec.head_dim),
            device="cuda",
            dtype=spec.dtype,
            generator=generator,
        )

    q = rand(spec.seqlen_q, spec.nheads_q)
    k = rand(spec.seqlen_k, spec.nheads_kv)
    v = rand(spec.seqlen_k, spec.nheads_kv)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    gqa = spec.nheads_q != spec.nheads_kv

    # SDPA's unequal-length causal mask is top-left aligned. Single-token
    # decode uses FlashAttention's bottom-right alignment, where the newest
    # query can see the complete cache.
    sdpa_is_causal = spec.causal and not spec.is_decode

    def sdpa(backend: SDPBackend) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            with sdpa_kernel(backend):
                return torch.nn.functional.scaled_dot_product_attention(
                    qt, kt, vt, is_causal=sdpa_is_causal, enable_gqa=gqa
                ).transpose(1, 2)

        return run

    if spec.is_decode:
        candidates: dict[str, Callable[[], torch.Tensor]] = {
            "helion-attention": lambda: helion_attention.flash_attn_with_kvcache(
                q, k, v, causal=spec.causal, shape=spec
            )
        }
        if _flash_attn_with_kvcache is not None:
            candidates["flash-attn"] = lambda: _flash_attn_with_kvcache(
                q, k, v, causal=spec.causal
            )
    else:
        candidates = {
            "helion-attention": lambda: helion_attention.flash_attn_func(
                q, k, v, causal=spec.causal, shape=spec
            )
        }
        if _flash_attn_func is not None:
            candidates["flash-attn"] = lambda: _flash_attn_func(
                q, k, v, causal=spec.causal
            )
    candidates["sdpa-flash"] = sdpa(SDPBackend.FLASH_ATTENTION)
    candidates["sdpa-cudnn"] = sdpa(SDPBackend.CUDNN_ATTENTION)

    reference = torch.nn.functional.scaled_dot_product_attention(
        qt.float(),
        kt.float(),
        vt.float(),
        is_causal=sdpa_is_causal,
        scale=1.0 / math.sqrt(spec.head_dim),
        enable_gqa=gqa,
    ).transpose(1, 2)
    return candidates, reference


def build_varlen_candidates(
    spec: AttnShape,
) -> tuple[dict[str, Callable[[], torch.Tensor]], torch.Tensor, float]:
    """Build a deterministic ragged workload for one maximum-shape kernel."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    lengths_q = [
        max(1, spec.seqlen_q * (spec.batch - index) // spec.batch)
        for index in range(spec.batch)
    ]
    lengths_k = list(
        reversed(
            [
                max(1, spec.seqlen_k * (spec.batch - index) // spec.batch)
                for index in range(spec.batch)
            ]
        )
    )

    def cumulative(lengths: list[int]) -> torch.Tensor:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, device="cuda", dtype=torch.int32)

    q = torch.randn(
        (sum(lengths_q), spec.nheads_q, spec.head_dim),
        device="cuda",
        dtype=spec.dtype,
        generator=generator,
    )
    k = torch.randn(
        (sum(lengths_k), spec.nheads_kv, spec.head_dim),
        device="cuda",
        dtype=spec.dtype,
        generator=generator,
    )
    v = torch.randn(
        k.shape,
        device="cuda",
        dtype=spec.dtype,
        generator=generator,
    )
    cu_q = cumulative(lengths_q)
    cu_k = cumulative(lengths_k)
    args = (q, k, v, cu_q, cu_k, spec.seqlen_q, spec.seqlen_k)
    candidates: dict[str, Callable[[], torch.Tensor]] = {
        "helion-attention": lambda: helion_attention.flash_attn_varlen_func(
            *args, causal=spec.causal, shape=spec
        )
    }
    if _flash_attn_varlen_func is not None:
        candidates["flash-attn"] = lambda: _flash_attn_varlen_func(
            *args, causal=spec.causal
        )

    reference = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    q_start = 0
    k_start = 0
    attended_pairs = 0
    for seqlen_q, seqlen_k in zip(lengths_q, lengths_k):
        q_seq = q[q_start : q_start + seqlen_q].float().transpose(0, 1)[None]
        k_seq = k[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        v_seq = v[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        mask = None
        if spec.causal:
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            mask = col <= row + seqlen_k - seqlen_q
            attended_pairs += int(mask.sum().item())
        else:
            attended_pairs += seqlen_q * seqlen_k
        result = torch.nn.functional.scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            attn_mask=mask,
            scale=1.0 / math.sqrt(spec.head_dim),
            enable_gqa=spec.nheads_q != spec.nheads_kv,
        )
        reference[q_start : q_start + seqlen_q] = result[0].transpose(0, 1)
        q_start += seqlen_q
        k_start += seqlen_k
    operation_count = (
        4.0 * spec.nheads_q * spec.head_dim * attended_pairs
    )
    return candidates, reference, operation_count


def measure(
    candidates: dict[str, Callable[[], torch.Tensor]],
    reference: torch.Tensor,
    rounds: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, list[float]] = {name: [] for name in candidates}
    errors: dict[str, float] = {}
    for name, run in candidates.items():
        try:
            errors[name] = (run().float() - reference).abs().max().item()
        except Exception as error:  # noqa: BLE001
            errors[name] = float("nan")
            print(f"  {name}: unavailable ({type(error).__name__}: {error})", file=sys.stderr)
    live = [name for name in candidates if not math.isnan(errors[name])]
    for _ in range(rounds):
        for name in live:
            results[name].append(triton.testing.do_bench(candidates[name], warmup=50, rep=200))
    return {
        name: {
            "ms": statistics.median(results[name]),
            "max_abs_error": errors[name],
        }
        for name in live
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--markdown", type=Path, default=None, help="also write a table here")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--only", default="", help="substring filter on the kernel key")
    args = parser.parse_args()

    entries = [
        (entry, False)
        for entry in available_shapes()
        if args.only in str(entry["key"])
    ] + [
        (entry, True)
        for entry in available_varlen_shapes()
        if args.only in str(entry["key"])
    ]
    if not entries:
        print("no kernels to benchmark", file=sys.stderr)
        return 1

    device_name = torch.cuda.get_device_name(0)
    report: dict[str, object] = {
        "device": device_name,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "results": [],
    }
    for entry, is_varlen in entries:
        spec = spec_from_manifest_entry(entry)
        print(f"benchmarking {spec.describe()}", file=sys.stderr)
        if is_varlen:
            candidates, reference, scale = build_varlen_candidates(spec)
        else:
            candidates, reference = build_candidates(spec)
            scale = flops(spec)
        measured = measure(candidates, reference, args.rounds)
        row = {
            "key": entry["key"],
            "description": entry["description"],
            "varlen": is_varlen,
            "note": entry.get("note", ""),
            "implementations": {
                name: {
                    "us": value["ms"] * 1000.0,
                    "tflops": scale / value["ms"] / 1e9,
                    "max_abs_error": value["max_abs_error"],
                }
                for name, value in measured.items()
            },
        }
        report["results"].append(row)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    table = render_markdown(report)
    if not args.json:
        sys.stdout.write(table)
    if args.markdown:
        args.markdown.write_text(table)
    return 0


def render_markdown(report: dict[str, object]) -> str:
    names = ["helion-attention", "flash-attn", "sdpa-flash", "sdpa-cudnn"]
    lines = [
        f"Measured on {report['device']}, torch {report['torch']}, triton {report['triton']}.",
        "",
        "| shape | " + " | ".join(f"{n} (µs)" for n in names) + " | helion vs flash-attn |",
        "| --- | " + " | ".join("---:" for _ in names) + " | ---: |",
    ]
    for row in report["results"]:  # type: ignore[index]
        impls = row["implementations"]
        cells = []
        for name in names:
            entry = impls.get(name)
            cells.append("n/a" if entry is None else f"{entry['us']:.0f}")
        helion = impls.get("helion-attention")
        flash = impls.get("flash-attn")
        speedup = (
            f"{flash['us'] / helion['us']:.2f}x"
            if helion is not None and flash is not None
            else "n/a"
        )
        lines.append(f"| {row['description']} | " + " | ".join(cells) + f" | {speedup} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
