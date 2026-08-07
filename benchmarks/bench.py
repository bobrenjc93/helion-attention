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
import helion_attention.vllm_flash_attn as vllm_flash_attn  # noqa: E402
from benchmarks.inventory import BenchmarkKind  # noqa: E402
from benchmarks.inventory import benchmark_entries  # noqa: E402
from benchmarks.inventory import benchmark_key  # noqa: E402
from helion_attention._registry import spec_from_manifest_entry  # noqa: E402
from helion_attention._sdpa import sdpa_causal_options  # noqa: E402
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

BenchmarkOutput = torch.Tensor | tuple[torch.Tensor, ...]
Candidate = Callable[[], BenchmarkOutput]
FLASH_ATTN_PAGE_SIZE = 256


def attended_pairs(seqlen_q: int, seqlen_k: int, causal: bool) -> int:
    if not causal:
        return seqlen_q * seqlen_k
    offset = seqlen_k - seqlen_q
    return sum(
        max(0, min(seqlen_k, row + offset + 1))
        for row in range(seqlen_q)
    )


def flops(spec: AttnShape) -> float:
    return (
        4.0
        * spec.batch
        * spec.nheads_q
        * attended_pairs(spec.seqlen_q, spec.seqlen_k, spec.causal)
        * spec.head_dim
    )


def build_candidates(
    spec: AttnShape,
) -> tuple[dict[str, Candidate], torch.Tensor]:
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

    causal_mask = None
    if spec.causal:
        row = torch.arange(spec.seqlen_q, device=q.device)[:, None]
        col = torch.arange(spec.seqlen_k, device=q.device)[None, :]
        causal_mask = col <= row + spec.seqlen_k - spec.seqlen_q

    # Fused SDPA can represent equal-length triangular masks through
    # is_causal. Unequal lengths need the explicit bottom-right mask, except
    # single-token decode where that mask is entirely true and can be omitted.
    sdpa_mask, sdpa_is_causal = sdpa_causal_options(spec, causal_mask)

    def sdpa(backend: SDPBackend) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            with sdpa_kernel(backend):
                return torch.nn.functional.scaled_dot_product_attention(
                    qt,
                    kt,
                    vt,
                    attn_mask=sdpa_mask,
                    is_causal=sdpa_is_causal,
                    enable_gqa=gqa,
                ).transpose(1, 2)

        return run

    if spec.is_decode:
        candidates: dict[str, Candidate] = {
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
        attn_mask=causal_mask,
        scale=1.0 / math.sqrt(spec.head_dim),
        enable_gqa=gqa,
    ).transpose(1, 2)
    return candidates, reference


def ragged_lengths(
    max_length: int, batch: int, *, reverse: bool = False
) -> list[int]:
    lengths = [
        max(1, max_length * (batch - index) // batch)
        for index in range(batch)
    ]
    return list(reversed(lengths)) if reverse else lengths


def cumulative(lengths: list[int]) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device="cuda", dtype=torch.int32)


def packed_reference(
    q: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    lengths_q: list[int],
    spec: AttnShape,
) -> tuple[torch.Tensor, float]:
    reference = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    q_start = 0
    total_pairs = 0
    for seqlen_q, key, value in zip(lengths_q, keys, values):
        seqlen_k = key.shape[0]
        q_seq = q[q_start : q_start + seqlen_q].float().transpose(0, 1)[None]
        k_seq = key.float().transpose(0, 1)[None]
        v_seq = value.float().transpose(0, 1)[None]
        mask = None
        if spec.causal:
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            mask = col <= row + seqlen_k - seqlen_q
        total_pairs += attended_pairs(seqlen_q, seqlen_k, spec.causal)
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
    operation_count = 4.0 * spec.nheads_q * spec.head_dim * total_pairs
    return reference, operation_count


def build_varlen_candidates(
    spec: AttnShape,
) -> tuple[dict[str, Candidate], torch.Tensor, float]:
    """Build a deterministic ragged workload for one maximum-shape kernel."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    lengths_q = ragged_lengths(spec.seqlen_q, spec.batch)
    lengths_k = ragged_lengths(spec.seqlen_k, spec.batch, reverse=True)

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
    candidates: dict[str, Candidate] = {
        "helion-attention": lambda: helion_attention.flash_attn_varlen_func(
            *args, causal=spec.causal, shape=spec
        )
    }
    if _flash_attn_varlen_func is not None:
        candidates["flash-attn"] = lambda: _flash_attn_varlen_func(
            *args, causal=spec.causal
        )

    reference, operation_count = packed_reference(
        q,
        list(k.split(lengths_k)),
        list(v.split(lengths_k)),
        lengths_q,
        spec,
    )
    return candidates, reference, operation_count


def pack_paged_cache(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    *,
    page_size: int,
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Store logical sequences in reverse physical order behind a block table."""
    max_blocks = math.ceil(max_seqlen_k / page_size)
    blocks_per_sequence = [math.ceil(key.shape[0] / page_size) for key in keys]
    total_blocks = sum(blocks_per_sequence)
    cache_shape = (total_blocks, page_size, keys[0].shape[1], keys[0].shape[2])
    key_cache = torch.zeros(cache_shape, device=keys[0].device, dtype=keys[0].dtype)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros(
        (len(keys), max_blocks), device=keys[0].device, dtype=torch.int32
    )

    physical_block = total_blocks - 1
    for batch_index, (key, value, block_count) in enumerate(
        zip(keys, values, blocks_per_sequence)
    ):
        for logical_block in range(block_count):
            block_table[batch_index, logical_block] = physical_block
            start = logical_block * page_size
            stop = min(start + page_size, key.shape[0])
            key_cache[physical_block, : stop - start] = key[start:stop]
            value_cache[physical_block, : stop - start] = value[start:stop]
            physical_block -= 1
    return key_cache, value_cache, block_table


def build_paged_candidates(
    spec: AttnShape,
    page_size: int,
) -> tuple[dict[str, Candidate], torch.Tensor, float]:
    """Build equivalent native paged-cache workloads for Helion and FlashAttention."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    lengths_q = ragged_lengths(spec.seqlen_q, spec.batch)
    lengths_k = ragged_lengths(spec.seqlen_k, spec.batch, reverse=True)
    q = torch.randn(
        (sum(lengths_q), spec.nheads_q, spec.head_dim),
        device="cuda",
        dtype=spec.dtype,
        generator=generator,
    )
    keys = [
        torch.randn(
            (length, spec.nheads_kv, spec.head_dim),
            device="cuda",
            dtype=spec.dtype,
            generator=generator,
        )
        for length in lengths_k
    ]
    values = [
        torch.randn(
            key.shape,
            device="cuda",
            dtype=spec.dtype,
            generator=generator,
        )
        for key in keys
    ]
    cu_q = cumulative(lengths_q)
    cu_k = cumulative(lengths_k)
    seqused_k = torch.tensor(lengths_k, device="cuda", dtype=torch.int32)
    helion_k, helion_v, helion_blocks = pack_paged_cache(
        keys, values, page_size=page_size, max_seqlen_k=spec.seqlen_k
    )
    candidates: dict[str, Candidate] = {
        "helion-attention": lambda: vllm_flash_attn.flash_attn_varlen_func(
            q=q,
            k=helion_k,
            v=helion_v,
            max_seqlen_q=spec.seqlen_q,
            cu_seqlens_q=cu_q,
            max_seqlen_k=spec.seqlen_k,
            seqused_k=seqused_k,
            block_table=helion_blocks,
            causal=spec.causal,
        )
    }
    flash_candidate = (
        _flash_attn_with_kvcache if spec.is_decode else _flash_attn_varlen_func
    )
    if flash_candidate is not None:
        flash_k, flash_v, flash_blocks = pack_paged_cache(
            keys,
            values,
            page_size=FLASH_ATTN_PAGE_SIZE,
            max_seqlen_k=spec.seqlen_k,
        )
        if spec.is_decode:
            flash_q = q.reshape(
                spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim
            )
            candidates["flash-attn"] = lambda: _flash_attn_with_kvcache(
                flash_q,
                flash_k,
                flash_v,
                cache_seqlens=seqused_k,
                block_table=flash_blocks,
                causal=spec.causal,
            ).reshape_as(q)
        else:
            candidates["flash-attn"] = lambda: _flash_attn_varlen_func(
                q,
                flash_k,
                flash_v,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=spec.causal,
                block_table=flash_blocks,
            )

    reference, operation_count = packed_reference(
        q, keys, values, lengths_q, spec
    )
    return candidates, reference, operation_count


def build_backward_candidates(
    spec: AttnShape,
) -> tuple[dict[str, Candidate], tuple[torch.Tensor, ...], float]:
    """Build retained forward graphs so timed calls contain backward work only."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def rand(seqlen: int, nheads: int) -> torch.Tensor:
        return torch.randn(
            (spec.batch, seqlen, nheads, spec.head_dim),
            device="cuda",
            dtype=spec.dtype,
            generator=generator,
        )

    q = rand(spec.seqlen_q, spec.nheads_q).requires_grad_()
    k = rand(spec.seqlen_k, spec.nheads_kv).requires_grad_()
    v = rand(spec.seqlen_k, spec.nheads_kv).requires_grad_()
    grad_out = rand(spec.seqlen_q, spec.nheads_q)
    scale = 1.0 / math.sqrt(spec.head_dim)

    def backward(output: torch.Tensor) -> Candidate:
        return lambda: torch.autograd.grad(
            output, (q, k, v), grad_out, retain_graph=True
        )

    helion_output = helion_attention.flash_attn_func(
        q, k, v, causal=spec.causal, shape=spec
    )
    candidates: dict[str, Candidate] = {
        "helion-attention": backward(helion_output)
    }
    if _flash_attn_func is not None:
        flash_output = _flash_attn_func(q, k, v, causal=spec.causal)
        candidates["flash-attn"] = backward(flash_output)

    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    mask = None
    if spec.causal:
        row = torch.arange(spec.seqlen_q, device=q.device)[:, None]
        col = torch.arange(spec.seqlen_k, device=q.device)[None, :]
        mask = col <= row + spec.seqlen_k - spec.seqlen_q
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        attn_mask=mask,
        scale=scale,
        enable_gqa=spec.nheads_q != spec.nheads_kv,
    ).transpose(1, 2)
    reference = tuple(
        gradient.detach()
        for gradient in torch.autograd.grad(
            reference_output, (q_ref, k_ref, v_ref), grad_out.float()
        )
    )
    # Recomputing attention plus dV, dP, dQ, and dK costs approximately 2.5x
    # the forward's two matrix-multiplication pairs.
    operation_count = 2.5 * flops(spec)
    return candidates, reference, operation_count


def max_abs_error(actual: BenchmarkOutput, reference: BenchmarkOutput) -> float:
    if isinstance(actual, torch.Tensor) and isinstance(reference, torch.Tensor):
        return (actual.float() - reference).abs().max().item()
    if isinstance(actual, tuple) and isinstance(reference, tuple):
        if len(actual) != len(reference):
            raise ValueError("candidate returned the wrong number of tensors")
        return max(
            max_abs_error(item, expected)
            for item, expected in zip(actual, reference)
        )
    raise TypeError("candidate and reference output structures do not match")


def measure(
    candidates: dict[str, Candidate],
    reference: BenchmarkOutput,
    rounds: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, list[float]] = {name: [] for name in candidates}
    errors: dict[str, float] = {}
    for name, run in candidates.items():
        try:
            errors[name] = max_abs_error(run(), reference)
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

    entries = benchmark_entries(args.only)
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
    for entry, kind in entries:
        spec = spec_from_manifest_entry(entry)
        print(f"benchmarking {kind} {spec.describe()}", file=sys.stderr)
        if kind == "varlen":
            candidates, reference, scale = build_varlen_candidates(spec)
        elif kind == "paged":
            candidates, reference, scale = build_paged_candidates(
                spec, int(entry["page_size"])
            )
        elif kind == "backward":
            candidates, reference, scale = build_backward_candidates(spec)
        else:
            candidates, reference = build_candidates(spec)
            scale = flops(spec)
        measured = measure(candidates, reference, args.rounds)
        row = {
            "key": benchmark_key(entry, kind),
            "description": (
                f"backward {entry['description']}"
                if kind == "backward"
                else entry["description"]
            ),
            "kind": kind,
            "varlen": kind == "varlen",
            "paged": kind == "paged",
            "backward": kind == "backward",
            "flash_attn_api": (
                "flash_attn_with_kvcache"
                if spec.is_decode and kind in ("forward", "paged")
                else "flash_attn_varlen_func"
                if kind in ("varlen", "paged")
                else "flash_attn_func backward"
                if kind == "backward"
                else "flash_attn_func"
            ),
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
    ]
    kinds = {row.get("kind") for row in report["results"]}  # type: ignore[index]
    notes = []
    if "paged" in kinds:
        notes.append(
            "Paged rows use identical logical caches with native page sizes "
            f"(Helion: 16; FlashAttention: {FLASH_ATTN_PAGE_SIZE}); paged decode "
            "uses flash_attn_with_kvcache and chunked prefill uses "
            "flash_attn_varlen_func."
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
            + " | helion vs flash-attn |",
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
        flash = impls.get("flash-attn")
        if helion is not None and flash is not None:
            speedup = flash["us"] / helion["us"]
            comparison = (
                f"{speedup:.2f}x faster"
                if speedup >= 1
                else f"{1 / speedup:.2f}x slower"
            )
        else:
            comparison = "n/a"
        lines.append(
            f"| {row['description']} | " + " | ".join(cells) + f" | {comparison} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
