"""Burner evaluation: could vLLM swap helion-attention in for FlashAttention?

vLLM reaches FlashAttention through exactly one module, ``vllm.vllm_flash_attn``,
and almost entirely through one function, ``flash_attn_varlen_func``, driven with
a *paged* KV cache. This harness is the executable definition of that contract:
it checks the symbols and the signature, then exercises the call patterns
``vllm/v1/attention/backends/flash_attn.py`` actually issues and compares each
against an fp32 oracle.

The target module is ``helion_attention.vllm_flash_attn``. Every check is worth
points, the script always exits 0, and a missing module scores 0 rather than
crashing the evaluation.

    python tools/eval_vllm_compat.py [--verbose]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODULE = "helion_attention.vllm_flash_attn"

# Full keyword surface of vllm_flash_attn.flash_attn_varlen_func. vLLM passes
# everything by keyword, so accepting these names is what matters, not order.
# The second group is not exercised by the behavioural checks below, but vLLM
# does pass several of them unconditionally -- FA3 MLA prefill always sends
# output_scale=None, for instance -- so omitting any one of them is a TypeError
# in normal serving rather than a missing optional feature.
VLLM_KEYWORDS = [
    "q", "k", "v",
    "max_seqlen_q", "cu_seqlens_q", "max_seqlen_k", "cu_seqlens_k",
    "seqused_k", "softmax_scale", "causal", "window_size", "softcap",
    "alibi_slopes", "block_table", "return_softmax_lse", "out",
    "scheduler_metadata", "q_descale", "k_descale", "v_descale",
    "num_splits", "fa_version", "s_aux",
    "q_v", "dropout_p", "deterministic", "return_attn_probs", "output_scale",
    "cp_world_size", "cp_rank", "cp_tot_seqused_k",
    "mask_mod", "aux_tensors", "aux_tensor_leading_dims",
    "block_sparse_tensors", "dynamic_causal",
]

HELPER_SYMBOLS = [
    "get_scheduler_metadata",
    "is_fa_version_supported",
    "fa_version_unsupported_reason",
]

DTYPE = torch.bfloat16
ATOL = 5e-2
RTOL = 2e-2


@dataclass
class Check:
    name: str
    weight: int
    run: Callable[[object], str]
    """Returns a detail string on success; raises on failure."""


# --------------------------------------------------------------------------
# fp32 oracle
# --------------------------------------------------------------------------


def reference_one_request(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    window: tuple[int, int] | None,
    softcap: float,
) -> torch.Tensor:
    """q [Lq, Hq, D], k/v [Lk, Hkv, D] -> [Lq, Hq, D], all in fp32 internally."""
    seqlen_q, nheads_q, _ = q.shape
    seqlen_k, nheads_kv, _ = k.shape
    group = nheads_q // nheads_kv
    query = q.float().transpose(0, 1)
    key = k.float().transpose(0, 1).repeat_interleave(group, dim=0)
    value = v.float().transpose(0, 1).repeat_interleave(group, dim=0)

    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if softcap and softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)

    # FlashAttention aligns the causal mask to the bottom right, so a request
    # whose query is shorter than its cache sees the whole history.
    offset = seqlen_k - seqlen_q
    rows = torch.arange(seqlen_q, device=q.device)[:, None]
    cols = torch.arange(seqlen_k, device=q.device)[None, :]
    keep = torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=q.device)
    if causal:
        keep &= cols <= rows + offset
    if window is not None:
        left, right = window
        if left >= 0:
            keep &= cols >= rows + offset - left
        if right >= 0:
            keep &= cols <= rows + offset + right

    scores = scores.masked_fill(~keep[None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, value).transpose(0, 1)


def build_paged_cache(
    per_request_kv: list[tuple[torch.Tensor, torch.Tensor]],
    page_size: int,
    nheads_kv: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter contiguous per-request K/V into a paged cache plus a block table."""
    blocks_per_request = [
        (k.shape[0] + page_size - 1) // page_size for k, _ in per_request_kv
    ]
    total_blocks = sum(blocks_per_request) + 3  # a few unused blocks, as in vLLM
    k_cache = torch.zeros(
        (total_blocks, page_size, nheads_kv, head_dim), device=device, dtype=DTYPE
    )
    v_cache = torch.zeros_like(k_cache)
    max_blocks = max(blocks_per_request)
    block_table = torch.zeros(
        (len(per_request_kv), max_blocks), device=device, dtype=torch.int32
    )
    seqused_k = torch.tensor(
        [k.shape[0] for k, _ in per_request_kv], device=device, dtype=torch.int32
    )
    # Hand out blocks back to front so a correct kernel cannot pass by assuming
    # the block table is the identity mapping.
    free = list(reversed(range(total_blocks)))
    for request, (key, value) in enumerate(per_request_kv):
        for block in range(blocks_per_request[request]):
            physical = free.pop()
            block_table[request, block] = physical
            start = block * page_size
            stop = min(start + page_size, key.shape[0])
            k_cache[physical, : stop - start] = key[start:stop]
            v_cache[physical, : stop - start] = value[start:stop]
    return k_cache, v_cache, block_table, seqused_k


def make_case(
    query_lens: list[int],
    kv_lens: list[int],
    nheads_q: int = 8,
    nheads_kv: int = 2,
    head_dim: int = 128,
    page_size: int = 16,
    seed: int = 0,
) -> dict[str, object]:
    """One vLLM-shaped batch: packed queries plus a paged KV cache."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=DTYPE, generator=generator)

    queries = [rand(length, nheads_q, head_dim) for length in query_lens]
    per_request_kv = [
        (rand(length, nheads_kv, head_dim), rand(length, nheads_kv, head_dim))
        for length in kv_lens
    ]
    q = torch.cat(queries, dim=0)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    k_cache, v_cache, block_table, seqused_k = build_paged_cache(
        per_request_kv, page_size, nheads_kv, head_dim, device
    )
    return {
        "q": q,
        "queries": queries,
        "per_request_kv": per_request_kv,
        "cu_seqlens_q": cu_seqlens_q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "seqused_k": seqused_k,
        "max_seqlen_q": max(query_lens),
        "max_seqlen_k": max(kv_lens),
        "scale": 1.0 / math.sqrt(head_dim),
    }


def expected_for(
    case: dict[str, object],
    causal: bool = True,
    window: tuple[int, int] | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    outputs = [
        reference_one_request(
            query, key, value, float(case["scale"]), causal, window, softcap
        )
        for query, (key, value) in zip(case["queries"], case["per_request_kv"])
    ]
    return torch.cat(outputs, dim=0)


def compare(got: torch.Tensor, expected: torch.Tensor, label: str) -> str:
    if got is None:
        raise AssertionError(f"{label}: returned None")
    if tuple(got.shape) != tuple(expected.shape):
        raise AssertionError(f"{label}: shape {tuple(got.shape)} != {tuple(expected.shape)}")
    error = (got.float() - expected).abs().max().item()
    if not math.isfinite(error) or error > ATOL:
        raise AssertionError(f"{label}: max abs error {error:.4g} exceeds {ATOL}")
    return f"{label}: max abs error {error:.3g}"


def paged_call(module: object, case: dict[str, object], **overrides: object) -> object:
    """Invoke exactly the way vLLM's unified attention path does."""
    kwargs: dict[str, object] = {
        "q": case["q"],
        "k": case["k_cache"],
        "v": case["v_cache"],
        "cu_seqlens_q": case["cu_seqlens_q"],
        "max_seqlen_q": case["max_seqlen_q"],
        "seqused_k": case["seqused_k"],
        "max_seqlen_k": case["max_seqlen_k"],
        "softmax_scale": case["scale"],
        "causal": True,
        "alibi_slopes": None,
        "window_size": None,
        "block_table": case["block_table"],
        "softcap": 0.0,
        "scheduler_metadata": None,
        "fa_version": 3,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "num_splits": 0,
        "s_aux": None,
    }
    kwargs.update(overrides)
    return module.flash_attn_varlen_func(**kwargs)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_signature(module: object) -> str:
    function = getattr(module, "flash_attn_varlen_func", None)
    if function is None:
        raise AssertionError("flash_attn_varlen_func is missing")
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    missing = [name for name in VLLM_KEYWORDS if name not in parameters]
    if missing and not accepts_kwargs:
        raise AssertionError("missing keyword arguments vLLM passes: " + ", ".join(missing))
    required = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name not in ("q", "k", "v", "max_seqlen_q", "cu_seqlens_q", "max_seqlen_k")
    ]
    if required:
        raise AssertionError(
            "vLLM never passes these, so they cannot be required: " + ", ".join(required)
        )
    return f"accepts all {len(VLLM_KEYWORDS)} vLLM keywords with no extra required arguments"


def check_helpers(module: object) -> str:
    missing = [name for name in HELPER_SYMBOLS if not hasattr(module, name)]
    if missing:
        raise AssertionError("missing helpers imported by vllm: " + ", ".join(missing))
    if not module.is_fa_version_supported(3):  # type: ignore[attr-defined]
        raise AssertionError("is_fa_version_supported(3) must be True for vLLM to select us")
    module.get_scheduler_metadata(  # type: ignore[attr-defined]
        batch_size=2,
        max_seqlen_q=1,
        max_seqlen_k=128,
        num_heads_q=8,
        num_heads_kv=2,
        headdim=128,
        cache_seqlens=torch.tensor([128, 64], device="cuda", dtype=torch.int32),
        page_size=16,
        causal=True,
    )
    return "get_scheduler_metadata / is_fa_version_supported / unsupported_reason present"


def check_paged_decode(module: object) -> str:
    case = make_case([1, 1, 1, 1], [37, 128, 1024, 5], seed=1)
    got = paged_call(module, case)
    return compare(got, expected_for(case), "paged decode (4 requests, ragged cache)")


def check_paged_chunked_prefill(module: object) -> str:
    # A chunk of new tokens appended to an existing cache: seqlen_q < seqlen_k,
    # which is where FlashAttention's bottom-right causal alignment matters.
    case = make_case([64, 200], [320, 200], seed=2)
    got = paged_call(module, case)
    return compare(got, expected_for(case), "paged chunked prefill (bottom-right causal)")


def check_mixed_batch(module: object) -> str:
    # vLLM packs decode rows and prefill rows into one call.
    case = make_case([1, 1, 96, 1, 33], [512, 17, 96, 1024, 200], seed=3)
    got = paged_call(module, case)
    return compare(got, expected_for(case), "mixed decode+prefill batch")


def check_out_parameter(module: object) -> str:
    case = make_case([1, 48], [256, 48], seed=4)
    out = torch.empty_like(case["q"])
    returned = paged_call(module, case, out=out)
    detail = compare(out, expected_for(case), "out= written in place")
    if returned is not None and not isinstance(returned, tuple):
        if returned.data_ptr() != out.data_ptr():
            raise AssertionError("out= was provided but a different tensor was returned")
    return detail


def check_return_softmax_lse(module: object) -> str:
    case = make_case([1, 64], [300, 64], seed=5)
    result = paged_call(module, case, return_softmax_lse=True)
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError("return_softmax_lse=True must return (out, softmax_lse)")
    out, lse = result
    detail = compare(out, expected_for(case), "return_softmax_lse output")
    total_q = int(case["q"].shape[0])
    nheads_q = int(case["q"].shape[1])
    if tuple(lse.shape) not in {(nheads_q, total_q), (total_q, nheads_q)}:
        raise AssertionError(
            f"softmax_lse shape {tuple(lse.shape)} is neither "
            f"{(nheads_q, total_q)} nor {(total_q, nheads_q)}"
        )
    if not torch.isfinite(lse.float()).all():
        raise AssertionError("softmax_lse contains non-finite values")
    return detail + f"; lse {tuple(lse.shape)}"


def check_sliding_window(module: object) -> str:
    case = make_case([1, 80], [600, 80], seed=6)
    window = [128, 0]
    got = paged_call(module, case, window_size=window)
    return compare(
        got, expected_for(case, window=(128, 0)), "sliding window window_size=[128, 0]"
    )


def check_softcap(module: object) -> str:
    case = make_case([1, 40], [256, 40], seed=7)
    got = paged_call(module, case, softcap=30.0)
    return compare(got, expected_for(case, softcap=30.0), "softcap=30.0")


def check_nonpaged_varlen(module: object) -> str:
    # The encoder path passes packed K/V with cu_seqlens_k and no block table.
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(8)
    lengths = [24, 100]
    nheads_q, nheads_kv, head_dim = 8, 2, 128

    def rand(length: int, heads: int) -> torch.Tensor:
        return torch.randn(
            (length, heads, head_dim), device=device, dtype=DTYPE, generator=generator
        )

    queries = [rand(length, nheads_q) for length in lengths]
    keys = [rand(length, nheads_kv) for length in lengths]
    values = [rand(length, nheads_kv) for length in lengths]
    cumulative = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    scale = 1.0 / math.sqrt(head_dim)
    got = module.flash_attn_varlen_func(  # type: ignore[attr-defined]
        q=torch.cat(queries),
        k=torch.cat(keys),
        v=torch.cat(values),
        cu_seqlens_q=cumulative,
        cu_seqlens_k=cumulative,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        softmax_scale=scale,
        causal=False,
        window_size=None,
        softcap=0.0,
        alibi_slopes=None,
        fa_version=3,
    )
    expected = torch.cat(
        [
            reference_one_request(q, k, v, scale, False, None, 0.0)
            for q, k, v in zip(queries, keys, values)
        ]
    )
    return compare(got, expected, "non-paged varlen encoder attention")


CHECKS = [
    Check("signature", 15, check_signature),
    Check("helpers", 5, check_helpers),
    Check("paged decode", 20, check_paged_decode),
    Check("paged chunked prefill", 20, check_paged_chunked_prefill),
    Check("mixed decode+prefill batch", 10, check_mixed_batch),
    Check("out= in place", 5, check_out_parameter),
    Check("return_softmax_lse", 5, check_return_softmax_lse),
    Check("non-paged varlen", 10, check_nonpaged_varlen),
    Check("sliding window", 5, check_sliding_window),
    Check("softcap", 5, check_softcap),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        module = importlib.import_module(MODULE)
    except Exception as error:  # noqa: BLE001
        json.dump(
            {
                "score": 0,
                "summary": f"{MODULE} does not import, so vLLM cannot swap us in",
                "evidence": [f"{type(error).__name__}: {error}"],
                "suggestions": [
                    f"Create {MODULE} exposing flash_attn_varlen_func with vLLM's keyword "
                    "surface (paged k/v plus block_table and seqused_k), get_scheduler_metadata, "
                    "is_fa_version_supported and fa_version_unsupported_reason. "
                    "Run tools/eval_vllm_compat.py --verbose to see every failing case."
                ],
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    earned = 0
    passed: list[str] = []
    failed: list[str] = []
    for check in CHECKS:
        try:
            detail = check.run(module)
            earned += check.weight
            passed.append(f"[+{check.weight}] {check.name}: {detail}")
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {error}".replace("\n", " ")
            failed.append(f"[-{check.weight}] {check.name}: {message[:300]}")
            if args.verbose:
                traceback.print_exc(file=sys.stderr)

    total = sum(check.weight for check in CHECKS)
    score = round(100 * earned / total)
    json.dump(
        {
            "score": score,
            "summary": (
                f"{len(passed)}/{len(CHECKS)} vLLM compatibility checks pass "
                f"({earned}/{total} weighted)"
            ),
            "evidence": (failed + passed)[:8],
            "suggestions": (
                [f"failing vLLM call patterns: {item}" for item in failed[:5]]
                or ["vLLM's full FlashAttention call surface is satisfied"]
            ),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
