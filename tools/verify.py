"""Verify one checked-in kernel in a process that never imports Helion.

Usage: python tools/verify.py [kernel_key ...]      (no args = every kernel)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import helion_attention  # noqa: E402
from helion_attention._registry import available_shapes  # noqa: E402
from helion_attention._registry import available_varlen_shapes  # noqa: E402
from helion_attention._registry import spec_from_manifest_entry as entry_to_spec  # noqa: E402
from helion_attention._shape import AttnShape  # noqa: E402

TOLERANCE = 5e-2
GRAD_TOLERANCE = 5e-2
GRAD_RTOL = 2e-2


def check(spec: AttnShape) -> float:
    generator = torch.Generator(device="cuda").manual_seed(1234)

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
    scale = 1.0 / math.sqrt(spec.head_dim)
    if spec.is_decode:
        got = helion_attention.flash_attn_with_kvcache(
            q, k, v, causal=spec.causal, shape=spec
        )
    else:
        got = helion_attention.flash_attn_func(q, k, v, causal=spec.causal, shape=spec)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        scale=scale,
        is_causal=spec.causal and not spec.is_decode,
        enable_gqa=spec.nheads_q != spec.nheads_kv,
    ).transpose(1, 2)
    return (got.float() - expected).abs().max().item()


def check_varlen(spec: AttnShape) -> float:
    """Exercise a generated packed kernel with a non-generation length profile."""
    generator = torch.Generator(device="cuda").manual_seed(5678)
    if spec.batch == 1:
        lengths_q = [spec.seqlen_q]
        lengths_k = [spec.seqlen_k]
    else:
        lengths_q = [spec.seqlen_q] + [
            max(1, spec.seqlen_q // (index + 2))
            for index in range(1, spec.batch)
        ]
        lengths_k = [max(1, spec.seqlen_k // 4), spec.seqlen_k] + [
            max(1, spec.seqlen_k * (index + 1) // (2 * spec.batch))
            for index in range(2, spec.batch)
        ]

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
    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        causal=spec.causal,
        shape=spec,
    )

    expected = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(spec.head_dim)
    q_offsets = cu_q.tolist()
    k_offsets = cu_k.tolist()
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]
    ):
        q_seq = q[q_start:q_end].float().transpose(0, 1).unsqueeze(0)
        k_seq = k[k_start:k_end].float().transpose(0, 1).unsqueeze(0)
        v_seq = v[k_start:k_end].float().transpose(0, 1).unsqueeze(0)
        mask = None
        if spec.causal:
            seqlen_q = q_end - q_start
            seqlen_k = k_end - k_start
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            mask = col <= row + seqlen_k - seqlen_q
        result = torch.nn.functional.scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            attn_mask=mask,
            scale=scale,
            enable_gqa=spec.nheads_q != spec.nheads_kv,
        )
        expected[q_start:q_end] = result.squeeze(0).transpose(0, 1)
    return (got.float() - expected).abs().max().item()


def check_gradients(
    spec: AttnShape, softmax_scale: float | None = None
) -> tuple[list[float], list[int]]:
    """Compare the public autograd path with fp32 SDPA."""
    generator = torch.Generator(device="cuda").manual_seed(4321)

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
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    got = helion_attention.flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        shape=spec,
    )
    got_grads = torch.autograd.grad(got, (q, k, v), grad_out)

    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        scale=scale,
        is_causal=spec.causal and not spec.is_decode,
        enable_gqa=spec.nheads_q != spec.nheads_kv,
    ).transpose(1, 2)
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )
    errors = [
        (actual.float() - reference).abs().max().item()
        for actual, reference in zip(got_grads, expected_grads)
    ]
    mismatches = [
        (~torch.isclose(actual.float(), reference, atol=GRAD_TOLERANCE, rtol=GRAD_RTOL))
        .sum()
        .item()
        for actual, reference in zip(got_grads, expected_grads)
    ]
    return errors, mismatches


def main(argv: list[str]) -> int:
    all_entries = available_shapes() + available_varlen_shapes()
    entries = {str(entry["key"]): entry for entry in all_entries}
    keys = argv or sorted(entries)
    if not keys:
        print("no kernels checked in yet")
        return 0
    failures = 0
    for key in keys:
        # A kernel key may be freshly generated and not yet in the manifest.
        if key in entries:
            entry = entries[key]
        else:
            module = __import__(f"helion_attention.kernels.{key}", fromlist=["KERNEL_SPEC"])
            entry = module.KERNEL_SPEC
        spec = entry_to_spec(entry)
        is_varlen = bool(entry.get("varlen", False))
        error = check_varlen(spec) if is_varlen else check(spec)
        status = "ok " if error <= TOLERANCE else "FAIL"
        failures += error > TOLERANCE
        print(f"[{status}] {key}: max abs error {error:.4g}")
        if entry.get("backward", False) and not is_varlen:
            for scale_name, softmax_scale in (("default", None), ("1.0", 1.0)):
                grad_errors, mismatches = check_gradients(spec, softmax_scale)
                grad_ok = not any(mismatches)
                failures += not grad_ok
                status = "ok " if grad_ok else "FAIL"
                details = ", ".join(
                    f"{name}={error:.4g} ({mismatch} mismatches)"
                    for name, error, mismatch in zip(
                        ("dq", "dk", "dv"), grad_errors, mismatches
                    )
                )
                print(
                    f"[{status}] {key} backward scale={scale_name}: {details}"
                )
    if "helion" in sys.modules:
        print("FAIL: importing helion_attention pulled in Helion")
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
