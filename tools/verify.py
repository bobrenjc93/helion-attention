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
from helion_attention._registry import spec_from_manifest_entry as entry_to_spec  # noqa: E402
from helion_attention._shape import AttnShape  # noqa: E402

TOLERANCE = 5e-2
GRAD_TOLERANCE = 5e-2


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


def check_gradients(spec: AttnShape) -> list[float]:
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
    scale = 1.0 / math.sqrt(spec.head_dim)
    got = helion_attention.flash_attn_func(
        q, k, v, causal=spec.causal, shape=spec
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
    return [
        (actual.float() - reference).abs().max().item()
        for actual, reference in zip(got_grads, expected_grads)
    ]


def main(argv: list[str]) -> int:
    entries = {str(entry["key"]): entry for entry in available_shapes()}
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
        error = check(spec)
        status = "ok " if error <= TOLERANCE else "FAIL"
        failures += error > TOLERANCE
        print(f"[{status}] {key}: max abs error {error:.4g}")
        if entry.get("backward", False):
            grad_errors = check_gradients(spec)
            grad_ok = max(grad_errors) <= GRAD_TOLERANCE
            failures += not grad_ok
            status = "ok " if grad_ok else "FAIL"
            details = ", ".join(
                f"{name}={error:.4g}"
                for name, error in zip(("dq", "dk", "dv"), grad_errors)
            )
            print(f"[{status}] {key} backward: {details}")
    if "helion" in sys.modules:
        print("FAIL: importing helion_attention pulled in Helion")
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
