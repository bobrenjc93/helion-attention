"""API-level tests. These import only the runtime package, never Helion."""

from __future__ import annotations

import math
import sys

import pytest
import torch

import helion_attention
from helion_attention import AttnShape
from helion_attention import UnsupportedShapeError
from helion_attention import available_shapes
from helion_attention._registry import spec_from_manifest_entry

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SHAPES = [entry for entry in available_shapes()]
IDS = [str(entry["key"]) for entry in SHAPES]


def make_inputs(spec: AttnShape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(7)

    def rand(seqlen: int, nheads: int) -> torch.Tensor:
        return torch.randn(
            (spec.batch, seqlen, nheads, spec.head_dim),
            device="cuda",
            dtype=spec.dtype,
            generator=generator,
        )

    return (
        rand(spec.seqlen_q, spec.nheads_q),
        rand(spec.seqlen_k, spec.nheads_kv),
        rand(spec.seqlen_k, spec.nheads_kv),
    )


def test_package_does_not_import_helion() -> None:
    assert "helion" not in sys.modules


def test_manifest_is_not_empty() -> None:
    assert SHAPES, "no kernels are checked in"


@requires_cuda
@pytest.mark.parametrize("entry", SHAPES, ids=IDS)
def test_matches_fp32_sdpa(entry: dict[str, object]) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec)
    got = helion_attention.flash_attn_func(q, k, v, causal=spec.causal, shape=spec)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        is_causal=spec.causal,
        scale=1.0 / math.sqrt(spec.head_dim),
        enable_gqa=spec.nheads_q != spec.nheads_kv,
    ).transpose(1, 2)
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("entry", SHAPES[:1], ids=IDS[:1])
def test_custom_softmax_scale(entry: dict[str, object]) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec)
    scale = 0.137
    got = helion_attention.flash_attn_func(
        q, k, v, softmax_scale=scale, causal=spec.causal, shape=spec
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        is_causal=spec.causal,
        scale=scale,
        enable_gqa=spec.nheads_q != spec.nheads_kv,
    ).transpose(1, 2)
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
def test_unsupported_shape_names_the_shape_and_the_issue_tracker() -> None:
    q = torch.randn(3, 77, 5, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(UnsupportedShapeError) as caught:
        helion_attention.flash_attn_func(q, q, q, shape=(3, 77, 5, 64))
    message = str(caught.value)
    assert "seqlen_q=77" in message
    assert "issues" in message


@requires_cuda
def test_shape_must_match_tensors() -> None:
    q = torch.randn(2, 1024, 32, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="declared shape"):
        helion_attention.flash_attn_func(q, q, q, shape=(2, 512, 32, 64))


@requires_cuda
def test_rejects_unimplemented_features() -> None:
    q = torch.randn(2, 1024, 32, 64, device="cuda", dtype=torch.bfloat16)
    for kwargs in (
        {"dropout_p": 0.1},
        {"window_size": (128, 0)},
        {"softcap": 30.0},
        {"return_attn_probs": True},
    ):
        with pytest.raises(NotImplementedError):
            helion_attention.flash_attn_func(q, q, q, shape=(2, 1024, 32, 64), **kwargs)


def test_shape_argument_is_required() -> None:
    q = torch.zeros(1, 1, 1, 1)
    with pytest.raises(TypeError):
        helion_attention.flash_attn_func(q, q, q)  # type: ignore[call-arg]


def test_shape_normalization_rejects_bad_tuples() -> None:
    from helion_attention._shape import normalize_shape

    with pytest.raises(ValueError):
        normalize_shape((1, 2, 3), torch.bfloat16, False)
    with pytest.raises(ValueError):
        normalize_shape((1, 2, 3, 0), torch.bfloat16, False)
    with pytest.raises(ValueError):
        normalize_shape((1, 8, 8, 6, 4, 64), torch.bfloat16, False)


@requires_cuda
@pytest.mark.parametrize("entry", SHAPES[:1], ids=IDS[:1])
def test_qkvpacked_matches_unpacked(entry: dict[str, object]) -> None:
    spec = spec_from_manifest_entry(entry)
    if spec.nheads_q != spec.nheads_kv or spec.seqlen_q != spec.seqlen_k:
        pytest.skip("packed layout requires identical q/k/v shapes")
    q, k, v = make_inputs(spec)
    qkv = torch.stack([q, k, v], dim=2)
    packed = helion_attention.flash_attn_qkvpacked_func(
        qkv, causal=spec.causal, shape=spec
    )
    plain = helion_attention.flash_attn_func(q, k, v, causal=spec.causal, shape=spec)
    torch.testing.assert_close(packed, plain)
