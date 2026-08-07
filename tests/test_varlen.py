"""Packed variable-length API and generated-kernel regressions."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import helion_attention
from helion_attention import AttnShape
from helion_attention._registry import spec_from_manifest_entry

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
VARLEN_SHAPES = helion_attention.available_varlen_shapes()
VARLEN_IDS = [str(entry["key"]) for entry in VARLEN_SHAPES]


def _lengths(maximum: int, batch: int, *, key: bool, variant: int) -> list[int]:
    if batch == 1:
        return [maximum]
    if variant == 2:
        values = [maximum] * batch
    elif variant == 0:
        values = [max(1, maximum // (index + 2)) for index in range(batch)]
        values[1 if key else 0] = maximum
    else:
        values = [
            max(1, maximum * (index + 1) // (2 * batch))
            for index in range(batch)
        ]
        values[-1 if key else batch // 2] = maximum
    return values


def _cumulative(lengths: list[int], device: torch.device | str) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device=device, dtype=torch.int32)


def make_packed(
    spec: AttnShape, *, variant: int = 0, seed: int = 123
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
]:
    lengths_q = _lengths(spec.seqlen_q, spec.batch, key=False, variant=variant)
    lengths_k = _lengths(spec.seqlen_k, spec.batch, key=True, variant=variant)
    generator = torch.Generator(device="cuda").manual_seed(seed)
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
    return (
        q,
        k,
        v,
        _cumulative(lengths_q, q.device),
        _cumulative(lengths_k, q.device),
        lengths_q,
        lengths_k,
    )


def reference_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths_q: list[int],
    lengths_k: list[int],
    *,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    q_start = 0
    k_start = 0
    for seqlen_q, seqlen_k in zip(lengths_q, lengths_k):
        q_seq = q[q_start : q_start + seqlen_q].float().transpose(0, 1)[None]
        k_seq = k[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        v_seq = v[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        mask = None
        if causal:
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            mask = col <= row + seqlen_k - seqlen_q
        result = torch.nn.functional.scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            attn_mask=mask,
            scale=scale,
            enable_gqa=q.size(1) != k.size(1),
        )
        out[q_start : q_start + seqlen_q] = result[0].transpose(0, 1)
        q_start += seqlen_q
        k_start += seqlen_k
    return out


def test_varlen_manifest_covers_causal_and_noncausal() -> None:
    assert VARLEN_SHAPES
    assert {bool(entry["causal"]) for entry in VARLEN_SHAPES} == {False, True}


def test_varlen_signature_matches_flash_attention_plus_shape() -> None:
    names = list(inspect.signature(helion_attention.flash_attn_varlen_func).parameters)
    assert names == [
        "q",
        "k",
        "v",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "dropout_p",
        "softmax_scale",
        "causal",
        "window_size",
        "softcap",
        "alibi_slopes",
        "deterministic",
        "return_attn_probs",
        "block_table",
        "shape",
    ]
    assert inspect.signature(helion_attention.flash_attn_varlen_func).parameters[
        "shape"
    ].kind is inspect.Parameter.KEYWORD_ONLY


@requires_cuda
@pytest.mark.parametrize("entry", VARLEN_SHAPES, ids=VARLEN_IDS)
@pytest.mark.parametrize(
    "variant", [0, 1, 2], ids=["profile-a", "profile-b", "all-full"]
)
def test_varlen_matches_fp32_sdpa_with_dynamic_token_totals(
    entry: dict[str, object], variant: int
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=variant
    )
    scale = 1.0 / math.sqrt(spec.head_dim)
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
    expected = reference_packed(
        q, k, v, lengths_q, lengths_k, causal=spec.causal, scale=scale
    )
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
def test_varlen_custom_scale_and_fully_masked_causal_rows() -> None:
    entry = next(item for item in VARLEN_SHAPES if item["causal"])
    spec = spec_from_manifest_entry(entry)
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(spec)
    scale = 0.37
    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=scale,
        causal=True,
        shape=spec,
    )
    expected = reference_packed(
        q, k, v, lengths_q, lengths_k, causal=True, scale=scale
    )
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    fully_masked = lengths_q[0] - lengths_k[0]
    assert fully_masked > 0
    torch.testing.assert_close(got[:fully_masked], torch.zeros_like(got[:fully_masked]))


@requires_cuda
def test_varlen_supports_cuda_graph_capture() -> None:
    entry = next(item for item in VARLEN_SHAPES if not item["causal"])
    spec = spec_from_manifest_entry(entry)
    q, k, v, cu_q, cu_k, *_ = make_packed(spec)

    def run() -> torch.Tensor:
        return helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            shape=spec,
        )

    expected = run()
    torch.cuda.synchronize(q.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(q.device)
    torch.testing.assert_close(captured, expected)


def test_varlen_shape_argument_is_required() -> None:
    tensor = torch.zeros(1, 1, 1)
    cu = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(TypeError):
        helion_attention.flash_attn_varlen_func(  # type: ignore[call-arg]
            tensor, tensor, tensor, cu, cu, 1, 1
        )


@requires_cuda
def test_varlen_validates_maxima_cu_seqlens_and_forward_only_contract() -> None:
    entry = next(item for item in VARLEN_SHAPES if not item["causal"])
    spec = spec_from_manifest_entry(entry)
    q, k, v, cu_q, cu_k, *_ = make_packed(spec)
    call_args = (q, k, v, cu_q, cu_k, spec.seqlen_q, spec.seqlen_k)

    with pytest.raises(ValueError, match="must match"):
        helion_attention.flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, spec.seqlen_q - 1, spec.seqlen_k, shape=spec
        )
    with pytest.raises(ValueError, match="torch.int32"):
        helion_attention.flash_attn_varlen_func(
            q, k, v, cu_q.to(torch.int64), cu_k, spec.seqlen_q, spec.seqlen_k, shape=spec
        )
    with pytest.raises(ValueError, match="batch"):
        helion_attention.flash_attn_varlen_func(
            q, k, v, cu_q[:-1], cu_k, spec.seqlen_q, spec.seqlen_k, shape=spec
        )
    with pytest.raises(NotImplementedError, match="block tables"):
        helion_attention.flash_attn_varlen_func(
            *call_args, block_table=torch.ones(1, device=q.device), shape=spec
        )

    q.requires_grad_()
    with pytest.raises(NotImplementedError, match="forward-only"):
        helion_attention.flash_attn_varlen_func(*call_args, shape=spec)


def test_support_queries_are_metadata_only_and_varlen_is_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion_attention import _registry

    def fail_load(key: str) -> None:
        raise AssertionError(f"support query tried to load {key}")

    monkeypatch.setattr(_registry, "_load", fail_load)
    monkeypatch.setattr(_registry, "_load_varlen", fail_load)
    monkeypatch.setattr(_registry, "_load_paged", fail_load)
    shape = (8, 512, 16, 64)
    assert helion_attention.is_shape_supported(shape)
    assert helion_attention.is_varlen_shape_supported(shape)
    assert helion_attention.is_paged_shape_supported(
        (4, 1, 1024, 8, 2, 128), causal=True, page_size=16
    )
    assert not helion_attention.is_varlen_shape_supported((3, 77, 5, 64))
