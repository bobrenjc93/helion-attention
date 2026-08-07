"""API-level tests. These import only the runtime package, never Helion."""

from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import helion_attention
from helion_attention import AttnShape
from helion_attention import UnsupportedShapeError
from helion_attention import available_shapes
from helion_attention._registry import spec_from_manifest_entry

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
requires_two_cuda_devices = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs"
)

SHAPES = [entry for entry in available_shapes()]
IDS = [str(entry["key"]) for entry in SHAPES]
DECODE_SHAPES = [
    entry
    for entry in SHAPES
    if int(entry["seqlen_q"]) == 1 and int(entry["seqlen_k"]) > 1
]
GRAPH_DECODE_SHAPES = [
    entry for entry in DECODE_SHAPES if int(entry["seqlen_k"]) in (1024, 16384)
]
LONG_DECODE = next(
    entry for entry in DECODE_SHAPES if int(entry["seqlen_k"]) == 16384
)
BACKWARD_SHAPES = [entry for entry in SHAPES if bool(entry.get("backward", False))]
QWEN_PREFILL = AttnShape(
    batch=1,
    seqlen_q=2048,
    seqlen_k=2048,
    nheads_q=28,
    nheads_kv=4,
    head_dim=128,
    dtype=torch.bfloat16,
    causal=True,
)
CHUNKED_PREFILL_KEY = "b1_sq64_sk320_hq8_hkv2_d128_bf16_causal"


def make_inputs(
    spec: AttnShape, device: torch.device | str = "cuda", *, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(seqlen: int, nheads: int) -> torch.Tensor:
        return torch.randn(
            (spec.batch, seqlen, nheads, spec.head_dim),
            device=device,
            dtype=spec.dtype,
            generator=generator,
        )

    return (
        rand(spec.seqlen_q, spec.nheads_q),
        rand(spec.seqlen_k, spec.nheads_kv),
        rand(spec.seqlen_k, spec.nheads_kv),
    )


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttnShape,
    scale: float,
) -> torch.Tensor:
    mask = None
    if spec.causal:
        row = torch.arange(spec.seqlen_q, device=q.device)[:, None]
        col = torch.arange(spec.seqlen_k, device=q.device)[None, :]
        mask = col <= row + spec.seqlen_k - spec.seqlen_q
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        attn_mask=mask,
        scale=scale,
        enable_gqa=spec.nheads_q != spec.nheads_kv,
    ).transpose(1, 2)


def test_package_does_not_import_helion() -> None:
    assert "helion" not in sys.modules


def test_manifest_is_not_empty() -> None:
    assert SHAPES, "no kernels are checked in"


def test_backward_catalogue_is_scoped_to_one_noncausal_shape() -> None:
    assert len(BACKWARD_SHAPES) == 1
    assert BACKWARD_SHAPES[0]["causal"] is False


def test_decode_cache_lengths_are_checked_in() -> None:
    lengths = {
        int(entry["seqlen_k"])
        for entry in DECODE_SHAPES
        if int(entry["nheads_q"]) > int(entry["nheads_kv"])
    }
    assert {1024, 4096, 16384} <= lengths


@requires_cuda
@pytest.mark.parametrize("entry", SHAPES, ids=IDS)
def test_matches_fp32_sdpa(entry: dict[str, object]) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec)
    got = helion_attention.flash_attn_func(q, k, v, causal=spec.causal, shape=spec)
    expected = reference_attention(q, k, v, spec, 1.0 / math.sqrt(spec.head_dim))
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize(
    "entry", BACKWARD_SHAPES, ids=[str(entry["key"]) for entry in BACKWARD_SHAPES]
)
@pytest.mark.parametrize(
    "softmax_scale",
    [None, 0.137, 1.0],
    ids=["default-scale", "near-default-scale", "large-scale"],
)
def test_gradients_match_fp32_sdpa(
    entry: dict[str, object], softmax_scale: float | None
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec, seed=123)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    grad_out = make_inputs(spec, seed=456)[0]

    got = helion_attention.flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        shape=spec,
    )
    got_grads = torch.autograd.grad(got, (q, k, v), grad_out)

    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    scale = 1.0 / math.sqrt(spec.head_dim) if softmax_scale is None else softmax_scale
    expected = reference_attention(
        q_ref, k_ref, v_ref, spec, scale
    )
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )

    for actual, reference in zip(got_grads, expected_grads):
        torch.testing.assert_close(
            actual.float(), reference, atol=5e-2, rtol=2e-2
        )


@requires_cuda
def test_requires_grad_rejects_shape_without_backward() -> None:
    entry = next(item for item in SHAPES if not item.get("backward", False))
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec)
    q.requires_grad_()
    with pytest.raises(NotImplementedError, match="backward kernel"):
        helion_attention.flash_attn_func(
            q, k, v, causal=spec.causal, shape=spec
        )


@requires_cuda
@pytest.mark.parametrize("entry", SHAPES[:1], ids=IDS[:1])
def test_custom_softmax_scale(entry: dict[str, object]) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec)
    scale = 0.137
    got = helion_attention.flash_attn_func(
        q, k, v, softmax_scale=scale, causal=spec.causal, shape=spec
    )
    expected = reference_attention(q, k, v, spec, scale)
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
def test_unequal_causal_mask_includes_bottom_right_boundary() -> None:
    entry = next(item for item in SHAPES if item["key"] == CHUNKED_PREFILL_KEY)
    spec = spec_from_manifest_entry(entry)
    q = torch.zeros(
        (spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim),
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros(
        (spec.batch, spec.seqlen_k, spec.nheads_kv, spec.head_dim),
        device="cuda",
        dtype=spec.dtype,
    )
    v = torch.zeros_like(k)
    # Row zero's last visible key is 320 - 64 = 256. Concentrating the value
    # there catches both top-left alignment and an exclusive-boundary error.
    v[:, 256] = 100.0

    got = helion_attention.flash_attn_func(q, k, v, causal=True, shape=spec)
    expected = reference_attention(q, k, v, spec, 1.0 / math.sqrt(spec.head_dim))

    assert got[0, 0, 0, 0].item() > 0.25
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in DECODE_SHAPES],
)
def test_decode_custom_softmax_scale(entry: dict[str, object]) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec, seed=11039)
    scale = 2.0
    got = helion_attention.flash_attn_with_kvcache(
        q, k, v, softmax_scale=scale, causal=spec.causal, shape=spec
    )
    expected = reference_attention(q, k, v, spec, scale)
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
def test_preloaded_kernel_launches_in_worker_thread() -> None:
    assert helion_attention.is_shape_supported(
        QWEN_PREFILL, dtype=QWEN_PREFILL.dtype, causal=QWEN_PREFILL.causal
    )
    q, k, v = make_inputs(QWEN_PREFILL)

    def invoke() -> torch.Tensor:
        out = helion_attention.flash_attn_func(
            q, k, v, causal=QWEN_PREFILL.causal, shape=QWEN_PREFILL
        )
        torch.cuda.synchronize(q.device)
        return out

    with ThreadPoolExecutor(max_workers=1) as pool:
        out = pool.submit(invoke).result()
    assert out.device == q.device


@requires_two_cuda_devices
def test_launches_on_tensor_device_when_it_is_not_current() -> None:
    original_device = torch.cuda.current_device()
    target_device = torch.device("cuda", 1)
    try:
        torch.cuda.set_device(0)
        q, k, v = make_inputs(QWEN_PREFILL, device=target_device)
        assert torch.cuda.current_device() == 0

        out = helion_attention.flash_attn_func(
            q, k, v, causal=QWEN_PREFILL.causal, shape=QWEN_PREFILL
        )
        torch.cuda.synchronize(target_device)

        assert out.device == target_device
        assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(original_device)


@requires_two_cuda_devices
def test_split_decode_launches_on_tensor_device_when_it_is_not_current() -> None:
    original_device = torch.cuda.current_device()
    target_device = torch.device("cuda", 1)
    try:
        torch.cuda.set_device(0)
        spec = spec_from_manifest_entry(LONG_DECODE)
        q, k_cache, v_cache = make_inputs(spec, device=target_device)

        out = helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            causal=spec.causal,
            shape=spec,
        )
        torch.cuda.synchronize(target_device)

        assert out.device == target_device
        assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(original_device)


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


def test_kvcache_shape_argument_is_required() -> None:
    q = torch.zeros(1, 1, 1, 1)
    with pytest.raises(TypeError):
        helion_attention.flash_attn_with_kvcache(q, q, q)  # type: ignore[call-arg]


def test_shape_normalization_rejects_bad_tuples() -> None:
    from helion_attention._shape import normalize_shape

    with pytest.raises(ValueError):
        normalize_shape((1, 2, 3), torch.bfloat16, False)
    with pytest.raises(ValueError):
        normalize_shape((1, 2, 3, 0), torch.bfloat16, False)
    with pytest.raises(ValueError):
        normalize_shape((1, 8, 8, 6, 4, 64), torch.bfloat16, False)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in DECODE_SHAPES],
)
def test_kvcache_entry_point_matches_plain_attention(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec)
    cached = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=spec.seqlen_k,
        causal=spec.causal,
        shape=spec,
    )
    plain = helion_attention.flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=spec.causal,
        shape=spec,
    )
    torch.testing.assert_close(cached, plain)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_entry_point_rejects_cache_mutation_and_partial_cache(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec)
    with pytest.raises(NotImplementedError, match="updating"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k_cache[:, :1],
            v=v_cache[:, :1],
            causal=spec.causal,
            shape=spec,
        )
    with pytest.raises(NotImplementedError, match="partial or ragged"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=spec.seqlen_k - 1,
            causal=spec.causal,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_default_noncausal_flag_reuses_equivalent_decode_kernel(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec)
    shape = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
    )
    got = helion_attention.flash_attn_with_kvcache(q, k_cache, v_cache, shape=shape)
    expected = reference_attention(
        q, k_cache, v_cache, spec, 1.0 / math.sqrt(spec.head_dim)
    )
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    GRAPH_DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in GRAPH_DECODE_SHAPES],
)
def test_kvcache_full_length_supports_cuda_graph_capture(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec)

    # Warm up Triton's JIT and the allocator before capture.
    expected = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=spec.seqlen_k,
        causal=spec.causal,
        shape=spec,
    )
    torch.cuda.synchronize(q.device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=spec.seqlen_k,
            causal=spec.causal,
            shape=spec,
        )
    graph.replay()
    torch.cuda.synchronize(q.device)

    torch.testing.assert_close(captured, expected)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_tensor_lengths_are_rejected_without_poisoning_cuda(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec)
    invalid_lengths = torch.full(
        (spec.batch,),
        spec.seqlen_k - 1,
        dtype=torch.int32,
        device=q.device,
    )

    with pytest.raises(NotImplementedError, match="tensor-valued cache_seqlens"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=invalid_lengths,
            causal=spec.causal,
            shape=spec,
        )

    # The rejection is a host-side exception, so later CUDA work remains valid.
    torch.testing.assert_close(q + 1, q.add(1))
    torch.cuda.synchronize(q.device)


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
