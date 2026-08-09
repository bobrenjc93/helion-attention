"""API-level tests. These import only the runtime package, never Helion."""

from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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
ENCODER_TRAINING = spec_from_manifest_entry(BACKWARD_SHAPES[0])
CUDNN_FAST_PATH_KEYS = (
    "b2_sq8192_sk8192_hq16_hkv16_d128_bf16_causal",
    "b4_sq4096_sk4096_hq32_hkv32_d128_bf16_causal",
)
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
CUDNN_FAST_PATHS = [
    spec_from_manifest_entry(
        next(entry for entry in SHAPES if entry["key"] == key)
    )
    for key in CUDNN_FAST_PATH_KEYS
]
CUDNN_FAST_PATH = CUDNN_FAST_PATHS[0]
PAGED_KVCACHE = AttnShape(
    batch=4,
    seqlen_q=1,
    seqlen_k=1024,
    nheads_q=8,
    nheads_kv=2,
    head_dim=128,
    dtype=torch.bfloat16,
    causal=True,
)
PAGED_CHUNKED_PREFILL = AttnShape(
    batch=2,
    seqlen_q=200,
    seqlen_k=320,
    nheads_q=8,
    nheads_kv=2,
    head_dim=128,
    dtype=torch.bfloat16,
    causal=True,
)
CHUNKED_PREFILL_KEY = "b1_sq64_sk320_hq8_hkv2_d128_bf16_causal"
GENERIC_DENSE_SPECS = [
    AttnShape(2, 23, 23, 4, 4, 32, torch.bfloat16, False),
    AttnShape(2, 19, 31, 8, 2, 64, torch.bfloat16, True),
    AttnShape(1, 29, 17, 4, 4, 128, torch.float16, True),
    AttnShape(1, 13, 21, 8, 2, 256, torch.float16, True),
]


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


def make_rotary_tables(
    spec: AttnShape, *, rotary_dim: int | None = None, seed: int = 11
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim is None:
        rotary_dim = spec.head_dim
    generator = torch.Generator(device="cuda").manual_seed(seed)
    angles = torch.randn(
        (spec.seqlen_k, rotary_dim // 2),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return angles.cos().to(spec.dtype), angles.sin().to(spec.dtype)


def reference_interleaved_rotary(
    tensor: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    position: int,
) -> torch.Tensor:
    rotary_dim = rotary_cos.shape[1] * 2
    pairs = tensor[..., :rotary_dim].float().reshape(
        *tensor.shape[:-1], rotary_dim // 2, 2
    )
    cos = rotary_cos[position].float()
    sin = rotary_sin[position].float()
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated_prefix = (
        torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        .flatten(-2)
        .to(tensor.dtype)
    )
    if rotary_dim == tensor.shape[-1]:
        return rotated_prefix
    return torch.cat((rotated_prefix, tensor[..., rotary_dim:]), dim=-1)


def page_logical_caches(
    spec: AttnShape,
    logical_caches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Store logical caches on reverse-ordered physical pages."""
    blocks_per_request = [
        (key.shape[0] + page_size - 1) // page_size
        for key, _ in logical_caches
    ]
    num_blocks = sum(blocks_per_request) + 3
    k_cache = torch.zeros(
        num_blocks,
        page_size,
        spec.nheads_kv,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(
        spec.batch,
        (spec.seqlen_k + page_size - 1) // page_size,
        device="cuda",
        dtype=torch.int32,
    )
    physical_block = num_blocks - 1
    for request, ((logical_k, logical_v), request_blocks) in enumerate(
        zip(logical_caches, blocks_per_request)
    ):
        for logical_block in range(request_blocks):
            block_table[request, logical_block] = physical_block
            start = logical_block * page_size
            stop = min(start + page_size, logical_k.shape[0])
            k_cache[physical_block, : stop - start] = logical_k[start:stop]
            v_cache[physical_block, : stop - start] = logical_v[start:stop]
            physical_block -= 1
    return k_cache, v_cache, block_table


def make_paged_kvcache_inputs(
    *,
    spec: AttnShape = PAGED_KVCACHE,
    lengths: list[int] | None = None,
    seed: int = 314159,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    """Build ragged logical caches backed by reverse-ordered physical pages."""
    page_size = 16
    if lengths is None:
        lengths = [37, 128, 1024, 5]
    if len(lengths) != spec.batch:
        raise ValueError("lengths must contain one cache length per request")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        spec.batch,
        spec.seqlen_q,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
        generator=generator,
    )
    logical_caches = [
        (
            torch.randn(
                length,
                spec.nheads_kv,
                spec.head_dim,
                device="cuda",
                dtype=spec.dtype,
                generator=generator,
            ),
            torch.randn(
                length,
                spec.nheads_kv,
                spec.head_dim,
                device="cuda",
                dtype=spec.dtype,
                generator=generator,
            ),
        )
        for length in lengths
    ]
    k_cache, v_cache, block_table = page_logical_caches(
        spec, logical_caches, page_size=page_size
    )
    cache_seqlens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return (
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        block_table,
        logical_caches,
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


def reference_single_token_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    spec: AttnShape,
    scale: float,
) -> torch.Tensor:
    grouped_q = q[:, 0].reshape(
        spec.batch,
        spec.nheads_kv,
        spec.nheads_q // spec.nheads_kv,
        spec.head_dim,
    ).float()
    grouped_k_t = k.float().permute(0, 2, 3, 1)
    scores = torch.matmul(grouped_q, grouped_k_t) * scale
    return torch.logsumexp(scores, dim=-1).reshape(
        spec.batch, spec.nheads_q, spec.seqlen_q
    )


def reference_paged_single_token_lse(
    q: torch.Tensor,
    logical_caches: list[tuple[torch.Tensor, torch.Tensor]],
    spec: AttnShape,
    scale: float,
) -> torch.Tensor:
    """Compute ragged GQA LSE without materializing padded cache rows."""
    group = spec.nheads_q // spec.nheads_kv
    lse: list[torch.Tensor] = []
    for query, (logical_k, _) in zip(q, logical_caches):
        grouped_q = query[0].reshape(
            spec.nheads_kv, group, spec.head_dim
        ).float()
        grouped_k_t = logical_k.float().permute(1, 2, 0)
        scores = torch.matmul(grouped_q, grouped_k_t) * scale
        lse.append(torch.logsumexp(scores, dim=-1).reshape(spec.nheads_q, 1))
    return torch.stack(lse)


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
    assert got.is_contiguous()
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
def test_generated_backward_retains_exact_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = spec_from_manifest_entry(BACKWARD_SHAPES[0])
    q, k, v = make_inputs(spec)
    q.requires_grad_()
    sentinel = torch.empty_like(q)
    dispatched: list[AttnShape] = []

    def generated(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg, scale_arg
        dispatched.append(spec_arg)
        return sentinel

    def reject_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("generated backward shape reached SDPA")

    monkeypatch.setattr(helion_attention, "attention_autograd", generated)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_sdpa)

    out = helion_attention.flash_attn_func(
        q, k, v, dropout_p=0.0, shape=spec
    )

    assert out is sentinel
    assert dispatched == [spec]


@requires_cuda
def test_registered_shape_without_backward_dispatches_to_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = next(item for item in SHAPES if item["key"] == CHUNKED_PREFILL_KEY)
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec)
    q.requires_grad_()
    sentinel = torch.empty_like(q)
    dispatched: list[AttnShape] = []

    def fallback(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg, scale_arg
        dispatched.append(spec_arg)
        return sentinel

    def reject_generated(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("shape without generated backward used exact dispatch")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", fallback)
    monkeypatch.setattr(helion_attention, "attention_autograd", reject_generated)

    out = helion_attention.flash_attn_func(
        q, k, v, causal=spec.causal, shape=spec
    )

    assert out is sentinel
    assert dispatched == [spec]


@requires_cuda
@pytest.mark.parametrize("spec", CUDNN_FAST_PATHS, ids=CUDNN_FAST_PATH_KEYS)
def test_cudnn_fast_path_dispatch_is_narrow(
    monkeypatch: pytest.MonkeyPatch, spec: AttnShape
) -> None:
    def empty(seqlen: int, nheads: int) -> torch.Tensor:
        return torch.empty(
            spec.batch,
            seqlen,
            nheads,
            spec.head_dim,
            device="cuda",
            dtype=spec.dtype,
        )

    q = empty(spec.seqlen_q, spec.nheads_q)
    k = empty(spec.seqlen_k, spec.nheads_kv)
    v = empty(spec.seqlen_k, spec.nheads_kv)
    sentinel = torch.empty((), device="cuda")
    dispatched: list[str] = []

    def cudnn(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg
        dispatched.append("cudnn")
        return sentinel

    def lookup_stub(spec_arg: AttnShape):
        del spec_arg

        def generated(
            q_arg: torch.Tensor,
            k_arg: torch.Tensor,
            v_arg: torch.Tensor,
            scale_arg: float,
        ) -> torch.Tensor:
            del q_arg, k_arg, v_arg, scale_arg
            dispatched.append("generated")
            return sentinel

        return generated

    def autograd_fallback(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg, scale_arg, spec_arg
        dispatched.append("autograd")
        return sentinel

    monkeypatch.setattr(
        helion_attention, "dense_attention_cudnn_default_scale", cudnn
    )
    monkeypatch.setattr(helion_attention, "lookup", lookup_stub)
    monkeypatch.setattr(
        helion_attention, "dense_attention_sdpa", autograd_fallback
    )
    monkeypatch.setattr(helion_attention, "_is_sm90", lambda device: True)

    out = helion_attention.flash_attn_func(
        q, k, v, causal=True, shape=spec
    )
    assert out is sentinel
    assert dispatched == ["cudnn"]

    # Supplying even the numerical default explicitly retains generated-kernel
    # dispatch; the fast path is specifically for an omitted/default scale.
    dispatched.clear()
    out = helion_attention.flash_attn_func(
        q,
        k,
        v,
        softmax_scale=1.0 / math.sqrt(spec.head_dim),
        causal=True,
        shape=spec,
    )
    assert out is sentinel
    assert dispatched == ["generated"]

    dispatched.clear()
    out = helion_attention.flash_attn_func(
        q, k, v, causal=True, deterministic=True, shape=spec
    )
    assert out is sentinel
    assert dispatched == ["generated"]

    dispatched.clear()
    monkeypatch.setattr(helion_attention, "_is_sm90", lambda device: False)
    out = helion_attention.flash_attn_func(
        q, k, v, causal=True, shape=spec
    )
    assert out is sentinel
    assert dispatched == ["generated"]

    dispatched.clear()
    monkeypatch.setattr(helion_attention, "_is_sm90", lambda device: True)
    monkeypatch.setattr(
        helion_attention,
        "dense_attention_cudnn_default_scale",
        lambda q_arg, k_arg, v_arg: None,
    )
    out = helion_attention.flash_attn_func(
        q, k, v, causal=True, shape=spec
    )
    assert out is sentinel
    assert dispatched == ["generated"]

    dispatched.clear()
    other_spec = spec_from_manifest_entry(
        next(entry for entry in SHAPES if entry["key"] == CHUNKED_PREFILL_KEY)
    )
    other_q, other_k, other_v = make_inputs(other_spec)
    out = helion_attention.flash_attn_func(
        other_q,
        other_k,
        other_v,
        causal=other_spec.causal,
        shape=other_spec,
    )
    assert out is sentinel
    assert dispatched == ["generated"]

    dispatched.clear()
    q.requires_grad_()
    out = helion_attention.flash_attn_func(
        q, k, v, causal=True, shape=spec
    )
    assert out is sentinel
    assert dispatched == ["autograd"]


@requires_cuda
@pytest.mark.parametrize("spec", CUDNN_FAST_PATHS, ids=CUDNN_FAST_PATH_KEYS)
def test_cudnn_fast_path_falls_back_when_ineligible(
    monkeypatch: pytest.MonkeyPatch, spec: AttnShape
) -> None:
    def empty(seqlen: int, nheads: int) -> torch.Tensor:
        return torch.empty(
            spec.batch,
            seqlen,
            nheads,
            spec.head_dim,
            device="cuda",
            dtype=spec.dtype,
        )

    q = empty(spec.seqlen_q, spec.nheads_q)
    k = empty(spec.seqlen_k, spec.nheads_kv)
    v = empty(spec.seqlen_k, spec.nheads_kv)
    sentinel = torch.empty((), device="cuda")
    generated_calls = 0

    def lookup_stub(spec_arg: AttnShape):
        assert spec_arg == spec

        def generated(
            q_arg: torch.Tensor,
            k_arg: torch.Tensor,
            v_arg: torch.Tensor,
            scale_arg: float,
        ) -> torch.Tensor:
            nonlocal generated_calls
            del q_arg, k_arg, v_arg
            assert scale_arg == 1.0 / math.sqrt(spec.head_dim)
            generated_calls += 1
            return sentinel

        return generated

    monkeypatch.setattr(helion_attention, "lookup", lookup_stub)
    monkeypatch.setattr(helion_attention, "_is_sm90", lambda device: True)

    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    try:
        torch.use_deterministic_algorithms(True)
        out = helion_attention.flash_attn_func(
            q, k, v, causal=True, shape=spec
        )
    finally:
        torch.use_deterministic_algorithms(
            deterministic_enabled, warn_only=deterministic_warn_only
        )
    assert out is sentinel
    assert generated_calls == 1

    monkeypatch.setattr(
        torch.backends.cuda,
        "can_use_cudnn_attention",
        lambda params: False,
    )
    out = helion_attention.flash_attn_func(
        q, k, v, causal=True, shape=spec
    )
    assert out is sentinel
    assert generated_calls == 2


@requires_two_cuda_devices
def test_cudnn_fast_path_uses_tensor_device_for_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm90_devices = [
        index
        for index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(index) == (9, 0)
    ]
    if not sm90_devices:
        pytest.skip("the cuDNN fast path requires an SM90 device")
    target_index = sm90_devices[0]
    other_index = next(
        index
        for index in range(torch.cuda.device_count())
        if index != target_index
    )
    target = torch.device("cuda", target_index)
    spec = CUDNN_FAST_PATH
    q, k, v = make_inputs(spec, device=target, seed=24601)

    with torch.cuda.device(target):
        preflight = helion_attention.dense_attention_cudnn_default_scale(
            q, k, v
        )
    if preflight is None:
        pytest.skip("cuDNN attention is unavailable for the fast-path shape")
    torch.cuda.synchronize(target)
    del preflight

    real_can_use_cudnn = torch.backends.cuda.can_use_cudnn_attention
    probe_devices: list[int] = []

    def observing_can_use_cudnn(params: object) -> bool:
        probe_devices.append(torch.cuda.current_device())
        return real_can_use_cudnn(params)

    def reject_generated(spec_arg: AttnShape):
        raise AssertionError(
            f"eligible non-current-device call fell back: {spec_arg.key}"
        )

    monkeypatch.setattr(
        torch.backends.cuda,
        "can_use_cudnn_attention",
        observing_can_use_cudnn,
    )
    monkeypatch.setattr(helion_attention, "lookup", reject_generated)

    with torch.cuda.device(other_index):
        out = helion_attention.flash_attn_func(
            q, k, v, causal=True, shape=spec
        )
        assert torch.cuda.current_device() == other_index
    torch.cuda.synchronize(target)

    assert probe_devices == [target_index]
    assert out.device == target
    assert out.is_contiguous()


@requires_cuda
def test_concurrent_cudnn_fast_path_preserves_sdpa_backend_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the cuDNN fast path requires SM90")
    spec = CUDNN_FAST_PATH
    q, k, v = make_inputs(spec, seed=8675309)
    preflight = helion_attention.dense_attention_cudnn_default_scale(q, k, v)
    if preflight is None:
        pytest.skip("cuDNN attention is unavailable for the fast-path shape")
    torch.cuda.synchronize(q.device)
    del preflight

    def reject_generated(spec_arg: AttnShape):
        raise AssertionError(
            f"eligible cuDNN call fell back to generated kernel: {spec_arg.key}"
        )

    monkeypatch.setattr(helion_attention, "lookup", reject_generated)
    flags_before = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )

    def run(barrier: Barrier) -> torch.Tensor:
        barrier.wait()
        return helion_attention.flash_attn_func(
            q, k, v, causal=True, shape=spec
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(30):
                barrier = Barrier(2)
                futures = [
                    pool.submit(run, barrier), pool.submit(run, barrier)
                ]
                for future in futures:
                    out = future.result()
                    assert out.shape == q.shape
                    assert out.is_contiguous()
        torch.cuda.synchronize()

        flags_after = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )
        assert flags_after == flags_before

        float32_probe = torch.randn(1, 1, 2, 8, device="cuda")
        probe_out = torch.nn.functional.scaled_dot_product_attention(
            float32_probe, float32_probe, float32_probe
        )
        assert probe_out.shape == float32_probe.shape
    finally:
        torch.backends.cuda.enable_flash_sdp(flags_before[0])
        torch.backends.cuda.enable_mem_efficient_sdp(flags_before[1])
        torch.backends.cuda.enable_math_sdp(flags_before[2])
        torch.backends.cuda.enable_cudnn_sdp(flags_before[3])


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
@pytest.mark.parametrize(
    "spec", GENERIC_DENSE_SPECS, ids=[spec.key for spec in GENERIC_DENSE_SPECS]
)
def test_unregistered_dense_fallback_matches_fp32_sdpa(spec: AttnShape) -> None:
    assert not helion_attention.is_shape_supported(
        spec, dtype=spec.dtype, causal=spec.causal
    )
    assert spec.key not in {str(entry["key"]) for entry in available_shapes()}

    q, k, v = make_inputs(spec, seed=20260808)
    got = helion_attention.flash_attn_func(
        q, k, v, causal=spec.causal, shape=spec
    )
    expected = reference_attention(q, k, v, spec, 1.0 / math.sqrt(spec.head_dim))

    assert got.shape == q.shape
    assert got.dtype == spec.dtype
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("nheads_kv", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
def test_dense_alibi_matches_flash_attention_2(
    causal: bool, nheads_kv: int, batched_slopes: bool
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = AttnShape(2, 11, 17, 4, nheads_kv, 32, torch.bfloat16, causal)
    q, k, v = make_inputs(spec, seed=20260809)
    head_slopes = torch.tensor([0.03, 0.07, 0.13, 0.29], device="cuda")
    slopes = (
        torch.stack((head_slopes, head_slopes.flip(0)))
        if batched_slopes
        else head_slopes
    )

    got = helion_attention.flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        alibi_slopes=slopes,
        shape=spec,
    )
    expected = flash_attn.flash_attn_func(
        q, k, v, causal=causal, alibi_slopes=slopes
    )

    assert got.shape == q.shape
    assert got.dtype == q.dtype
    torch.testing.assert_close(got.float(), expected.float(), atol=5e-2, rtol=2e-2)


@requires_cuda
def test_dense_alibi_bypasses_registered_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = next(item for item in SHAPES if item["key"] == CHUNKED_PREFILL_KEY)
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec, seed=112358)
    slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    sentinel = torch.empty_like(q)
    dispatched: list[tuple[AttnShape, torch.Tensor | None]] = []

    def generic(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
        slopes_arg: torch.Tensor | None,
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg, scale_arg
        dispatched.append((spec_arg, slopes_arg))
        return sentinel

    def reject_specialized(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("ALiBi call reached a generated specialization")

    monkeypatch.setattr(helion_attention, "_generic_dense_forward", generic)
    monkeypatch.setattr(helion_attention, "lookup", reject_specialized)

    out = helion_attention.flash_attn_func(
        q, k, v, causal=spec.causal, alibi_slopes=slopes, shape=spec
    )

    assert out is sentinel
    assert dispatched == [(spec, slopes)]


@requires_cuda
def test_unregistered_dense_packed_entry_points_use_fallback() -> None:
    mha_spec = GENERIC_DENSE_SPECS[0]
    q, k, v = make_inputs(mha_spec, seed=314159)
    qkv = torch.stack((q, k, v), dim=2)
    qkv_out = helion_attention.flash_attn_qkvpacked_func(qkv, shape=mha_spec)
    qkv_expected = reference_attention(
        q, k, v, mha_spec, 1.0 / math.sqrt(mha_spec.head_dim)
    )
    torch.testing.assert_close(
        qkv_out.float(), qkv_expected, atol=5e-2, rtol=2e-2
    )

    gqa_spec = GENERIC_DENSE_SPECS[1]
    q, k, v = make_inputs(gqa_spec, seed=271828)
    kv = torch.stack((k, v), dim=2)
    kv_out = helion_attention.flash_attn_kvpacked_func(
        q, kv, causal=gqa_spec.causal, shape=gqa_spec
    )
    kv_expected = reference_attention(
        q, k, v, gqa_spec, 1.0 / math.sqrt(gqa_spec.head_dim)
    )
    torch.testing.assert_close(kv_out.float(), kv_expected, atol=5e-2, rtol=2e-2)


@requires_cuda
def test_dense_packed_entry_points_inherit_alibi_support() -> None:
    mha_spec = GENERIC_DENSE_SPECS[0]
    q, k, v = make_inputs(mha_spec, seed=173205)
    slopes = torch.linspace(0.02, 0.2, mha_spec.nheads_q, device=q.device)
    qkv = torch.stack((q, k, v), dim=2)
    expected = helion_attention.flash_attn_func(
        q, k, v, alibi_slopes=slopes, shape=mha_spec
    )
    packed = helion_attention.flash_attn_qkvpacked_func(
        qkv, alibi_slopes=slopes, shape=mha_spec
    )
    torch.testing.assert_close(packed, expected)

    gqa_spec = GENERIC_DENSE_SPECS[1]
    q, k, v = make_inputs(gqa_spec, seed=223607)
    slopes = torch.linspace(
        0.01,
        0.2,
        gqa_spec.batch * gqa_spec.nheads_q,
        device=q.device,
    ).view(gqa_spec.batch, gqa_spec.nheads_q)
    kv = torch.stack((k, v), dim=2)
    expected = helion_attention.flash_attn_func(
        q,
        k,
        v,
        causal=gqa_spec.causal,
        alibi_slopes=slopes,
        shape=gqa_spec,
    )
    packed = helion_attention.flash_attn_kvpacked_func(
        q,
        kv,
        causal=gqa_spec.causal,
        alibi_slopes=slopes,
        shape=gqa_spec,
    )
    torch.testing.assert_close(packed, expected)


@requires_cuda
def test_dense_packed_entry_points_propagate_sdpa_gradients() -> None:
    mha_spec = GENERIC_DENSE_SPECS[0]
    q, k, v = make_inputs(mha_spec, seed=12345)
    qkv = torch.stack((q, k, v), dim=2).requires_grad_()
    grad_out = make_inputs(mha_spec, seed=54321)[0]
    qkv_out = helion_attention.flash_attn_qkvpacked_func(qkv, shape=mha_spec)
    qkv_grad = torch.autograd.grad(qkv_out, qkv, grad_out)[0]

    qkv_ref = qkv.float().detach().requires_grad_()
    q_ref, k_ref, v_ref = (qkv_ref[:, :, index] for index in range(3))
    qkv_expected = reference_attention(
        q_ref, k_ref, v_ref, mha_spec, 1.0 / math.sqrt(mha_spec.head_dim)
    )
    qkv_expected_grad = torch.autograd.grad(
        qkv_expected, qkv_ref, grad_out.float()
    )[0]
    torch.testing.assert_close(
        qkv_grad.float(), qkv_expected_grad, atol=5e-2, rtol=2e-2
    )

    gqa_spec = GENERIC_DENSE_SPECS[1]
    q, k, v = make_inputs(gqa_spec, seed=67890)
    q.requires_grad_()
    kv = torch.stack((k, v), dim=2).requires_grad_()
    grad_out = make_inputs(gqa_spec, seed=9876)[0]
    kv_out = helion_attention.flash_attn_kvpacked_func(
        q, kv, causal=gqa_spec.causal, shape=gqa_spec
    )
    q_grad, kv_grad = torch.autograd.grad(kv_out, (q, kv), grad_out)

    q_ref = q.float().detach().requires_grad_()
    kv_ref = kv.float().detach().requires_grad_()
    k_ref, v_ref = (kv_ref[:, :, index] for index in range(2))
    kv_expected = reference_attention(
        q_ref, k_ref, v_ref, gqa_spec, 1.0 / math.sqrt(gqa_spec.head_dim)
    )
    q_expected_grad, kv_expected_grad = torch.autograd.grad(
        kv_expected, (q_ref, kv_ref), grad_out.float()
    )
    torch.testing.assert_close(
        q_grad.float(), q_expected_grad, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        kv_grad.float(), kv_expected_grad, atol=5e-2, rtol=2e-2
    )


@requires_cuda
def test_encoder_dropout_dense_forward_and_qkv_gradients_match_direct_sdpa() -> None:
    spec = ENCODER_TRAINING
    q, k, v = make_inputs(spec, seed=244949)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    grad_out = make_inputs(spec, seed=264575)[0]
    dropout_p = 0.25
    scale = 0.137

    torch.cuda.manual_seed_all(20260808)
    got = helion_attention.flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=scale,
        shape=spec,
    )
    torch.cuda.manual_seed_all(20260808)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        scale=scale,
    ).transpose(1, 2).contiguous()

    got_grads = torch.autograd.grad(got, (q, k, v), grad_out)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad_out)

    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)
    for actual, reference in zip(got_grads, expected_grads):
        torch.testing.assert_close(actual, reference, atol=2e-3, rtol=1e-2)


@requires_cuda
def test_encoder_dropout_qkvpacked_forward_and_gradient_match_direct_sdpa() -> None:
    spec = ENCODER_TRAINING
    q, k, v = make_inputs(spec, seed=282842)
    qkv = torch.stack((q, k, v), dim=2).requires_grad_()
    grad_out = make_inputs(spec, seed=316227)[0]
    dropout_p = 0.25
    scale = 1.0 / math.sqrt(spec.head_dim)

    torch.cuda.manual_seed_all(20260808)
    got = helion_attention.flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        shape=spec,
    )
    q_ref, k_ref, v_ref = (
        qkv[:, :, index].contiguous() for index in range(3)
    )
    torch.cuda.manual_seed_all(20260808)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        dropout_p=dropout_p,
        scale=scale,
    ).transpose(1, 2).contiguous()

    got_grad = torch.autograd.grad(got, qkv, grad_out)[0]
    expected_grad = torch.autograd.grad(expected, qkv, grad_out)[0]

    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(got_grad, expected_grad, atol=2e-3, rtol=1e-2)


@requires_cuda
def test_encoder_dropout_dispatches_to_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ENCODER_TRAINING
    q, k, v = make_inputs(spec, seed=331662)
    sentinel = torch.empty_like(q)
    dispatched: list[tuple[AttnShape, float]] = []

    def fallback(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
        dropout_arg: float,
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg, scale_arg
        dispatched.append((spec_arg, dropout_arg))
        return sentinel

    def reject_generated(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("dropout call reached a generated specialization")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", fallback)
    monkeypatch.setattr(helion_attention, "attention_autograd", reject_generated)
    monkeypatch.setattr(helion_attention, "lookup", reject_generated)

    out = helion_attention.flash_attn_func(
        q, k, v, dropout_p=0.25, shape=spec
    )

    assert out is sentinel
    assert dispatched == [(spec, 0.25)]


@requires_cuda
def test_registered_dense_shape_does_not_use_generic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = next(item for item in SHAPES if item["key"] == CHUNKED_PREFILL_KEY)
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec, seed=161803)

    def reject_fallback(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("registered shape reached generic dense fallback")

    monkeypatch.setattr(helion_attention, "_generic_dense_forward", reject_fallback)
    out = helion_attention.flash_attn_func(q, k, v, causal=True, shape=spec)
    assert out.shape == q.shape


@requires_cuda
@pytest.mark.parametrize(
    "spec", GENERIC_DENSE_SPECS, ids=[spec.key for spec in GENERIC_DENSE_SPECS]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.137], ids=["default-scale", "custom-scale"]
)
def test_dense_sdpa_fallback_gradients_match_fp32(
    spec: AttnShape, softmax_scale: float | None
) -> None:
    q, k, v = make_inputs(spec, seed=4242)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    grad_out = make_inputs(spec, seed=9090)[0]

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
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    expected = reference_attention(q_ref, k_ref, v_ref, spec, scale)
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )

    assert got.shape == q.shape
    assert got.dtype == spec.dtype
    assert got.is_contiguous()
    torch.testing.assert_close(
        got.float(), expected, atol=5e-2, rtol=2e-2
    )
    for actual, reference in zip(got_grads, expected_grads):
        torch.testing.assert_close(
            actual.float(), reference, atol=5e-2, rtol=2e-2
        )
    if spec.causal and spec.seqlen_q > spec.seqlen_k:
        masked_rows = spec.seqlen_q - spec.seqlen_k
        assert torch.count_nonzero(got[:, :masked_rows]).item() == 0
        assert torch.count_nonzero(got_grads[0][:, :masked_rows]).item() == 0


@requires_cuda
@pytest.mark.parametrize(
    ("input_dtype", "autocast_dtype"),
    [
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
    ids=["bf16-input-fp16-autocast", "fp16-input-bf16-autocast"],
)
def test_dense_sdpa_fallback_preserves_dtype_under_cross_dtype_autocast(
    input_dtype: torch.dtype, autocast_dtype: torch.dtype
) -> None:
    spec = AttnShape(1, 17, 19, 4, 4, 32, input_dtype, True)
    q, k, v = make_inputs(spec, seed=2468)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    grad_out = make_inputs(spec, seed=1357)[0]

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        got = helion_attention.flash_attn_func(
            q, k, v, causal=True, shape=spec
        )
    got_grads = torch.autograd.grad(got, (q, k, v), grad_out)

    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    expected = reference_attention(
        q_ref, k_ref, v_ref, spec, 1.0 / math.sqrt(spec.head_dim)
    )
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )

    assert got.dtype == input_dtype
    assert all(grad.dtype == input_dtype for grad in got_grads)
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    for actual, reference in zip(got_grads, expected_grads):
        torch.testing.assert_close(
            actual.float(), reference, atol=5e-2, rtol=2e-2
        )


@requires_cuda
def test_unequal_causal_sdpa_fallback_has_bounded_peak_allocation() -> None:
    # A materialized [8191, 8192] bool mask alone is about 64 MiB. The lazy
    # lower-right bias should leave only fused-SDPA output/autograd storage.
    spec = AttnShape(1, 8191, 8192, 4, 1, 32, torch.float16, True)
    q, k, v = make_inputs(spec, seed=8642)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    torch.cuda.synchronize(q.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(q.device)
    allocated_before = torch.cuda.memory_allocated(q.device)

    out = helion_attention.flash_attn_func(
        q, k, v, causal=True, shape=spec
    )
    torch.cuda.synchronize(q.device)
    incremental_peak = (
        torch.cuda.max_memory_allocated(q.device) - allocated_before
    )

    assert out.shape == q.shape
    assert out.grad_fn is not None
    assert incremental_peak < 32 * 1024 * 1024, incremental_peak


@requires_cuda
def test_dense_sdpa_fallback_rejects_deterministic_backward() -> None:
    spec = GENERIC_DENSE_SPECS[0]
    q, k, v = make_inputs(spec)
    q.requires_grad_()
    with pytest.raises(NotImplementedError, match="deterministic=True"):
        helion_attention.flash_attn_func(
            q, k, v, deterministic=True, shape=spec
        )


def test_generic_dense_layout_accepts_exact_int32_element_offset_boundary() -> None:
    # 8,388,608 rows * 256 elements has a final offset of INT32_MAX.
    spec = AttnShape(
        1, 8_388_608, 8_388_608, 1, 1, 256, torch.bfloat16, False
    )
    assert helion_attention._validate_generic_dense_layout(spec) == (
        8_388_608,
        8_388_608,
    )


@pytest.mark.parametrize(
    ("spec", "layout"),
    [
        (
            AttnShape(1, 8_388_609, 1, 1, 1, 256, torch.bfloat16, False),
            "Q/output",
        ),
        (
            AttnShape(1, 1, 8_388_609, 1, 1, 256, torch.bfloat16, False),
            "K/V",
        ),
    ],
    ids=["query-output", "key-value"],
)
def test_generic_dense_layout_rejects_int32_element_offset_overflow(
    spec: AttnShape, layout: str
) -> None:
    with pytest.raises(
        UnsupportedShapeError,
        match=rf"{layout} requires maximum offset .*limit 2147483647",
    ):
        helion_attention._validate_generic_dense_layout(spec)


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
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in DECODE_SHAPES],
)
@pytest.mark.parametrize(
    "entry_point", ["dense", "kvpacked"], ids=["dense", "kv-packed"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_decode_return_attn_probs_matches_fa2(
    entry: dict[str, object],
    entry_point: str,
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = spec_from_manifest_entry(entry)
    q, k, v = make_inputs(spec, seed=20260808)
    declared_shape = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
    )

    with torch.no_grad():
        if entry_point == "dense":
            got = helion_attention.flash_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                return_attn_probs=True,
                shape=declared_shape,
            )
            expected = flash_attn.flash_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                return_attn_probs=True,
            )
        else:
            kv = torch.stack((k, v), dim=2)
            got = helion_attention.flash_attn_kvpacked_func(
                q,
                kv,
                softmax_scale=softmax_scale,
                return_attn_probs=True,
                shape=declared_shape,
            )
            expected = flash_attn.flash_attn_kvpacked_func(
                q,
                kv,
                softmax_scale=softmax_scale,
                return_attn_probs=True,
            )

    assert isinstance(got, tuple) and len(got) == 3
    out, softmax_lse, s_dmask = got
    expected_out, expected_lse, expected_s_dmask = expected
    assert out.shape == expected_out.shape == q.shape
    assert out.dtype == expected_out.dtype == q.dtype
    assert softmax_lse.shape == expected_lse.shape == (
        spec.batch,
        spec.nheads_q,
        spec.seqlen_q,
    )
    assert softmax_lse.dtype == expected_lse.dtype == torch.float32
    assert s_dmask.shape == expected_s_dmask.shape == (0,)
    assert s_dmask.dtype == expected_s_dmask.dtype == q.dtype
    assert s_dmask.device == expected_s_dmask.device == q.device
    torch.testing.assert_close(
        out.float(), expected_out.float(), atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        softmax_lse, expected_lse, atol=2e-3, rtol=1e-5
    )


@requires_cuda
@pytest.mark.parametrize(
    "entry_point", ["dense", "kvpacked"], ids=["dense", "kv-packed"]
)
def test_decode_return_attn_probs_false_retains_exact_dispatch(
    entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = spec_from_manifest_entry(DECODE_SHAPES[0])
    q, k, v = make_inputs(spec, seed=1701)
    sentinel = torch.empty_like(q)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def kernel(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(helion_attention, "lookup", lambda _spec: kernel)
    if entry_point == "dense":
        out = helion_attention.flash_attn_func(
            q, k, v, causal=spec.causal, return_attn_probs=False, shape=spec
        )
    else:
        kv = torch.stack((k, v), dim=2)
        out = helion_attention.flash_attn_kvpacked_func(
            q,
            kv,
            causal=spec.causal,
            return_attn_probs=False,
            shape=spec,
        )

    assert out is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 4
    assert kwargs == {}


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("grad", "grad-enabled"),
        ("alibi", "ALiBi"),
        ("non-decode", "three shipped"),
        ("deterministic", "deterministic=False"),
        ("dropout", "dropout"),
        ("window", "sliding-window"),
        ("softcap", "softcap"),
    ],
)
def test_return_attn_probs_rejects_unsupported_calls_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    decode_spec = spec_from_manifest_entry(DECODE_SHAPES[0])
    q, k, v = make_inputs(decode_spec, seed=31415)
    kwargs: dict[str, object] = {
        "causal": decode_spec.causal,
        "return_attn_probs": True,
        "shape": decode_spec,
    }
    if case == "grad":
        q = q.requires_grad_()
    elif case == "alibi":
        kwargs["alibi_slopes"] = torch.ones(
            decode_spec.nheads_q, device=q.device, dtype=torch.float32
        )
    elif case == "non-decode":
        prefill_spec = spec_from_manifest_entry(
            next(entry for entry in SHAPES if entry["key"] == CHUNKED_PREFILL_KEY)
        )
        q, k, v = make_inputs(prefill_spec, seed=31415)
        kwargs["causal"] = prefill_spec.causal
        kwargs["shape"] = prefill_spec
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "window":
        kwargs["window_size"] = (128, 0)
    else:
        kwargs["softcap"] = 30.0

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported diagnostic call reached dispatch")

    monkeypatch.setattr(helion_attention, "lookup", reject_dispatch)
    monkeypatch.setattr(helion_attention, "_generic_dense_forward", reject_dispatch)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_func(q, k, v, **kwargs)


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
def test_unregistered_dense_fallback_rejects_head_dim_above_256() -> None:
    q = torch.randn(1, 2, 1, 257, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(UnsupportedShapeError, match="head_dim <= 256"):
        helion_attention.flash_attn_func(q, q, q, shape=(1, 2, 1, 257))


@requires_cuda
def test_shape_must_match_tensors() -> None:
    q = torch.randn(2, 1024, 32, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="declared shape"):
        helion_attention.flash_attn_func(q, q, q, shape=(2, 512, 32, 64))


@requires_cuda
def test_rejects_unimplemented_features() -> None:
    q = torch.randn(1, 7, 2, 32, device="cuda", dtype=torch.bfloat16)
    for kwargs in (
        {"dropout_p": 0.1},
        {"window_size": (128, 0)},
        {"softcap": 30.0},
        {"return_attn_probs": True},
    ):
        with pytest.raises(NotImplementedError):
            helion_attention.flash_attn_func(q, q, q, shape=(1, 7, 2, 32), **kwargs)


@pytest.mark.parametrize(
    ("dropout_p", "error", "message"),
    [
        (-0.1, ValueError, r"0\.0 <= dropout_p < 1\.0"),
        (1.0, ValueError, r"0\.0 <= dropout_p < 1\.0"),
        (float("nan"), ValueError, r"0\.0 <= dropout_p < 1\.0"),
        (float("inf"), ValueError, r"0\.0 <= dropout_p < 1\.0"),
        (True, TypeError, "real number"),
        ("0.1", TypeError, "real number"),
    ],
    ids=["negative", "one", "nan", "infinity", "bool", "string"],
)
def test_dense_dropout_rejects_invalid_probability(
    dropout_p: object, error: type[Exception], message: str
) -> None:
    q = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    with pytest.raises(error, match=message):
        helion_attention.flash_attn_func(
            q,
            q,
            q,
            dropout_p=dropout_p,  # type: ignore[arg-type]
            shape=(1, 1, 1, 1),
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("other-shape", "only for the shipped encoder-training profile"),
        ("causal", "only for the shipped encoder-training profile"),
        ("deterministic", "deterministic=True"),
        ("alibi", "dropout combined with ALiBi"),
        ("window", "sliding-window"),
        ("softcap", "softcap"),
        ("return-probs", "return_attn_probs=True"),
    ],
)
def test_dense_dropout_rejects_out_of_scope_calls_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    q = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    kwargs: dict[str, object] = {
        "dropout_p": 0.25,
        "shape": (
            ENCODER_TRAINING.batch,
            ENCODER_TRAINING.seqlen_q,
            ENCODER_TRAINING.nheads_q,
            ENCODER_TRAINING.head_dim,
        ),
    }
    if case == "other-shape":
        kwargs["shape"] = (1, 1, 1, 1)
    elif case == "causal":
        kwargs["causal"] = True
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "alibi":
        kwargs["alibi_slopes"] = torch.ones(ENCODER_TRAINING.nheads_q)
    elif case == "window":
        kwargs["window_size"] = (128, 0)
    elif case == "softcap":
        kwargs["softcap"] = 1.0
    else:
        kwargs["return_attn_probs"] = True

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> torch.Tensor:
        raise AssertionError("out-of-scope dropout call reached dispatch")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    monkeypatch.setattr(helion_attention, "attention_autograd", reject_dispatch)
    monkeypatch.setattr(helion_attention, "lookup", reject_dispatch)
    monkeypatch.setattr(helion_attention, "_generic_dense_forward", reject_dispatch)

    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_func(
            q,
            q,
            q,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not-tensor", "torch.Tensor"),
        ("dtype", "dtype torch.float32"),
        ("head-shape", "shape.*nheads_q"),
        ("batch-shape", "shape.*nheads_q"),
        ("rank", "shape.*nheads_q"),
        ("cpu", "CUDA tensor"),
        ("last-stride", "contiguous.*last dimension"),
    ],
)
def test_dense_alibi_rejects_malformed_slopes_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = GENERIC_DENSE_SPECS[0]
    q, k, v = make_inputs(spec, seed=244949)
    if case == "not-tensor":
        slopes: object = [0.1] * spec.nheads_q
    elif case == "dtype":
        slopes = torch.ones(spec.nheads_q, device=q.device, dtype=torch.float16)
    elif case == "head-shape":
        slopes = torch.ones(spec.nheads_q - 1, device=q.device)
    elif case == "batch-shape":
        slopes = torch.ones(spec.batch + 1, spec.nheads_q, device=q.device)
    elif case == "rank":
        slopes = torch.ones(spec.batch, 1, spec.nheads_q, device=q.device)
    elif case == "cpu":
        slopes = torch.ones(spec.nheads_q)
    else:
        slopes = torch.ones(spec.nheads_q, 2, device=q.device)[:, 0]

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("malformed ALiBi slopes reached generic dispatch")

    monkeypatch.setattr(helion_attention, "_generic_dense_forward", reject_dispatch)
    with pytest.raises((TypeError, ValueError), match=message):
        helion_attention.flash_attn_func(
            q,
            k,
            v,
            alibi_slopes=slopes,  # type: ignore[arg-type]
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize("gradient_source", ["q", "slopes"])
def test_dense_alibi_rejects_gradients_before_dispatch(
    gradient_source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = GENERIC_DENSE_SPECS[0]
    q, k, v = make_inputs(spec, seed=264575)
    slopes = torch.ones(spec.nheads_q, device=q.device)
    if gradient_source == "q":
        q.requires_grad_()
    else:
        slopes.requires_grad_()

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("grad-enabled ALiBi call reached generic dispatch")

    monkeypatch.setattr(helion_attention, "_generic_dense_forward", reject_dispatch)
    with pytest.raises(NotImplementedError, match="ALiBi backward"):
        helion_attention.flash_attn_func(
            q, k, v, alibi_slopes=slopes, shape=spec
        )


def test_shape_argument_is_required() -> None:
    q = torch.zeros(1, 1, 1, 1)
    with pytest.raises(TypeError):
        helion_attention.flash_attn_func(q, q, q)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("packed_shape", "message"),
    [
        ((1, 1, 3, 1), r"rank 5.*got rank 4"),
        ((1, 1, 2, 1, 1), r"packed axis \(dimension 2\).*exactly 3.*got 2"),
        ((1, 1, 4, 1, 1), r"packed axis \(dimension 2\).*exactly 3.*got 4"),
    ],
    ids=["wrong-rank", "undersized-axis", "oversized-axis"],
)
def test_qkvpacked_validates_dense_layout_before_slicing(
    packed_shape: tuple[int, ...], message: str
) -> None:
    qkv = torch.zeros(packed_shape)
    with pytest.raises(ValueError, match=message):
        helion_attention.flash_attn_qkvpacked_func(
            qkv, shape=(1, 1, 1, 1)
        )


@pytest.mark.parametrize(
    ("packed_shape", "message"),
    [
        ((1, 1, 2, 1), r"rank 5.*got rank 4"),
        ((1, 1, 1, 1, 1), r"packed axis \(dimension 2\).*exactly 2.*got 1"),
        ((1, 1, 3, 1, 1), r"packed axis \(dimension 2\).*exactly 2.*got 3"),
    ],
    ids=["wrong-rank", "undersized-axis", "oversized-axis"],
)
def test_kvpacked_validates_dense_layout_before_slicing(
    packed_shape: tuple[int, ...], message: str
) -> None:
    q = torch.zeros(1, 1, 1, 1)
    kv = torch.zeros(packed_shape)
    with pytest.raises(ValueError, match=message):
        helion_attention.flash_attn_kvpacked_func(
            q, kv, shape=(1, 1, 1, 1)
        )


def test_kvcache_shape_argument_is_required() -> None:
    q = torch.zeros(1, 1, 1, 1)
    with pytest.raises(TypeError):
        helion_attention.flash_attn_with_kvcache(q, q, q)  # type: ignore[call-arg]


@requires_cuda
@pytest.mark.parametrize(
    "causal", [True, False], ids=["causal", "default-noncausal"]
)
def test_paged_kvcache_matches_fp32_for_ragged_permuted_pages(
    causal: bool,
) -> None:
    q, k_cache, v_cache, cache_seqlens, block_table, logical_caches = (
        make_paged_kvcache_inputs()
    )
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    scale = 0.37

    declared_shape = PAGED_KVCACHE if causal else (4, 1, 1024, 8, 2, 128)
    causal_kwargs = {"causal": True} if causal else {}
    got = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
        shape=declared_shape,
        **causal_kwargs,
    )
    expected = torch.stack(
        [
            torch.nn.functional.scaled_dot_product_attention(
                query.float().transpose(0, 1).unsqueeze(0),
                logical_k.float().transpose(0, 1).unsqueeze(0),
                logical_v.float().transpose(0, 1).unsqueeze(0),
                scale=scale,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            for query, (logical_k, logical_v) in zip(q, logical_caches)
        ]
    )

    assert got.shape == q.shape
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)


@requires_cuda
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "length_stride", [1, 2], ids=["contiguous-lengths", "stride-2-lengths"]
)
def test_paged_kvcache_returns_fp32_lse_for_ragged_permuted_pages(
    softmax_scale: float | None,
    length_stride: int,
) -> None:
    q, k_cache, v_cache, cache_seqlens, block_table, logical_caches = (
        make_paged_kvcache_inputs()
    )
    if length_stride != 1:
        length_storage = torch.empty(
            PAGED_KVCACHE.batch * length_stride,
            device=cache_seqlens.device,
            dtype=cache_seqlens.dtype,
        )
        length_storage[::length_stride].copy_(cache_seqlens)
        cache_seqlens = length_storage[::length_stride]
    assert cache_seqlens.stride() == (length_stride,)
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    scale = (
        1.0 / math.sqrt(PAGED_KVCACHE.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    result = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
        shape=(4, 1, 1024, 8, 2, 128),
    )
    assert isinstance(result, tuple)
    out, lse = result
    expected_out = torch.stack(
        [
            torch.nn.functional.scaled_dot_product_attention(
                query.float().transpose(0, 1).unsqueeze(0),
                logical_k.float().transpose(0, 1).unsqueeze(0),
                logical_v.float().transpose(0, 1).unsqueeze(0),
                scale=scale,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            for query, (logical_k, logical_v) in zip(q, logical_caches)
        ]
    )
    expected_lse = reference_paged_single_token_lse(
        q, logical_caches, PAGED_KVCACHE, scale
    )

    assert out.shape == (4, 1, 8, 128)
    assert lse.shape == (4, 8, 1)
    assert lse.dtype == torch.float32
    assert lse.device == q.device
    torch.testing.assert_close(out.float(), expected_out, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=1e-3)
    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)


@requires_cuda
def test_paged_kvcache_chunked_prefill_matches_fp32_and_fa2() -> None:
    spec = PAGED_CHUNKED_PREFILL
    q, k_cache, v_cache, cache_seqlens, block_table, logical_caches = (
        make_paged_kvcache_inputs(
            spec=spec,
            lengths=[113, 271],
            seed=20260808,
        )
    )
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    scale = 0.37

    got = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
        causal=True,
        shape=spec,
    )
    expected_fp32 = []
    for query, (logical_k, logical_v) in zip(q, logical_caches):
        row = torch.arange(query.shape[0], device=q.device)[:, None]
        col = torch.arange(logical_k.shape[0], device=q.device)[None, :]
        bottom_right_mask = col <= row + logical_k.shape[0] - query.shape[0]
        expected_fp32.append(
            torch.nn.functional.scaled_dot_product_attention(
                query.float().transpose(0, 1).unsqueeze(0),
                logical_k.float().transpose(0, 1).unsqueeze(0),
                logical_v.float().transpose(0, 1).unsqueeze(0),
                attn_mask=bottom_right_mask,
                scale=scale,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    expected_fp32_tensor = torch.stack(expected_fp32)

    assert got.shape == q.shape
    torch.testing.assert_close(
        got.float(), expected_fp32_tensor, atol=5e-2, rtol=2e-2
    )
    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)

    # FlashAttention is an optional benchmark dependency. FA2 requires paged
    # cache blocks to be a multiple of 256, so re-page the same logical cache.
    try:
        import flash_attn
    except ImportError:
        return
    fa2_k, fa2_v, fa2_block_table = page_logical_caches(
        spec, logical_caches, page_size=256
    )
    expected_fa2 = flash_attn.flash_attn_with_kvcache(
        q,
        fa2_k,
        fa2_v,
        cache_seqlens=cache_seqlens,
        block_table=fa2_block_table,
        softmax_scale=scale,
        causal=True,
    )
    torch.testing.assert_close(got, expected_fa2, atol=2e-2, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize(
    ("spec", "lengths"),
    [
        pytest.param(PAGED_KVCACHE, [37, 128, 1024, 5], id="decode"),
        pytest.param(
            PAGED_CHUNKED_PREFILL,
            [113, 271],
            id="chunked-prefill",
        ),
    ],
)
def test_paged_kvcache_routes_through_core_varlen(
    monkeypatch: pytest.MonkeyPatch,
    spec: AttnShape,
    lengths: list[int],
) -> None:
    q, k_cache, v_cache, cache_seqlens, block_table, _ = (
        make_paged_kvcache_inputs(spec=spec, lengths=lengths)
    )
    sentinel = torch.full_like(q.flatten(0, 1), 3.0)
    seen: dict[str, object] = {}

    def fake_varlen(*args: object, **kwargs: object) -> torch.Tensor:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(helion_attention, "flash_attn_varlen_func", fake_varlen)
    got = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=0.19,
        causal=True,
        shape=spec,
    )

    args = seen["args"]
    kwargs = seen["kwargs"]
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[0].data_ptr() == q.data_ptr()
    assert args[0].shape == sentinel.shape
    assert args[1] is k_cache
    assert args[2] is v_cache
    torch.testing.assert_close(
        args[3],
        torch.arange(
            0,
            (spec.batch + 1) * spec.seqlen_q,
            spec.seqlen_q,
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(args[4][1:] - args[4][:-1], cache_seqlens)
    assert kwargs["block_table"] is block_table
    assert kwargs["softmax_scale"] == 0.19
    assert kwargs["shape"] is spec
    assert got.shape == q.shape
    torch.testing.assert_close(got.flatten(0, 1), sentinel)


@requires_cuda
def test_paged_kvcache_rejects_unsupported_modes_before_varlen_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k_cache, v_cache, cache_seqlens, block_table, _ = (
        make_paged_kvcache_inputs()
    )
    original_k = k_cache.clone()
    original_v = v_cache.clone()

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported paged KV-cache call reached core varlen")

    monkeypatch.setattr(
        helion_attention, "flash_attn_varlen_func", reject_dispatch
    )
    base: dict[str, object] = {
        "cache_seqlens": cache_seqlens,
        "block_table": block_table,
        "causal": True,
        "shape": PAGED_KVCACHE,
    }

    update_shape = (
        PAGED_KVCACHE.batch,
        1,
        PAGED_KVCACHE.nheads_kv,
        PAGED_KVCACHE.head_dim,
    )
    update_k = torch.zeros(update_shape, device=q.device, dtype=q.dtype)
    update_v = torch.zeros_like(update_k)
    with pytest.raises(NotImplementedError, match="read-only"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=update_k,
            v=update_v,
            return_softmax_lse=True,
            **base,
        )
    with pytest.raises(NotImplementedError, match="rotary embeddings"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            rotary_cos=q,
            return_softmax_lse=True,
            **base,
        )
    with pytest.raises(NotImplementedError, match="autograd"):
        helion_attention.flash_attn_with_kvcache(
            q.detach().requires_grad_(),
            k_cache,
            v_cache,
            return_softmax_lse=True,
            **base,
        )

    other_profile = AttnShape(
        4, 1, 512, 8, 2, 128, torch.bfloat16, True
    )
    with pytest.raises(UnsupportedShapeError, match="supports only"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            return_softmax_lse=True,
            **{**base, "shape": other_profile},
        )
    page_32_k = torch.empty(
        1, 32, 2, 128, device=q.device, dtype=torch.bfloat16
    )
    page_32_v = torch.empty_like(page_32_k)
    with pytest.raises(UnsupportedShapeError, match="page_size=16"):
        helion_attention.flash_attn_with_kvcache(
            q,
            page_32_k,
            page_32_v,
            return_softmax_lse=True,
            **base,
        )

    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)


@requires_cuda
def test_paged_kvcache_chunked_prefill_rejects_unsupported_modes_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = PAGED_CHUNKED_PREFILL
    q, k_cache, v_cache, cache_seqlens, block_table, _ = (
        make_paged_kvcache_inputs(spec=spec, lengths=[113, 271])
    )
    original_k = k_cache.clone()
    original_v = v_cache.clone()

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported chunked prefill reached core varlen")

    monkeypatch.setattr(
        helion_attention, "flash_attn_varlen_func", reject_dispatch
    )
    base: dict[str, object] = {
        "cache_seqlens": cache_seqlens,
        "block_table": block_table,
        "causal": True,
        "shape": spec,
    }
    update_shape = (
        spec.batch,
        1,
        spec.nheads_kv,
        spec.head_dim,
    )
    update_k = torch.zeros(update_shape, device=q.device, dtype=q.dtype)
    update_v = torch.zeros_like(update_k)

    with pytest.raises(NotImplementedError, match="read-only"):
        helion_attention.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=update_k, v=update_v, **base
        )
    with pytest.raises(NotImplementedError, match="return_softmax_lse"):
        helion_attention.flash_attn_with_kvcache(
            q, k_cache, v_cache, return_softmax_lse=True, **base
        )
    with pytest.raises(NotImplementedError, match="rotary embeddings"):
        helion_attention.flash_attn_with_kvcache(
            q, k_cache, v_cache, rotary_cos=q, **base
        )
    with pytest.raises(NotImplementedError, match="autograd"):
        helion_attention.flash_attn_with_kvcache(
            q.detach().requires_grad_(), k_cache, v_cache, **base
        )
    with pytest.raises(UnsupportedShapeError, match="causal=True"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False,
            shape=(2, 200, 320, 8, 2, 128),
        )

    other_profile = AttnShape(
        2, 199, 320, 8, 2, 128, torch.bfloat16, True
    )
    with pytest.raises(UnsupportedShapeError, match="supports only"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            **{**base, "shape": other_profile},
        )

    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)


@requires_cuda
def test_paged_kvcache_requires_cuda_int32_batch_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k_cache, v_cache, cache_seqlens, block_table, _ = (
        make_paged_kvcache_inputs()
    )

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("invalid cache lengths reached core varlen")

    monkeypatch.setattr(
        helion_attention, "flash_attn_varlen_func", reject_dispatch
    )
    base: dict[str, object] = {
        "block_table": block_table,
        "causal": True,
        "shape": PAGED_KVCACHE,
    }
    invalid = [
        (None, NotImplementedError, "CUDA int32 tensor"),
        (1024, NotImplementedError, "CUDA int32 tensor"),
        (cache_seqlens[:3], ValueError, "shape"),
        (cache_seqlens.to(torch.int64), ValueError, "torch.int32"),
        (cache_seqlens.cpu(), ValueError, "CUDA tensor"),
    ]
    for lengths, error, message in invalid:
        with pytest.raises(error, match=message):
            helion_attention.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=lengths,
                **base,
            )


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
    DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in DECODE_SHAPES],
)
@pytest.mark.parametrize(
    "softmax_scale",
    [None, 0.37],
    ids=["default-scale", "custom-scale"],
)
def test_kvcache_returns_flash_compatible_softmax_lse(
    entry: dict[str, object], softmax_scale: float | None
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec, seed=20260807)
    out, lse = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        return_softmax_lse=True,
        shape=spec,
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    expected_out = reference_attention(q, k_cache, v_cache, spec, scale)
    expected_lse = reference_single_token_lse(q, k_cache, spec, scale)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert lse.shape == (spec.batch, spec.nheads_q, spec.seqlen_q)
    assert lse.dtype == torch.float32
    assert lse.device == q.device
    torch.testing.assert_close(out.float(), expected_out, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in DECODE_SHAPES],
)
@pytest.mark.parametrize(
    "autocast_dtype",
    [torch.bfloat16, torch.float16],
    ids=["bf16-autocast", "fp16-autocast"],
)
def test_kvcache_softmax_lse_is_not_changed_by_autocast(
    entry: dict[str, object], autocast_dtype: torch.dtype
) -> None:
    spec = spec_from_manifest_entry(entry)
    q = torch.full(
        (spec.batch, 1, spec.nheads_q, spec.head_dim),
        32.0,
        dtype=spec.dtype,
        device="cuda",
    )
    k_cache = torch.full(
        (spec.batch, spec.seqlen_k, spec.nheads_kv, spec.head_dim),
        32.0,
        dtype=spec.dtype,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    scale = 0.1

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        _, lse = helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            softmax_scale=scale,
            causal=spec.causal,
            return_softmax_lse=True,
            shape=spec,
        )

    # Every score is identical, so this reference is independent of the
    # implementation and remains finite even though fp16 matmul would overflow.
    expected_value = (
        32.0 * 32.0 * spec.head_dim * scale + math.log(spec.seqlen_k)
    )
    expected = torch.full_like(lse, expected_value)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse, expected, atol=2e-3, rtol=0.0)


@requires_cuda
def test_kvcache_softmax_lse_has_bounded_peak_allocation() -> None:
    spec = spec_from_manifest_entry(LONG_DECODE)
    q, k_cache, v_cache = make_inputs(spec, seed=4815)

    def invoke() -> tuple[torch.Tensor, torch.Tensor]:
        return helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            causal=spec.causal,
            return_softmax_lse=True,
            shape=spec,
        )

    # Warm Triton's compilation and allocator setup before measuring only the
    # per-call tensors. The split kernel needs about 128 KiB of fp32 partials.
    invoke()
    torch.cuda.synchronize(q.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(q.device)
    allocated_before = torch.cuda.memory_allocated(q.device)

    out, lse = invoke()
    torch.cuda.synchronize(q.device)
    incremental_peak = (
        torch.cuda.max_memory_allocated(q.device) - allocated_before
    )

    assert out.shape == q.shape
    assert lse.shape == (spec.batch, spec.nheads_q, 1)
    assert incremental_peak < 1024 * 1024


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in DECODE_SHAPES],
)
@pytest.mark.parametrize("return_softmax_lse", [False, True], ids=["out", "out-lse"])
def test_kvcache_appends_one_token_in_place_and_attends_to_it(
    entry: dict[str, object], return_softmax_lse: bool
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec, seed=8675309)
    q.fill_(1.0)
    k_cache.zero_()
    v_cache.zero_()
    update_shape = (spec.batch, 1, spec.nheads_kv, spec.head_dim)
    # This final key's score is 4 * sqrt(head_dim), so it dominates even the
    # 16K cache. Its distinctive value makes a missing or delayed V copy turn
    # the output from approximately 7 into zero.
    new_k = torch.full(
        update_shape,
        4.0,
        device=q.device,
        dtype=spec.dtype,
    )
    new_v = torch.full(
        update_shape,
        7.0,
        device=q.device,
        dtype=spec.dtype,
    )
    expected_k = k_cache.clone()
    expected_v = v_cache.clone()
    expected_k[:, -1:] = new_k
    expected_v[:, -1:] = new_v
    k_pointer = k_cache.data_ptr()
    v_pointer = v_cache.data_ptr()

    result = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=new_k,
        v=new_v,
        cache_seqlens=spec.seqlen_k - 1,
        causal=spec.causal,
        return_softmax_lse=return_softmax_lse,
        shape=spec,
    )
    if return_softmax_lse:
        out, lse = result
    else:
        assert isinstance(result, torch.Tensor)
        out = result

    assert k_cache.data_ptr() == k_pointer
    assert v_cache.data_ptr() == v_pointer
    assert torch.equal(k_cache, expected_k)
    assert torch.equal(v_cache, expected_v)
    scale = 1.0 / math.sqrt(spec.head_dim)
    expected_out = reference_attention(q, expected_k, expected_v, spec, scale)
    torch.testing.assert_close(out.float(), expected_out, atol=5e-2, rtol=2e-2)
    assert out.float().amin().item() > 6.5
    if return_softmax_lse:
        expected_lse = reference_single_token_lse(q, expected_k, spec, scale)
        torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES,
    ids=[str(entry["key"]) for entry in DECODE_SHAPES],
)
@pytest.mark.parametrize("return_softmax_lse", [False, True], ids=["out", "out-lse"])
@pytest.mark.parametrize(
    "rotary_dim", [64, 128], ids=["half-head", "full-head"]
)
def test_kvcache_interleaved_rotary_append_matches_fa2(
    entry: dict[str, object], return_softmax_lse: bool, rotary_dim: int
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = spec_from_manifest_entry(entry)
    q, initial_k, initial_v = make_inputs(spec, seed=57721)
    new_k = initial_k[:, :1].clone()
    new_v = initial_v[:, :1].clone()
    rotary_cos, rotary_sin = make_rotary_tables(
        spec, rotary_dim=rotary_dim, seed=46349
    )
    position = spec.seqlen_k - 1

    q_helion = q.clone()
    k_helion = initial_k.clone()
    v_helion = initial_v.clone()
    got = helion_attention.flash_attn_with_kvcache(
        q_helion,
        k_helion,
        v_helion,
        k=new_k,
        v=new_v,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=position,
        causal=spec.causal,
        return_softmax_lse=return_softmax_lse,
        shape=spec,
    )

    q_fa2 = q.clone()
    k_fa2 = initial_k.clone()
    v_fa2 = initial_v.clone()
    expected = flash_attn.flash_attn_with_kvcache(
        q_fa2,
        k_fa2,
        v_fa2,
        k=new_k,
        v=new_v,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=position,
        causal=spec.causal,
        return_softmax_lse=return_softmax_lse,
    )

    expected_appended_k = reference_interleaved_rotary(
        new_k, rotary_cos, rotary_sin, position
    )
    assert torch.equal(q_helion, q)
    assert torch.equal(q_fa2, q)
    assert torch.equal(new_k, initial_k[:, :1])
    assert torch.equal(k_helion, k_fa2)
    assert torch.equal(k_helion[:, -1:], expected_appended_k)
    assert torch.equal(
        k_helion[:, -1:, :, rotary_dim:], new_k[..., rotary_dim:]
    )
    assert torch.equal(v_helion, v_fa2)
    assert torch.equal(v_helion[:, -1:], new_v)

    if return_softmax_lse:
        got_out, got_lse = got
        expected_out, expected_lse = expected
        torch.testing.assert_close(got_lse, expected_lse, atol=2e-6, rtol=0.0)
    else:
        assert isinstance(got, torch.Tensor)
        assert isinstance(expected, torch.Tensor)
        got_out, expected_out = got, expected
    torch.testing.assert_close(
        got_out.float(), expected_out.float(), atol=1e-3, rtol=0.0
    )


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_rejects_invalid_updates_before_mutating_either_cache(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec, seed=271828)
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    new_k = k_cache[:, :1].clone()
    new_v = v_cache[:, :1].clone()
    base_kwargs: dict[str, object] = {
        "k": new_k,
        "v": new_v,
        "cache_seqlens": spec.seqlen_k - 1,
    }

    def reject(
        error: type[Exception],
        match: str,
        *,
        q_arg: torch.Tensor = q,
        **overrides: object,
    ) -> None:
        kwargs = {**base_kwargs, **overrides}
        with pytest.raises(error, match=match):
            helion_attention.flash_attn_with_kvcache(
                q_arg,
                k_cache,
                v_cache,
                causal=spec.causal,
                shape=spec,
                **kwargs,
            )
        assert torch.equal(k_cache, original_k)
        assert torch.equal(v_cache, original_v)

    reject(ValueError, "provided together", v=None)
    reject(ValueError, "provided together", k=None)

    multi_k = torch.cat((new_k, new_k), dim=1)
    multi_v = torch.cat((new_v, new_v), dim=1)
    reject(NotImplementedError, "multi-token", k=multi_k, v=multi_v)
    reject(
        NotImplementedError,
        "tensor-valued cache_seqlens",
        cache_seqlens=torch.tensor(
            [spec.seqlen_k - 1], device=q.device, dtype=torch.int32
        ),
    )
    reject(ValueError, "must be a Python int", cache_seqlens=None)
    reject(NotImplementedError, "final cache slot", cache_seqlens=spec.seqlen_k - 2)
    reject(ValueError, "out of range", cache_seqlens=-1)
    reject(ValueError, "out of range", cache_seqlens=spec.seqlen_k)
    reject(ValueError, "dtype", v=new_v.to(torch.float16))

    noncontiguous_v = torch.randn(
        (spec.batch, 1, spec.head_dim, spec.nheads_kv),
        device=q.device,
        dtype=spec.dtype,
    ).transpose(-1, -2)
    assert not noncontiguous_v.is_contiguous()
    reject(ValueError, "contiguous", v=noncontiguous_v)

    grad_q = q.detach().requires_grad_()
    reject(NotImplementedError, "autograd", q_arg=grad_q)
    reject(TypeError, "float", softmax_scale=object())

    reject(
        NotImplementedError,
        "partial or ragged",
        k=None,
        v=None,
        cache_seqlens=spec.seqlen_k - 1,
    )


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_rejects_invalid_rotary_before_mutating_either_cache(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec, seed=31623)
    original_q = q.clone()
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    new_k = k_cache[:, :1].clone()
    new_v = v_cache[:, :1].clone()
    rotary_cos, rotary_sin = make_rotary_tables(
        spec, rotary_dim=64, seed=14159
    )
    base_kwargs: dict[str, object] = {
        "k": new_k,
        "v": new_v,
        "rotary_cos": rotary_cos,
        "rotary_sin": rotary_sin,
        "cache_seqlens": spec.seqlen_k - 1,
    }

    def reject(error: type[Exception], match: str, **overrides: object) -> None:
        kwargs = {**base_kwargs, **overrides}
        with pytest.raises(error, match=match):
            helion_attention.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                causal=spec.causal,
                shape=spec,
                **kwargs,
            )
        assert torch.equal(q, original_q)
        assert torch.equal(k_cache, original_k)
        assert torch.equal(v_cache, original_v)

    reject(ValueError, "provided together", rotary_sin=None)
    reject(ValueError, "provided together", rotary_cos=None)
    reject(NotImplementedError, "non-interleaved", rotary_interleaved=False)
    reject(NotImplementedError, "one-token KV-cache append", k=None, v=None)

    partial_cos = rotary_cos[:, :-1].contiguous()
    partial_sin = rotary_sin[:, :-1].contiguous()
    reject(
        NotImplementedError,
        "partial rotary dimensions",
        rotary_cos=partial_cos,
        rotary_sin=partial_sin,
    )
    reject(
        ValueError,
        "same shape",
        rotary_sin=rotary_sin[:, :-1].contiguous(),
    )
    reject(
        ValueError,
        "final cache position",
        rotary_cos=rotary_cos[:-1].contiguous(),
        rotary_sin=rotary_sin[:-1].contiguous(),
    )
    reject(ValueError, "dtype", rotary_cos=rotary_cos.float())
    reject(ValueError, "CUDA tensor", rotary_sin=rotary_sin.cpu())
    reject(
        ValueError,
        "contiguous",
        rotary_cos=rotary_cos.T.contiguous().T,
    )
    reject(
        ValueError,
        "shape",
        rotary_cos=rotary_cos.unsqueeze(0),
    )
    reject(TypeError, "torch.Tensor", rotary_sin=object())

    grad_cos = rotary_cos.detach().requires_grad_()
    reject(NotImplementedError, "autograd", rotary_cos=grad_cos)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_inference_tensor_lifecycle_cannot_half_apply_update(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, initial_k, initial_v = make_inputs(spec, seed=141421)
    new_k = initial_k[:, :1].clone()
    new_v = initial_v[:, :1].clone()
    assert not torch.equal(initial_k[:, -1:], new_k)
    assert not torch.equal(initial_v[:, -1:], new_v)
    with torch.inference_mode():
        k_cache = initial_k.clone()
        v_cache = initial_v.clone()

    assert k_cache.is_inference()
    assert v_cache.is_inference()
    with pytest.raises(RuntimeError, match="while torch.inference_mode.*enabled"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=new_k,
            v=new_v,
            cache_seqlens=spec.seqlen_k - 1,
            causal=spec.causal,
            shape=spec,
        )

    assert torch.equal(k_cache, initial_k)
    assert torch.equal(v_cache, initial_v)

    # The same inference caches remain appendable in their valid lifecycle.
    with torch.inference_mode():
        result = helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=new_k,
            v=new_v,
            cache_seqlens=spec.seqlen_k - 1,
            causal=spec.causal,
            shape=spec,
        )

    assert isinstance(result, torch.Tensor)
    assert torch.equal(k_cache[:, -1:], new_k)
    assert torch.equal(v_cache[:, -1:], new_v)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_stages_v_update_aliased_to_k_cache(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec, seed=173205)
    q.fill_(1.0)
    k_cache.zero_()
    v_cache.zero_()
    update_shape = (spec.batch, 1, spec.nheads_kv, spec.head_dim)
    new_k = torch.full(update_shape, 4.0, device=q.device, dtype=spec.dtype)
    aliased_v = k_cache[:, -1:]
    aliased_v.fill_(7.0)
    expected_v = aliased_v.clone()

    result = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=new_k,
        v=aliased_v,
        cache_seqlens=spec.seqlen_k - 1,
        causal=spec.causal,
        shape=spec,
    )

    assert isinstance(result, torch.Tensor)
    assert torch.equal(k_cache[:, -1:], new_k)
    assert torch.equal(v_cache[:, -1:], expected_v)
    assert result.float().amin().item() > 6.5


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_stages_partially_overlapping_v_update(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec, seed=223607)
    update_shape = (spec.batch, 1, spec.nheads_kv, spec.head_dim)
    elements_per_token = spec.nheads_kv * spec.head_dim
    target_offset = (
        v_cache.storage_offset() + (spec.seqlen_k - 1) * elements_per_token
    )
    overlapping_v = v_cache.as_strided(
        update_shape,
        v_cache.stride(),
        storage_offset=target_offset - 1,
    )
    assert overlapping_v.is_contiguous()
    expected_v = overlapping_v.clone()
    new_k = k_cache[:, :1].clone()

    result = helion_attention.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=new_k,
        v=overlapping_v,
        cache_seqlens=spec.seqlen_k - 1,
        causal=spec.causal,
        shape=spec,
    )

    assert isinstance(result, torch.Tensor)
    assert torch.equal(k_cache[:, -1:], new_k)
    assert torch.equal(v_cache[:, -1:], expected_v)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_rejects_overlapping_caches_before_mutation(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, _ = make_inputs(spec, seed=244949)
    v_cache = k_cache
    original_q = q.clone()
    original_cache = k_cache.clone()
    update_shape = (spec.batch, 1, spec.nheads_kv, spec.head_dim)
    new_k = torch.full(update_shape, 3.0, device=q.device, dtype=spec.dtype)
    new_v = torch.full(update_shape, 5.0, device=q.device, dtype=spec.dtype)

    with pytest.raises(ValueError, match="k_cache and v_cache must not overlap"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=new_k,
            v=new_v,
            cache_seqlens=spec.seqlen_k - 1,
            causal=spec.causal,
            shape=spec,
        )

    assert torch.equal(q, original_q)
    assert torch.equal(k_cache, original_cache)
    assert torch.equal(v_cache, original_cache)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_rejects_query_cache_overlap_before_mutation(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    _, k_cache, v_cache = make_inputs(spec, seed=264575)
    query_tokens = spec.nheads_q // spec.nheads_kv
    q = k_cache[:, -query_tokens:].reshape(
        spec.batch, 1, spec.nheads_q, spec.head_dim
    )
    assert q.is_contiguous()
    original_q = q.clone()
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    new_k = k_cache[:, :1].clone()
    new_v = v_cache[:, :1].clone()

    with pytest.raises(ValueError, match="q must not overlap"):
        helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=new_k,
            v=new_v,
            cache_seqlens=spec.seqlen_k - 1,
            causal=spec.causal,
            shape=spec,
        )

    assert torch.equal(q, original_q)
    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_one_token_append_supports_cuda_graph_capture(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec, seed=161803)
    new_k = k_cache[:, :1].clone()
    new_v = v_cache[:, :1].clone()

    def append() -> torch.Tensor:
        result = helion_attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=new_k,
            v=new_v,
            cache_seqlens=spec.seqlen_k - 1,
            causal=spec.causal,
            shape=spec,
        )
        assert isinstance(result, torch.Tensor)
        return result

    # Warm Triton's JIT before capture, as callers do for the read-only path.
    expected = append().clone()
    torch.cuda.synchronize(q.device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = append()

    k_cache[:, -1:].zero_()
    v_cache[:, -1:].zero_()
    graph.replay()
    torch.cuda.synchronize(q.device)

    assert torch.equal(k_cache[:, -1:], new_k)
    assert torch.equal(v_cache[:, -1:], new_v)
    torch.testing.assert_close(captured, expected)


@requires_cuda
@pytest.mark.parametrize(
    "entry",
    DECODE_SHAPES[:1],
    ids=[str(entry["key"]) for entry in DECODE_SHAPES[:1]],
)
def test_kvcache_read_only_partial_cache_is_rejected(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)
    q, k_cache, v_cache = make_inputs(spec)
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
