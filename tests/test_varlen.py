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
PAGED_DECODE = AttnShape(
    batch=4,
    seqlen_q=1,
    seqlen_k=1024,
    nheads_q=8,
    nheads_kv=2,
    head_dim=128,
    dtype=torch.bfloat16,
    causal=True,
)


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


def make_paged_decode(
    *, seed: int = 2026
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    """Build ragged logical caches backed by reverse-ordered physical pages."""
    spec = PAGED_DECODE
    page_size = 16
    lengths_k = [37, 128, 1024, 5]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        spec.batch,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
        generator=generator,
    )
    request_kv = [
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
        for length in lengths_k
    ]
    blocks_per_request = [
        (length + page_size - 1) // page_size for length in lengths_k
    ]
    total_blocks = sum(blocks_per_request) + 3
    k = torch.zeros(
        total_blocks,
        page_size,
        spec.nheads_kv,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    v = torch.zeros_like(k)
    block_table = torch.zeros(
        spec.batch,
        spec.seqlen_k // page_size,
        device="cuda",
        dtype=torch.int32,
    )
    physical = total_blocks - 1
    for request, ((key, value), request_blocks) in enumerate(
        zip(request_kv, blocks_per_request)
    ):
        for logical in range(request_blocks):
            block_table[request, logical] = physical
            start = logical * page_size
            stop = min(start + page_size, key.shape[0])
            k[physical, : stop - start] = key[start:stop]
            v[physical, : stop - start] = value[start:stop]
            physical -= 1
    cu_q = torch.arange(spec.batch + 1, device="cuda", dtype=torch.int32)
    cu_k = _cumulative(lengths_k, q.device)
    return q, k, v, cu_q, cu_k, block_table, lengths_k, request_kv


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


@pytest.mark.parametrize(
    ("name", "parameter_names"),
    [
        (
            "flash_attn_varlen_qkvpacked_func",
            [
                "qkv",
                "cu_seqlens",
                "max_seqlen",
                "dropout_p",
                "softmax_scale",
                "causal",
                "window_size",
                "softcap",
                "alibi_slopes",
                "deterministic",
                "return_attn_probs",
                "shape",
            ],
        ),
        (
            "flash_attn_varlen_kvpacked_func",
            [
                "q",
                "kv",
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
                "shape",
            ],
        ),
    ],
)
def test_varlen_packed_signatures_match_flash_attention_2_8_3_plus_shape(
    name: str, parameter_names: list[str]
) -> None:
    assert name in helion_attention.__all__
    signature = inspect.signature(getattr(helion_attention, name))
    assert list(signature.parameters) == parameter_names
    assert signature.parameters["shape"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["shape"].default is inspect.Parameter.empty
    expected_defaults = {
        "dropout_p": 0.0,
        "softmax_scale": None,
        "causal": False,
        "window_size": (-1, -1),
        "softcap": 0.0,
        "alibi_slopes": None,
        "deterministic": False,
        "return_attn_probs": False,
    }
    for parameter, default in expected_defaults.items():
        assert signature.parameters[parameter].default == default


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("dropout_p", 0.1, "dropout"),
        ("window_size", (1, 1), "sliding-window"),
        ("softcap", 1.0, "softcap"),
        ("alibi_slopes", torch.ones(1), "ALiBi"),
        ("return_attn_probs", True, "return_attn_probs"),
    ],
)
@pytest.mark.parametrize(
    "name",
    ["flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func"],
)
def test_varlen_packed_entry_points_reject_unsupported_options(
    name: str, option: str, value: object, message: str
) -> None:
    cu_seqlens = torch.zeros(2, dtype=torch.int32)
    kwargs = {option: value, "shape": (1, 1, 1, 1)}
    with pytest.raises(NotImplementedError, match=message):
        if name == "flash_attn_varlen_qkvpacked_func":
            helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.zeros(1, 3, 1, 1), cu_seqlens, 1, **kwargs
            )
        else:
            helion_attention.flash_attn_varlen_kvpacked_func(
                torch.zeros(1, 1, 1),
                torch.zeros(1, 2, 1, 1),
                cu_seqlens,
                cu_seqlens,
                1,
                1,
                **kwargs,
            )


def test_varlen_packed_entry_points_validate_the_packed_axis() -> None:
    cu_seqlens = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(ValueError, match="qkv must have shape"):
        helion_attention.flash_attn_varlen_qkvpacked_func(
            torch.zeros(1, 2, 1, 1),
            cu_seqlens,
            1,
            shape=(1, 1, 1, 1),
        )
    with pytest.raises(ValueError, match="kv must have shape"):
        helion_attention.flash_attn_varlen_kvpacked_func(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 3, 1, 1),
            cu_seqlens,
            cu_seqlens,
            1,
            1,
            shape=(1, 1, 1, 1),
        )


@requires_cuda
@pytest.mark.parametrize("entry", VARLEN_SHAPES, ids=VARLEN_IDS)
def test_varlen_packed_entry_points_match_unpacked(
    entry: dict[str, object],
) -> None:
    spec = spec_from_manifest_entry(entry)

    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    kv = torch.stack((k, v), dim=1)
    kvpacked = helion_attention.flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        causal=spec.causal,
        shape=spec,
    )
    unpacked = helion_attention.flash_attn_varlen_func(
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
    torch.testing.assert_close(kvpacked, unpacked)

    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2)
    qkv = torch.stack((q, k, v), dim=1)
    qkvpacked = helion_attention.flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_q,
        spec.seqlen_q,
        causal=spec.causal,
        shape=spec,
    )
    unpacked = helion_attention.flash_attn_varlen_func(
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
    torch.testing.assert_close(qkvpacked, unpacked)


@requires_cuda
@pytest.mark.parametrize(
    "name",
    ["flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func"],
)
def test_varlen_packed_entry_points_reject_gradients(name: str) -> None:
    entry = next(item for item in VARLEN_SHAPES if not item["causal"])
    spec = spec_from_manifest_entry(entry)
    variant = 2 if name == "flash_attn_varlen_qkvpacked_func" else 0
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=variant)

    with pytest.raises(NotImplementedError, match="forward-only"):
        if name == "flash_attn_varlen_qkvpacked_func":
            qkv = torch.stack((q, k, v), dim=1).requires_grad_()
            helion_attention.flash_attn_varlen_qkvpacked_func(
                qkv, cu_q, spec.seqlen_q, shape=spec
            )
        else:
            kv = torch.stack((k, v), dim=1).requires_grad_()
            helion_attention.flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                shape=spec,
            )


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


@requires_cuda
def test_core_varlen_paged_decode_matches_fp32_with_permuted_pages() -> None:
    q, k, v, cu_q, cu_k, block_table, _, request_kv = make_paged_decode()
    scale = 0.37
    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=block_table,
        shape=PAGED_DECODE,
    )
    expected = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                query.float().transpose(0, 1).unsqueeze(0),
                key.float().transpose(0, 1).unsqueeze(0),
                value.float().transpose(0, 1).unsqueeze(0),
                scale=scale,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            for query, (key, value) in zip(q.split(1), request_kv)
        ]
    )
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)


@requires_cuda
def test_core_varlen_derives_paged_used_lengths_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, block_table, lengths_k, _ = make_paged_decode()
    seen: dict[str, object] = {}

    def fake_kernel(*args: object) -> torch.Tensor:
        seen["args"] = args
        return torch.zeros_like(q)

    def fake_lookup(spec: AttnShape, page_size: int):
        assert spec == PAGED_DECODE
        assert page_size == 16
        return fake_kernel

    monkeypatch.setattr(helion_attention, "lookup_paged", fake_lookup)
    result = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        causal=True,
        block_table=block_table,
        shape=PAGED_DECODE,
    )

    args = seen["args"]
    assert isinstance(args, tuple)
    seqused_k = args[4]
    assert isinstance(seqused_k, torch.Tensor)
    assert seqused_k.is_cuda
    assert seqused_k.device == cu_k.device
    assert seqused_k.dtype == torch.int32
    torch.testing.assert_close(
        seqused_k, torch.tensor(lengths_k, device="cuda", dtype=torch.int32)
    )
    torch.testing.assert_close(result, torch.zeros_like(q))


@requires_cuda
def test_core_varlen_paged_rejects_malformed_storage_before_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode()

    def reject_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("malformed paged inputs reached kernel lookup")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_lookup)
    call = (
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
    )
    kwargs = {"causal": True, "block_table": block_table, "shape": PAGED_DECODE}

    with pytest.raises(ValueError, match="paged KV cache"):
        helion_attention.flash_attn_varlen_func(
            q,
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            **kwargs,
        )
    with pytest.raises(ValueError, match="same shape"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v[:-1],
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            **kwargs,
        )
    with pytest.raises(ValueError, match="capacity"):
        helion_attention.flash_attn_varlen_func(
            *call, **{**kwargs, "block_table": block_table[:, :-1]}
        )
    with pytest.raises(ValueError, match="torch.int32"):
        helion_attention.flash_attn_varlen_func(
            *call, **{**kwargs, "block_table": block_table.to(torch.int64)}
        )


@requires_cuda
def test_core_varlen_paged_rejects_unsupported_page_and_shape() -> None:
    q, _, _, cu_q, cu_k, _, *_ = make_paged_decode()
    k = torch.zeros(
        1, 32, 2, 128, device="cuda", dtype=PAGED_DECODE.dtype
    )
    v = torch.zeros_like(k)
    block_table = torch.zeros(4, 32, device="cuda", dtype=torch.int32)
    with pytest.raises(helion_attention.UnsupportedShapeError, match="page_size=16"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            causal=True,
            block_table=block_table,
            shape=PAGED_DECODE,
        )

    chunked = AttnShape(2, 200, 320, 8, 2, 128, torch.bfloat16, True)
    q = q[:2]
    k = torch.zeros(1, 16, 2, 128, device="cuda", dtype=chunked.dtype)
    v = torch.zeros_like(k)
    cu_q = torch.arange(3, device="cuda", dtype=torch.int32)
    cu_k = torch.arange(3, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(2, 20, device="cuda", dtype=torch.int32)
    with pytest.raises(helion_attention.UnsupportedShapeError, match="supports only"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            chunked.seqlen_q,
            chunked.seqlen_k,
            causal=True,
            block_table=block_table,
            shape=chunked,
        )


@requires_cuda
def test_core_varlen_paged_rejects_gradients_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode()

    def reject_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("gradient-bearing input reached paged kernel lookup")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_lookup)
    q.requires_grad_()
    with pytest.raises(NotImplementedError, match="paged-cache backward"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            causal=True,
            block_table=block_table,
            shape=PAGED_DECODE,
        )


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
    with pytest.raises(ValueError, match="paged KV cache"):
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
