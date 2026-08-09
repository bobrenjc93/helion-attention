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
VARLEN_ALIBI = AttnShape(
    batch=8,
    seqlen_q=512,
    seqlen_k=512,
    nheads_q=16,
    nheads_kv=16,
    head_dim=64,
    dtype=torch.bfloat16,
    causal=True,
)
VARLEN_DIAGNOSTIC = AttnShape(
    batch=8,
    seqlen_q=512,
    seqlen_k=512,
    nheads_q=16,
    nheads_kv=16,
    head_dim=64,
    dtype=torch.bfloat16,
    causal=False,
)
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
PagedInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    list[tuple[torch.Tensor, torch.Tensor]],
]


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


def make_paged_inputs(
    spec: AttnShape,
    lengths_q: list[int],
    lengths_k: list[int],
    *,
    seed: int,
) -> PagedInputs:
    """Build ragged packed queries and reverse-mapped physical cache pages."""
    page_size = 16
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sum(lengths_q),
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
    return (
        q,
        k,
        v,
        _cumulative(lengths_q, q.device),
        _cumulative(lengths_k, q.device),
        block_table,
        lengths_q,
        lengths_k,
        request_kv,
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
    q, k, v, cu_q, cu_k, block_table, _, lengths_k, request_kv = (
        make_paged_inputs(
            PAGED_DECODE,
            [1] * PAGED_DECODE.batch,
            [37, 128, 1024, 5],
            seed=seed,
        )
    )
    return q, k, v, cu_q, cu_k, block_table, lengths_k, request_kv


def make_paged_chunked_prefill(
    *,
    lengths_q: tuple[int, int] = (137, 200),
    lengths_k: tuple[int, int] = (233, 320),
    seed: int = 2027,
) -> PagedInputs:
    """Build ragged chunked-prefill inputs on reverse-mapped cache pages."""
    return make_paged_inputs(
        PAGED_CHUNKED_PREFILL,
        list(lengths_q),
        list(lengths_k),
        seed=seed,
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
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    q_start = 0
    k_start = 0
    for batch, (seqlen_q, seqlen_k) in enumerate(zip(lengths_q, lengths_k)):
        q_seq = q[q_start : q_start + seqlen_q].float().transpose(0, 1)[None]
        k_seq = k[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        v_seq = v[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        mask: torch.Tensor | None = None
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        if causal:
            mask = col <= row + seqlen_k - seqlen_q
        if alibi_slopes is not None:
            slopes = (
                alibi_slopes
                if alibi_slopes.ndim == 1
                else alibi_slopes[batch]
            )
            distance = torch.abs(row + seqlen_k - seqlen_q - col)
            bias = -slopes.float()[:, None, None] * distance.float()[None]
            if mask is not None:
                bias = bias.masked_fill(~mask[None], float("-inf"))
            mask = bias[None]
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
                torch.zeros(1, 3, 1, 1, dtype=torch.bfloat16),
                cu_seqlens,
                1,
                **kwargs,
            )
        else:
            helion_attention.flash_attn_varlen_kvpacked_func(
                torch.zeros(1, 1, 1, dtype=torch.bfloat16),
                torch.zeros(1, 2, 1, 1, dtype=torch.bfloat16),
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
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_varlen_return_attn_probs_matches_fa2_for_dynamic_ragged_calls(
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_DIAGNOSTIC
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0, seed=20260808)

    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            return_attn_probs=True,
            shape=spec,
        )
        expected = flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            return_attn_probs=True,
        )

    assert isinstance(got, tuple) and len(got) == 3
    out, softmax_lse, s_dmask = got
    expected_out, expected_lse, expected_s_dmask = expected
    assert out.shape == expected_out.shape == q.shape
    assert out.dtype == expected_out.dtype == torch.bfloat16
    assert softmax_lse.shape == expected_lse.shape == (
        spec.nheads_q,
        q.shape[0],
    )
    assert softmax_lse.dtype == expected_lse.dtype == torch.float32
    assert s_dmask.shape == expected_s_dmask.shape == (0,)
    assert s_dmask.dtype == expected_s_dmask.dtype == torch.bfloat16
    assert s_dmask.device == expected_s_dmask.device == q.device
    torch.testing.assert_close(
        out.float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        softmax_lse, expected_lse, atol=2e-3, rtol=1e-5
    )


@requires_cuda
@pytest.mark.parametrize("entry_point", ["qkvpacked", "kvpacked"])
def test_varlen_packed_entry_points_inherit_diagnostic_return(
    entry_point: str,
) -> None:
    spec = VARLEN_DIAGNOSTIC
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=1, seed=271828)
    if entry_point == "qkvpacked":
        # QKV packing requires self-attention metadata. Preserve the ragged
        # query lengths while giving K and V distinct values.
        k = q.roll(1, dims=0).contiguous()
        v = q.roll(2, dims=0).contiguous()
        cu_k = cu_q

    with torch.no_grad():
        expected = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=0.37,
            return_attn_probs=True,
            shape=spec,
        )
        if entry_point == "qkvpacked":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=0.37,
                return_attn_probs=True,
                shape=spec,
            )
        else:
            got = helion_attention.flash_attn_varlen_kvpacked_func(
                q,
                torch.stack((k, v), dim=1),
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                softmax_scale=0.37,
                return_attn_probs=True,
                shape=spec,
            )

    assert isinstance(got, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_cuda
def test_varlen_return_attn_probs_false_retains_generated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_DIAGNOSTIC
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generated(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_diagnostic(*args: object, **kwargs: object) -> object:
        raise AssertionError("ordinary varlen call reached diagnostic dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", lambda _spec: generated)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_diagnostic
    )
    out = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        return_attn_probs=False,
        shape=spec,
    )

    assert out is sentinel
    assert len(calls) == 1
    assert len(calls[0]) == 9


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("causal", "noncausal"),
        ("paged", "block_table"),
        ("grad", "grad-enabled"),
        ("alibi", "ALiBi"),
        ("deterministic", "deterministic=False"),
        ("dropout", "dropout"),
        ("window", "sliding-window"),
        ("softcap", "softcap"),
        ("other-shape", "noncausal"),
    ],
)
def test_varlen_return_attn_probs_rejects_out_of_scope_calls_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_DIAGNOSTIC
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    kwargs: dict[str, object] = {
        "return_attn_probs": True,
        "shape": spec,
    }
    if case == "causal":
        spec = VARLEN_ALIBI
        q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
        kwargs.update(causal=True, shape=spec)
    elif case == "paged":
        kwargs["block_table"] = torch.zeros(1, device=q.device)
    elif case == "grad":
        q.requires_grad_()
    elif case == "alibi":
        kwargs["alibi_slopes"] = torch.ones(
            spec.nheads_q, device=q.device, dtype=torch.float32
        )
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "window":
        kwargs["window_size"] = (128, 0)
    elif case == "softcap":
        kwargs["softcap"] = 1.0
    else:
        spec = AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, False)
        kwargs["shape"] = spec

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError("out-of-scope diagnostic call reached dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_dispatch
    )
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("batched_slopes", "softmax_scale", "variant"),
    [
        pytest.param(False, None, 0, id="head-default-scale"),
        pytest.param(True, None, 1, id="batch-head-default-scale"),
        pytest.param(False, 0.37, 1, id="head-custom-scale"),
        pytest.param(True, 0.37, 0, id="batch-head-custom-scale"),
    ],
)
def test_varlen_alibi_matches_fa2_and_fp32_for_dynamic_ragged_calls(
    batched_slopes: bool, softmax_scale: float | None, variant: int
) -> None:
    spec = VARLEN_ALIBI
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=variant, seed=20260808
    )
    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )
        if batched_slopes
        else head_slopes
    )

    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        alibi_slopes=slopes,
        shape=spec,
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    expected_fp32 = reference_packed(
        q,
        k,
        v,
        lengths_q,
        lengths_k,
        causal=True,
        scale=scale,
        alibi_slopes=slopes,
    )

    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

    # FlashAttention is an optional benchmark dependency. The independent fp32
    # assertion above must still run in environments with only the dev extras.
    try:
        import flash_attn
    except ImportError:
        return
    expected_fa2 = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        alibi_slopes=slopes,
    )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=1e-2, rtol=1e-2
    )


@requires_cuda
def test_varlen_packed_entry_points_inherit_alibi_support() -> None:
    spec = VARLEN_ALIBI
    slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device="cuda")

    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2, seed=314159)
    unpacked = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        causal=True,
        alibi_slopes=slopes,
        shape=spec,
    )
    qkv = torch.stack((q, k, v), dim=1)
    qkvpacked = helion_attention.flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_q,
        spec.seqlen_q,
        causal=True,
        alibi_slopes=slopes,
        shape=spec,
    )
    torch.testing.assert_close(qkvpacked, unpacked)

    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0, seed=271828)
    batched_slopes = torch.stack(
        [slopes.roll(batch) for batch in range(spec.batch)], dim=0
    )
    unpacked = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        causal=True,
        alibi_slopes=batched_slopes,
        shape=spec,
    )
    kv = torch.stack((k, v), dim=1)
    kvpacked = helion_attention.flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        causal=True,
        alibi_slopes=batched_slopes,
        shape=spec,
    )
    torch.testing.assert_close(kvpacked, unpacked)


@requires_cuda
def test_varlen_without_alibi_retains_generated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_ALIBI
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    sentinel = torch.empty_like(q)
    seen: list[AttnShape] = []

    def generated(*args: object) -> torch.Tensor:
        return sentinel

    def lookup(spec_arg: AttnShape) -> object:
        seen.append(spec_arg)
        return generated

    def reject_generic(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("slope-free call reached generic varlen dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", lookup)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_generic
    )
    out = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        causal=True,
        alibi_slopes=None,
        shape=spec,
    )

    assert out is sentinel
    assert seen == [spec]


@requires_cuda
def test_varlen_alibi_bypasses_generated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_ALIBI
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=1)
    slopes = torch.ones(spec.nheads_q, device=q.device)
    sentinel = torch.empty_like(q)
    seen: list[tuple[float, AttnShape, torch.Tensor]] = []

    def reject_generated(*args: object, **kwargs: object) -> object:
        raise AssertionError("ALiBi call reached generated varlen dispatch")

    def generic(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        cu_q_arg: torch.Tensor,
        cu_k_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
        slopes_arg: torch.Tensor,
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg, cu_q_arg, cu_k_arg
        seen.append((scale_arg, spec_arg, slopes_arg))
        return sentinel

    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_generated)
    monkeypatch.setattr(helion_attention, "_generic_varlen_alibi_forward", generic)
    out = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=0.37,
        causal=True,
        alibi_slopes=slopes,
        shape=spec,
    )

    assert out is sentinel
    assert seen == [(0.37, spec, slopes)]


@requires_cuda
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            AttnShape(8, 512, 512, 16, 16, 64, torch.bfloat16, False),
            id="noncausal-profile",
        ),
        pytest.param(
            AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, True),
            id="other-causal-shape",
        ),
    ],
)
def test_varlen_alibi_rejects_other_profiles(spec: AttnShape) -> None:
    q = torch.zeros(
        spec.batch,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    cu_seqlens = torch.arange(spec.batch + 1, device="cuda", dtype=torch.int32)
    slopes = torch.ones(spec.nheads_q, device="cuda")

    with pytest.raises(NotImplementedError, match="implemented only"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=spec.causal,
            alibi_slopes=slopes,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize("gradient_source", ["q", "slopes"])
def test_varlen_alibi_rejects_gradients_before_dispatch(
    gradient_source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_ALIBI
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    slopes = torch.ones(spec.nheads_q, device=q.device)
    if gradient_source == "q":
        q.requires_grad_()
    else:
        slopes.requires_grad_()

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("grad-enabled ALiBi call reached generic dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_dispatch
    )
    with pytest.raises(NotImplementedError, match="ALiBi backward"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            alibi_slopes=slopes,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("dropout_p", 0.1, "dropout"),
        ("window_size", (1, 1), "sliding-window"),
        ("softcap", 1.0, "softcap"),
        ("return_attn_probs", True, "return_attn_probs"),
    ],
)
def test_varlen_alibi_rejects_incompatible_options(
    option: str, value: object, message: str
) -> None:
    spec = VARLEN_ALIBI
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    kwargs = {
        option: value,
        "causal": True,
        "alibi_slopes": torch.ones(spec.nheads_q, device=q.device),
        "shape": spec,
    }
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,
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
def test_varlen_alibi_rejects_malformed_slopes_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_ALIBI
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
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

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_dispatch
    )
    with pytest.raises((TypeError, ValueError), match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            alibi_slopes=slopes,  # type: ignore[arg-type]
            shape=spec,
        )


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
@pytest.mark.parametrize(
    ("lengths_q", "lengths_k"),
    [
        pytest.param((137, 200), (233, 320), id="maxima-split-across-requests"),
        pytest.param((200, 17), (113, 271), id="fully-masked-prefix"),
    ],
)
def test_core_varlen_paged_chunked_prefill_matches_fp32_with_permuted_pages(
    lengths_q: tuple[int, int], lengths_k: tuple[int, int]
) -> None:
    q, k, v, cu_q, cu_k, block_table, actual_q, actual_k, request_kv = (
        make_paged_chunked_prefill(lengths_q=lengths_q, lengths_k=lengths_k)
    )
    scale = 0.37
    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_CHUNKED_PREFILL.seqlen_q,
        PAGED_CHUNKED_PREFILL.seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=block_table,
        shape=PAGED_CHUNKED_PREFILL,
    )
    expected = reference_packed(
        q,
        torch.cat([key for key, _ in request_kv]),
        torch.cat([value for _, value in request_kv]),
        actual_q,
        actual_k,
        causal=True,
        scale=scale,
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

    unsupported = AttnShape(2, 199, 320, 8, 2, 128, torch.bfloat16, True)
    q = q[:2]
    k = torch.zeros(1, 16, 2, 128, device="cuda", dtype=unsupported.dtype)
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
            unsupported.seqlen_q,
            unsupported.seqlen_k,
            causal=True,
            block_table=block_table,
            shape=unsupported,
        )


@requires_cuda
def test_core_varlen_paged_rejects_alibi_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode()

    def reject_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("paged ALiBi call reached kernel lookup")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_lookup)
    with pytest.raises(NotImplementedError, match="ALiBi"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            causal=True,
            alibi_slopes=torch.ones(PAGED_DECODE.nheads_q, device=q.device),
            block_table=block_table,
            shape=PAGED_DECODE,
        )


@requires_cuda
def test_core_varlen_paged_chunked_prefill_rejects_noncausal_mode_before_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = PAGED_CHUNKED_PREFILL
    q = torch.zeros(2, 8, 128, device="cuda", dtype=spec.dtype)
    k = torch.zeros(1, 16, 2, 128, device="cuda", dtype=spec.dtype)
    v = torch.zeros_like(k)
    cu_seqlens = torch.arange(3, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(2, 20, device="cuda", dtype=torch.int32)

    def reject_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-causal chunked prefill reached kernel lookup")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_lookup)
    with pytest.raises(helion_attention.UnsupportedShapeError, match="causal=True"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=False,
            block_table=block_table,
            shape=(2, 200, 320, 8, 2, 128),
        )


@requires_cuda
@pytest.mark.parametrize(
    ("spec", "factory"),
    [
        pytest.param(PAGED_DECODE, make_paged_decode, id="decode"),
        pytest.param(
            PAGED_CHUNKED_PREFILL,
            make_paged_chunked_prefill,
            id="chunked-prefill",
        ),
    ],
)
def test_core_varlen_paged_rejects_gradients_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    spec: AttnShape,
    factory: object,
) -> None:
    assert callable(factory)
    q, k, v, cu_q, cu_k, block_table, *_ = factory()

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
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            block_table=block_table,
            shape=spec,
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
