"""Packed variable-length API and generated-kernel regressions."""

from __future__ import annotations

import inspect
import math
import sys

import pytest
import torch

import helion_attention
from helion_attention import AttnShape
from helion_attention._registry import spec_from_manifest_entry

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
VARLEN_SHAPES = helion_attention.available_varlen_shapes()
VARLEN_IDS = [str(entry["key"]) for entry in VARLEN_SHAPES]
VARLEN_ALIBI_CAUSAL = AttnShape(
    batch=8,
    seqlen_q=512,
    seqlen_k=512,
    nheads_q=16,
    nheads_kv=16,
    head_dim=64,
    dtype=torch.bfloat16,
    causal=True,
)
VARLEN_ALIBI_NONCAUSAL = AttnShape(
    batch=8,
    seqlen_q=512,
    seqlen_k=512,
    nheads_q=16,
    nheads_kv=16,
    head_dim=64,
    dtype=torch.bfloat16,
    causal=False,
)
VARLEN_ALIBI_PROFILES = (VARLEN_ALIBI_CAUSAL, VARLEN_ALIBI_NONCAUSAL)
VARLEN_DIAGNOSTIC = VARLEN_ALIBI_CAUSAL
VARLEN_SYMMETRIC_WINDOW = VARLEN_ALIBI_NONCAUSAL
VARLEN_CAUSAL_LEFT_WINDOW = VARLEN_ALIBI_CAUSAL
VARLEN_CAUSAL_LEFT_WINDOW_SIZE = (127, 0)
VARLEN_SOFTCAP = VARLEN_ALIBI_CAUSAL
VARLEN_SOFTCAP_VALUE = 50.0
VARLEN_NON_DIAGNOSTIC_SOFTCAP = 30.0
LLAMA3_VARLEN_INFERENCE = AttnShape(
    batch=4,
    seqlen_q=256,
    seqlen_k=256,
    nheads_q=32,
    nheads_kv=8,
    head_dim=128,
    dtype=torch.bfloat16,
    causal=True,
)
BERT_BASE_VARLEN_INFERENCE = AttnShape(
    batch=16,
    seqlen_q=512,
    seqlen_k=512,
    nheads_q=12,
    nheads_kv=12,
    head_dim=64,
    dtype=torch.bfloat16,
    causal=False,
)
FULL_VARLEN_BACKWARD_PROFILES = (
    *VARLEN_ALIBI_PROFILES,
    BERT_BASE_VARLEN_INFERENCE,
)
FULL_VARLEN_BACKWARD_IDS = ["causal", "noncausal", "bert-base"]
BERT_BASE_RAGGED_SELF_LENGTHS = [
    512,
    479,
    443,
    401,
    367,
    331,
    293,
    257,
    211,
    179,
    149,
    113,
    79,
    47,
    19,
    3,
]
BERT_BASE_RAGGED_CROSS_KEY_LENGTHS = [
    7,
    31,
    59,
    89,
    127,
    163,
    197,
    229,
    263,
    307,
    349,
    383,
    419,
    457,
    491,
    512,
]
RAGGED_SELF_LENGTHS = [512, 401, 300, 255, 128, 63, 17, 1]
MIXED_EMPTY_SELF_LENGTHS = [512, 0, 300, 255, 0, 63, 17, 0]
RAGGED_CROSS_QUERY_LENGTHS = [512, 401, 300, 255, 128, 63, 17, 1]
RAGGED_CROSS_KEY_LENGTHS = [3, 29, 97, 191, 257, 319, 443, 511]
MIXED_EMPTY_CROSS_QUERY_LENGTHS = [512, 0, 300, 255, 0, 63, 17, 0]
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
CORE_PAGED_SOFTCAP_VALUE = 50.0
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


def make_ragged_self_packed(
    spec: AttnShape,
    *,
    seed: int = 123,
    lengths: list[int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
]:
    if lengths is None:
        lengths = RAGGED_SELF_LENGTHS
    assert spec.batch == len(lengths)
    assert max(lengths) <= spec.seqlen_q == spec.seqlen_k
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (sum(lengths), spec.nheads_q, spec.head_dim),
        device="cuda",
        dtype=spec.dtype,
        generator=generator,
    )
    k = torch.randn(
        (sum(lengths), spec.nheads_kv, spec.head_dim),
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
    return q, k, v, _cumulative(lengths, q.device), lengths


def make_ragged_cross_packed(
    spec: AttnShape,
    *,
    seed: int = 123,
    lengths_q: list[int] | None = None,
    lengths_k: list[int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
]:
    if lengths_q is None:
        lengths_q = RAGGED_CROSS_QUERY_LENGTHS
    if lengths_k is None:
        lengths_k = RAGGED_CROSS_KEY_LENGTHS
    assert spec.batch == len(lengths_q) == len(lengths_k)
    assert max(lengths_q) <= spec.seqlen_q
    assert max(lengths_k) <= spec.seqlen_k
    assert lengths_q != lengths_k
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
    page_size: int = 16,
) -> PagedInputs:
    """Build ragged packed queries and reverse-mapped physical cache pages."""
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
        (spec.seqlen_k + page_size - 1) // page_size,
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
    *, seed: int = 2026, page_size: int = 16
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
            page_size=page_size,
        )
    )
    return q, k, v, cu_q, cu_k, block_table, lengths_k, request_kv


def make_paged_chunked_prefill(
    *,
    lengths_q: tuple[int, int] = (137, 200),
    lengths_k: tuple[int, int] = (233, 320),
    seed: int = 2027,
    page_size: int = 16,
) -> PagedInputs:
    """Build ragged chunked-prefill inputs on reverse-mapped cache pages."""
    return make_paged_inputs(
        PAGED_CHUNKED_PREFILL,
        list(lengths_q),
        list(lengths_k),
        seed=seed,
        page_size=page_size,
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
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    q_start = 0
    k_start = 0
    for batch, (seqlen_q, seqlen_k) in enumerate(zip(lengths_q, lengths_k)):
        if seqlen_q == 0:
            k_start += seqlen_k
            continue
        q_seq = q[q_start : q_start + seqlen_q].float().transpose(0, 1)[None]
        k_seq = k[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        v_seq = v[k_start : k_start + seqlen_k].float().transpose(0, 1)[None]
        mask: torch.Tensor | None = None
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        if causal:
            mask = col <= row + seqlen_k - seqlen_q
        if window_size != (-1, -1):
            aligned_row = row + seqlen_k - seqlen_q
            left, right = window_size
            window_mask = torch.ones_like(
                aligned_row + col, dtype=torch.bool
            )
            if left >= 0:
                window_mask &= col >= aligned_row - left
            if right >= 0:
                window_mask &= col <= aligned_row + right
            mask = window_mask if mask is None else mask & window_mask
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


def reference_softcap_packed_diagnostics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths_q: list[int],
    lengths_k: list[int],
    *,
    causal: bool,
    scale: float,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ragged softcapped attention and LSE directly in fp32."""
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    softmax_lse = torch.empty(
        q.shape[1], q.shape[0], device=q.device, dtype=torch.float32
    )
    q_start = 0
    k_start = 0
    for seqlen_q, seqlen_k in zip(lengths_q, lengths_k):
        q_seq = q[q_start : q_start + seqlen_q].float().transpose(0, 1)
        k_seq = k[k_start : k_start + seqlen_k].float().transpose(0, 1)
        v_seq = v[k_start : k_start + seqlen_k].float().transpose(0, 1)
        if q_seq.shape[0] != k_seq.shape[0]:
            group_size = q_seq.shape[0] // k_seq.shape[0]
            k_seq = k_seq.repeat_interleave(group_size, dim=0)
            v_seq = v_seq.repeat_interleave(group_size, dim=0)
        scores = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * scale
        scores = softcap * torch.tanh(scores / softcap)
        if causal:
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            keep = col <= row + seqlen_k - seqlen_q
            scores.masked_fill_(~keep[None], float("-inf"))
        softmax_lse[:, q_start : q_start + seqlen_q] = torch.logsumexp(
            scores, dim=-1
        )
        probabilities = torch.softmax(scores, dim=-1)
        # Bottom-right alignment can leave leading query rows with no keys.
        probabilities.nan_to_num_(nan=0.0)
        result = torch.matmul(probabilities, v_seq).transpose(0, 1)
        out[q_start : q_start + seqlen_q] = result
        q_start += seqlen_q
        k_start += seqlen_k
    return out, softmax_lse


def reference_softcap_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths_q: list[int],
    lengths_k: list[int],
    *,
    causal: bool,
    scale: float,
    softcap: float,
) -> torch.Tensor:
    """Compute ragged softcapped attention directly in fp32."""
    out, _ = reference_softcap_packed_diagnostics(
        q,
        k,
        v,
        lengths_q,
        lengths_k,
        causal=causal,
        scale=scale,
        softcap=softcap,
    )
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
    "entry_point", ["unpacked", "kv-packed"], ids=["unpacked", "kv-packed"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "empty_query_slots", [False, True], ids=["nonempty", "empty-query"]
)
def test_ragged_causal_varlen_cross_attention_backward_matches_fp32_and_fa2(
    entry_point: str,
    softmax_scale: float | None,
    empty_query_slots: bool,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_ALIBI_CAUSAL
    requested_lengths_q = (
        MIXED_EMPTY_CROSS_QUERY_LENGTHS if empty_query_slots else None
    )
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_ragged_cross_packed(
        spec, seed=20260809, lengths_q=requested_lengths_q
    )
    generator = torch.Generator(device=q.device).manual_seed(20260810)
    grad_out = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=generator,
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    if entry_point == "unpacked":
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
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
            shape=spec,
        )
        got_inputs = (q, k, v)

        q_ref = q.float().detach().requires_grad_()
        k_ref = k.float().detach().requires_grad_()
        v_ref = v.float().detach().requires_grad_()
        expected = reference_packed(
            q_ref,
            k_ref,
            v_ref,
            lengths_q,
            lengths_k,
            causal=True,
            scale=scale,
        )
        reference_inputs = (q_ref, k_ref, v_ref)

        q_fa2 = q.detach().requires_grad_()
        k_fa2 = k.detach().requires_grad_()
        v_fa2 = v.detach().requires_grad_()
        expected_fa2 = flash_attn.flash_attn_varlen_func(
            q_fa2,
            k_fa2,
            v_fa2,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
        )
        fa2_inputs = (q_fa2, k_fa2, v_fa2)
    else:
        q.requires_grad_()
        kv = torch.stack((k, v), dim=1).requires_grad_()
        got = helion_attention.flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            shape=spec,
        )
        got_inputs = (q, kv)

        q_ref = q.float().detach().requires_grad_()
        kv_ref = kv.float().detach().requires_grad_()
        k_ref, v_ref = (kv_ref[:, index] for index in range(2))
        expected = reference_packed(
            q_ref,
            k_ref,
            v_ref,
            lengths_q,
            lengths_k,
            causal=True,
            scale=scale,
        )
        reference_inputs = (q_ref, kv_ref)

        q_fa2 = q.detach().requires_grad_()
        kv_fa2 = kv.detach().requires_grad_()
        expected_fa2 = flash_attn.flash_attn_varlen_kvpacked_func(
            q_fa2,
            kv_fa2,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
        )
        fa2_inputs = (q_fa2, kv_fa2)

    got_grads = torch.autograd.grad(got, got_inputs, grad_out)
    expected_grads = torch.autograd.grad(
        expected, reference_inputs, grad_out.float()
    )
    expected_fa2_grads = torch.autograd.grad(
        expected_fa2, fa2_inputs, grad_out
    )

    assert got.shape == q.shape
    assert got.dtype == q.dtype
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )
    for actual, reference, reference_fa2 in zip(
        got_grads, expected_grads, expected_fa2_grads
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=8e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(), reference_fa2.float(), atol=5e-2, rtol=2e-2
        )

    if empty_query_slots:
        if entry_point == "unpacked":
            k_grad, v_grad = got_grads[1:]
        else:
            k_grad, v_grad = got_grads[1].unbind(dim=1)
        k_start = 0
        for query_length, key_length in zip(lengths_q, lengths_k):
            if query_length == 0:
                key_slice = slice(k_start, k_start + key_length)
                assert torch.count_nonzero(k_grad[key_slice]).item() == 0
                assert torch.count_nonzero(v_grad[key_slice]).item() == 0
            k_start += key_length


@requires_cuda
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "empty_slots", [False, True], ids=["nonempty", "mixed-empty"]
)
def test_ragged_causal_varlen_backward_matches_fp32_and_fa2(
    softmax_scale: float | None,
    empty_slots: bool,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_ALIBI_CAUSAL
    requested_lengths = MIXED_EMPTY_SELF_LENGTHS if empty_slots else None
    q, k, v, cu_seqlens, lengths = make_ragged_self_packed(
        spec, seed=20260809, lengths=requested_lengths
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    generator = torch.Generator(device=q.device).manual_seed(20260810)
    grad_out = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=generator,
    )

    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens.clone(),
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        shape=spec,
    )
    got_grads = torch.autograd.grad(got, (q, k, v), grad_out)

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    expected = reference_packed(
        q_ref,
        k_ref,
        v_ref,
        lengths,
        lengths,
        causal=True,
        scale=scale,
    )
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )

    q_fa2 = q.detach().requires_grad_()
    k_fa2 = k.detach().requires_grad_()
    v_fa2 = v.detach().requires_grad_()
    expected_fa2 = flash_attn.flash_attn_varlen_func(
        q_fa2,
        k_fa2,
        v_fa2,
        cu_seqlens,
        cu_seqlens,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
    )
    expected_fa2_grads = torch.autograd.grad(
        expected_fa2, (q_fa2, k_fa2, v_fa2), grad_out
    )

    assert got.shape == q.shape
    assert got.dtype == q.dtype
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )
    for actual, reference, reference_fa2 in zip(
        got_grads, expected_grads, expected_fa2_grads
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=8e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(), reference_fa2.float(), atol=5e-2, rtol=2e-2
        )


@requires_cuda
@pytest.mark.parametrize(
    ("name", "empty_slots"),
    [
        ("flash_attn_varlen_qkvpacked_func", False),
        ("flash_attn_varlen_kvpacked_func", False),
        ("flash_attn_varlen_qkvpacked_func", True),
    ],
    ids=["qkv-packed", "kv-packed", "qkv-packed-mixed-empty"],
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_ragged_causal_varlen_packed_adapters_match_fp32_and_fa2(
    name: str, empty_slots: bool, softmax_scale: float | None
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_ALIBI_CAUSAL
    requested_lengths = MIXED_EMPTY_SELF_LENGTHS if empty_slots else None
    q, k, v, cu_seqlens, lengths = make_ragged_self_packed(
        spec, seed=161803, lengths=requested_lengths
    )
    generator = torch.Generator(device=q.device).manual_seed(271828)
    grad_out = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=generator,
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    if name == "flash_attn_varlen_qkvpacked_func":
        packed = torch.stack((q, k, v), dim=1).requires_grad_()
        got = helion_attention.flash_attn_varlen_qkvpacked_func(
            packed,
            cu_seqlens,
            spec.seqlen_q,
            softmax_scale=softmax_scale,
            causal=True,
            shape=spec,
        )
        got_inputs = (packed,)

        packed_ref = packed.float().detach().requires_grad_()
        q_ref, k_ref, v_ref = (packed_ref[:, index] for index in range(3))
        expected = reference_packed(
            q_ref,
            k_ref,
            v_ref,
            lengths,
            lengths,
            causal=True,
            scale=scale,
        )
        reference_inputs = (packed_ref,)

        packed_fa2 = packed.detach().requires_grad_()
        expected_fa2 = flash_attn.flash_attn_varlen_qkvpacked_func(
            packed_fa2,
            cu_seqlens,
            spec.seqlen_q,
            softmax_scale=softmax_scale,
            causal=True,
        )
        fa2_inputs = (packed_fa2,)
    else:
        q.requires_grad_()
        packed = torch.stack((k, v), dim=1).requires_grad_()
        got = helion_attention.flash_attn_varlen_kvpacked_func(
            q,
            packed,
            cu_seqlens,
            cu_seqlens.clone(),
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            shape=spec,
        )
        got_inputs = (q, packed)

        q_ref = q.float().detach().requires_grad_()
        packed_ref = packed.float().detach().requires_grad_()
        k_ref, v_ref = (packed_ref[:, index] for index in range(2))
        expected = reference_packed(
            q_ref,
            k_ref,
            v_ref,
            lengths,
            lengths,
            causal=True,
            scale=scale,
        )
        reference_inputs = (q_ref, packed_ref)

        q_fa2 = q.detach().requires_grad_()
        packed_fa2 = packed.detach().requires_grad_()
        expected_fa2 = flash_attn.flash_attn_varlen_kvpacked_func(
            q_fa2,
            packed_fa2,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
        )
        fa2_inputs = (q_fa2, packed_fa2)

    got_grads = torch.autograd.grad(got, got_inputs, grad_out)
    expected_grads = torch.autograd.grad(
        expected, reference_inputs, grad_out.float()
    )
    expected_fa2_grads = torch.autograd.grad(
        expected_fa2, fa2_inputs, grad_out
    )
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )
    for actual, reference, reference_fa2 in zip(
        got_grads, expected_grads, expected_fa2_grads
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=8e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(), reference_fa2.float(), atol=5e-2, rtol=2e-2
        )


@requires_cuda
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "spec", FULL_VARLEN_BACKWARD_PROFILES, ids=FULL_VARLEN_BACKWARD_IDS
)
def test_full_varlen_backward_matches_fp32_and_fa2(
    softmax_scale: float | None, spec: AttnShape
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=2, seed=20260808
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    generator = torch.Generator(device=q.device).manual_seed(20260809)
    grad_out = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=generator,
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
        causal=spec.causal,
        shape=spec,
    )
    got_grads = torch.autograd.grad(got, (q, k, v), grad_out)

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    fp32_gradient_atol = 8e-2 if spec.causal else 5e-2
    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    expected = reference_packed(
        q_ref,
        k_ref,
        v_ref,
        lengths_q,
        lengths_k,
        causal=spec.causal,
        scale=scale,
    )
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )

    q_fa2 = q.detach().requires_grad_()
    k_fa2 = k.detach().requires_grad_()
    v_fa2 = v.detach().requires_grad_()
    expected_fa2 = flash_attn.flash_attn_varlen_func(
        q_fa2,
        k_fa2,
        v_fa2,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=spec.causal,
    )
    expected_fa2_grads = torch.autograd.grad(
        expected_fa2, (q_fa2, k_fa2, v_fa2), grad_out
    )

    assert got.shape == q.shape
    assert got.dtype == q.dtype
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )
    for actual, reference, reference_fa2 in zip(
        got_grads, expected_grads, expected_fa2_grads
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=fp32_gradient_atol, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(), reference_fa2.float(), atol=2e-2, rtol=2e-2
        )


@requires_cuda
@pytest.mark.parametrize(
    "name",
    [
        "flash_attn_varlen_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func",
    ],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "spec", FULL_VARLEN_BACKWARD_PROFILES, ids=FULL_VARLEN_BACKWARD_IDS
)
def test_deterministic_full_varlen_training_is_repeatable_and_matches_references(
    name: str,
    softmax_scale: float | None,
    spec: AttnShape,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    base_q, base_k, base_v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=2, seed=20260808
    )
    generator = torch.Generator(device=base_q.device).manual_seed(20260809)
    grad_out = torch.randn(
        base_q.shape,
        device=base_q.device,
        dtype=base_q.dtype,
        generator=generator,
    )

    def run(api: object, *, pass_shape: bool) -> tuple[torch.Tensor, ...]:
        kwargs: dict[str, object] = {
            "softmax_scale": softmax_scale,
            "causal": spec.causal,
            "deterministic": True,
        }
        if pass_shape:
            kwargs["shape"] = spec

        if name == "flash_attn_varlen_func":
            q = base_q.detach().requires_grad_()
            k = base_k.detach().requires_grad_()
            v = base_v.detach().requires_grad_()
            out = api.flash_attn_varlen_func(  # type: ignore[attr-defined]
                q,
                k,
                v,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                **kwargs,
            )
            grads = torch.autograd.grad(out, (q, k, v), grad_out)
        elif name == "flash_attn_varlen_qkvpacked_func":
            qkv = torch.stack((base_q, base_k, base_v), dim=1).requires_grad_()
            out = api.flash_attn_varlen_qkvpacked_func(  # type: ignore[attr-defined]
                qkv,
                cu_q,
                spec.seqlen_q,
                **kwargs,
            )
            qkv_grad = torch.autograd.grad(out, qkv, grad_out)[0]
            grads = tuple(qkv_grad[:, index] for index in range(3))
        else:
            q = base_q.detach().requires_grad_()
            kv = torch.stack((base_k, base_v), dim=1).requires_grad_()
            out = api.flash_attn_varlen_kvpacked_func(  # type: ignore[attr-defined]
                q,
                kv,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                **kwargs,
            )
            q_grad, kv_grad = torch.autograd.grad(out, (q, kv), grad_out)
            grads = (q_grad, kv_grad[:, 0], kv_grad[:, 1])
        return (out.detach(), *(grad.detach() for grad in grads))

    first = run(helion_attention, pass_shape=True)
    repeated = run(helion_attention, pass_shape=True)
    assert all(
        torch.equal(actual, again)
        for actual, again in zip(first, repeated)
    )

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    q_ref = base_q.float().requires_grad_()
    k_ref = base_k.float().requires_grad_()
    v_ref = base_v.float().requires_grad_()
    expected = reference_packed(
        q_ref,
        k_ref,
        v_ref,
        lengths_q,
        lengths_k,
        causal=spec.causal,
        scale=scale,
    )
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )
    expected_fa2 = run(flash_attn, pass_shape=False)

    assert first[0].shape == base_q.shape
    assert first[0].dtype == spec.dtype
    assert first[0].is_contiguous()
    torch.testing.assert_close(
        first[0].float(), expected, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        first[0].float(), expected_fa2[0].float(), atol=5e-2, rtol=2e-2
    )
    fa2_gradient_atol = 8e-2 if softmax_scale is not None else 2e-2
    for actual, reference, reference_fa2 in zip(
        first[1:], expected_grads, expected_fa2[1:]
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=8e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(),
            reference_fa2.float(),
            atol=fa2_gradient_atol,
            rtol=2e-2,
        )


@requires_cuda
@pytest.mark.parametrize(
    "name",
    ["flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func"],
    ids=["qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "spec", FULL_VARLEN_BACKWARD_PROFILES, ids=FULL_VARLEN_BACKWARD_IDS
)
def test_full_varlen_packed_adapters_match_fp32_and_fa2(
    name: str, softmax_scale: float | None, spec: AttnShape
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=2, seed=161803
    )
    generator = torch.Generator(device=q.device).manual_seed(271828)
    grad_out = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=generator,
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    fp32_gradient_atol = (
        8e-2 if spec.causal or softmax_scale is not None else 5e-2
    )

    if name == "flash_attn_varlen_qkvpacked_func":
        qkv = torch.stack((q, k, v), dim=1).requires_grad_()
        got = helion_attention.flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_q,
            spec.seqlen_q,
            softmax_scale=softmax_scale,
            causal=spec.causal,
            shape=spec,
        )
        got_inputs = (qkv,)

        qkv_ref = qkv.float().detach().requires_grad_()
        q_ref, k_ref, v_ref = (qkv_ref[:, index] for index in range(3))
        expected = reference_packed(
            q_ref,
            k_ref,
            v_ref,
            lengths_q,
            lengths_k,
            causal=spec.causal,
            scale=scale,
        )
        reference_inputs = (qkv_ref,)

        qkv_fa2 = qkv.detach().requires_grad_()
        expected_fa2 = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_fa2,
            cu_q,
            spec.seqlen_q,
            softmax_scale=softmax_scale,
            causal=spec.causal,
        )
        fa2_inputs = (qkv_fa2,)
    else:
        q.requires_grad_()
        kv = torch.stack((k, v), dim=1).requires_grad_()
        got = helion_attention.flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=spec.causal,
            shape=spec,
        )
        got_inputs = (q, kv)

        q_ref = q.float().detach().requires_grad_()
        kv_ref = kv.float().detach().requires_grad_()
        k_ref, v_ref = (kv_ref[:, index] for index in range(2))
        expected = reference_packed(
            q_ref,
            k_ref,
            v_ref,
            lengths_q,
            lengths_k,
            causal=spec.causal,
            scale=scale,
        )
        reference_inputs = (q_ref, kv_ref)

        q_fa2 = q.detach().requires_grad_()
        kv_fa2 = kv.detach().requires_grad_()
        expected_fa2 = flash_attn.flash_attn_varlen_kvpacked_func(
            q_fa2,
            kv_fa2,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=spec.causal,
        )
        fa2_inputs = (q_fa2, kv_fa2)

    got_grads = torch.autograd.grad(got, got_inputs, grad_out)
    expected_grads = torch.autograd.grad(
        expected, reference_inputs, grad_out.float()
    )
    expected_fa2_grads = torch.autograd.grad(
        expected_fa2, fa2_inputs, grad_out
    )
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )
    for actual, reference, reference_fa2 in zip(
        got_grads, expected_grads, expected_fa2_grads
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=fp32_gradient_atol, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(), reference_fa2.float(), atol=2e-2, rtol=2e-2
        )


@requires_cuda
@pytest.mark.parametrize(
    "name",
    [
        "flash_attn_varlen_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func",
    ],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    "spec", VARLEN_ALIBI_PROFILES, ids=["causal", "noncausal"]
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_full_varlen_no_grad_retains_generated_dispatch(
    name: str,
    spec: AttnShape,
    deterministic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2)
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generated(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("no-grad varlen call reached dense SDPA")

    def reject_llama3(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("registered varlen call reached Llama-3 fallback")

    monkeypatch.setattr(helion_attention, "lookup_varlen", lambda _spec: generated)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_sdpa)
    monkeypatch.setattr(helion_attention, "dense_attention_math_sdpa", reject_sdpa)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_llama3
    )
    with torch.no_grad():
        if name == "flash_attn_varlen_func":
            out = helion_attention.flash_attn_varlen_func(
                q.requires_grad_(),
                k.requires_grad_(),
                v.requires_grad_(),
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=spec.causal,
                deterministic=deterministic,
                shape=spec,
            )
        elif name == "flash_attn_varlen_qkvpacked_func":
            qkv = torch.stack((q, k, v), dim=1).requires_grad_()
            out = helion_attention.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_q,
                spec.seqlen_q,
                causal=spec.causal,
                deterministic=deterministic,
                shape=spec,
            )
        else:
            kv = torch.stack((k, v), dim=1).requires_grad_()
            out = helion_attention.flash_attn_varlen_kvpacked_func(
                q.requires_grad_(),
                kv,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=spec.causal,
                deterministic=deterministic,
                shape=spec,
            )

    assert out is sentinel
    assert len(calls) == 1


@requires_cuda
@pytest.mark.parametrize(
    "name",
    [
        "flash_attn_varlen_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func",
    ],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    "empty_slots", [False, True], ids=["nonempty", "mixed-empty"]
)
def test_ragged_causal_no_grad_softcap_zero_retains_generated_dispatch(
    name: str, empty_slots: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_ALIBI_CAUSAL
    requested_lengths = MIXED_EMPTY_SELF_LENGTHS if empty_slots else None
    q, k, v, cu_seqlens, _ = make_ragged_self_packed(
        spec, lengths=requested_lengths
    )
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generated(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("no-grad ragged varlen call reached SDPA")

    def reject_softcap(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("softcap=0 call reached softcap dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", lambda _spec: generated)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_sdpa)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_softcap_forward", reject_softcap
    )
    with torch.no_grad():
        if name == "flash_attn_varlen_func":
            out = helion_attention.flash_attn_varlen_func(
                q.requires_grad_(),
                k.requires_grad_(),
                v.requires_grad_(),
                cu_seqlens,
                cu_seqlens,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=True,
                softcap=0.0,
                shape=spec,
            )
        elif name == "flash_attn_varlen_qkvpacked_func":
            packed = torch.stack((q, k, v), dim=1).requires_grad_()
            out = helion_attention.flash_attn_varlen_qkvpacked_func(
                packed,
                cu_seqlens,
                spec.seqlen_q,
                causal=True,
                softcap=0.0,
                shape=spec,
            )
        else:
            packed = torch.stack((k, v), dim=1).requires_grad_()
            out = helion_attention.flash_attn_varlen_kvpacked_func(
                q.requires_grad_(),
                packed,
                cu_seqlens,
                cu_seqlens,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=True,
                softcap=0.0,
                shape=spec,
            )

    assert out is sentinel
    assert len(calls) == 1


@requires_cuda
@pytest.mark.parametrize(
    "name",
    ["flash_attn_varlen_func", "flash_attn_varlen_kvpacked_func"],
    ids=["unpacked", "kv-packed"],
)
@pytest.mark.parametrize(
    "empty_query_slots", [False, True], ids=["nonempty", "empty-query"]
)
def test_ragged_causal_cross_attention_no_grad_retains_generated_dispatch(
    name: str,
    empty_query_slots: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_ALIBI_CAUSAL
    requested_lengths_q = (
        MIXED_EMPTY_CROSS_QUERY_LENGTHS if empty_query_slots else None
    )
    q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
        spec, lengths_q=requested_lengths_q
    )
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generated(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("no-grad ragged cross-attention reached SDPA")

    monkeypatch.setattr(helion_attention, "lookup_varlen", lambda _spec: generated)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_sdpa)
    with torch.no_grad():
        if name == "flash_attn_varlen_func":
            out = helion_attention.flash_attn_varlen_func(
                q.requires_grad_(),
                k.requires_grad_(),
                v.requires_grad_(),
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=True,
                shape=spec,
            )
        else:
            kv = torch.stack((k, v), dim=1).requires_grad_()
            out = helion_attention.flash_attn_varlen_kvpacked_func(
                q.requires_grad_(),
                kv,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=True,
                shape=spec,
            )

    assert out is sentinel
    assert len(calls) == 1


@requires_cuda
@pytest.mark.parametrize(
    "name",
    [
        "flash_attn_varlen_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func",
    ],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    ("deterministic", "expected_dispatch"),
    [(False, "ordinary"), (True, "math")],
)
@pytest.mark.parametrize(
    "spec", FULL_VARLEN_BACKWARD_PROFILES, ids=FULL_VARLEN_BACKWARD_IDS
)
def test_full_varlen_deterministic_backward_dispatch_is_narrow(
    name: str,
    deterministic: bool,
    expected_dispatch: str,
    spec: AttnShape,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2)
    sentinel = torch.empty(
        spec.batch,
        spec.seqlen_q,
        spec.nheads_q,
        spec.head_dim,
        device=q.device,
        dtype=q.dtype,
    )
    dispatched: list[str] = []

    def bridge(name: str):
        def run(
            q_arg: torch.Tensor,
            k_arg: torch.Tensor,
            v_arg: torch.Tensor,
            scale_arg: float,
            spec_arg: AttnShape,
        ) -> torch.Tensor:
            assert q_arg.shape == sentinel.shape
            assert k_arg.shape == sentinel.shape
            assert v_arg.shape == sentinel.shape
            assert scale_arg == 1.0 / math.sqrt(spec.head_dim)
            assert spec_arg == spec
            dispatched.append(name)
            return sentinel

        return run

    monkeypatch.setattr(
        helion_attention, "dense_attention_sdpa", bridge("ordinary")
    )
    monkeypatch.setattr(
        helion_attention, "dense_attention_math_sdpa", bridge("math")
    )
    if name == "flash_attn_varlen_func":
        out = helion_attention.flash_attn_varlen_func(
            q.requires_grad_(),
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=spec.causal,
            deterministic=deterministic,
            shape=spec,
        )
    elif name == "flash_attn_varlen_qkvpacked_func":
        out = helion_attention.flash_attn_varlen_qkvpacked_func(
            torch.stack((q, k, v), dim=1).requires_grad_(),
            cu_q,
            spec.seqlen_q,
            causal=spec.causal,
            deterministic=deterministic,
            shape=spec,
        )
    else:
        out = helion_attention.flash_attn_varlen_kvpacked_func(
            q.requires_grad_(),
            torch.stack((k, v), dim=1).requires_grad_(),
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=spec.causal,
            deterministic=deterministic,
            shape=spec,
        )

    assert out.shape == q.shape
    assert out.untyped_storage().data_ptr() == sentinel.untyped_storage().data_ptr()
    assert dispatched == [expected_dispatch]


@requires_cuda
def test_deterministic_varlen_backward_rejects_other_profile_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = AttnShape(1, 2, 2, 1, 1, 8, torch.bfloat16, False)
    q = torch.randn(2, 1, 8, device="cuda", dtype=spec.dtype).requires_grad_()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = _cumulative([2], q.device)

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported deterministic profile reached dispatch")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "dense_attention_math_sdpa", reject_dispatch
    )
    with pytest.raises(
        NotImplementedError, match="varlen backward is implemented only"
    ):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            deterministic=True,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("dropout_p", 0.1, "dropout"),
        ("window_size", (31, 31), "window"),
        ("softcap", VARLEN_SOFTCAP_VALUE, "softcap"),
        ("alibi_slopes", "slopes", "ALiBi backward"),
        ("return_attn_probs", True, "deterministic=False"),
    ],
)
@pytest.mark.parametrize(
    "spec", FULL_VARLEN_BACKWARD_PROFILES, ids=FULL_VARLEN_BACKWARD_IDS
)
def test_deterministic_full_varlen_backward_rejects_optional_features(
    option: str,
    value: object,
    message: str,
    spec: AttnShape,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2)
    q.requires_grad_()
    if value == "slopes":
        value = torch.ones(spec.nheads_q, device=q.device)

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported deterministic call reached dispatch")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "dense_attention_math_sdpa", reject_dispatch
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
            causal=spec.causal,
            deterministic=True,
            shape=spec,
            **{option: value},
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("noncausal-empty-query", "causal.*profile"),
        ("empty-key-cross", "empty key sequence"),
        ("all-empty-self", "at least one nonempty query"),
        ("all-empty-query-cross", "at least one nonempty query"),
        ("deterministic", "deterministic=True"),
        ("deterministic-noncausal", "deterministic=True"),
        ("deterministic-empty", "deterministic=True"),
        ("deterministic-empty-query-cross", "deterministic=True"),
    ],
)
def test_ragged_varlen_backward_rejects_out_of_scope_calls(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = (
        VARLEN_ALIBI_NONCAUSAL
        if case in {"noncausal-empty-query", "deterministic-noncausal"}
        else VARLEN_ALIBI_CAUSAL
    )
    if case in {
        "noncausal-empty-query",
        "all-empty-query-cross",
        "deterministic-empty-query-cross",
    }:
        lengths_q = (
            [0] * spec.batch
            if case == "all-empty-query-cross"
            else MIXED_EMPTY_CROSS_QUERY_LENGTHS
        )
        q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
            spec, lengths_q=lengths_q
        )
    elif case == "empty-key-cross":
        lengths_k = RAGGED_CROSS_KEY_LENGTHS.copy()
        lengths_k[-1] = 0
        q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
            spec, lengths_k=lengths_k
        )
    else:
        q, k, v, cu_q, _ = make_ragged_self_packed(spec)
        cu_k = cu_q.clone()
    if case == "all-empty-self":
        q = q[:0].contiguous()
        k = k[:0].contiguous()
        v = v[:0].contiguous()
        cu_q = _cumulative([0] * spec.batch, q.device)
        cu_k = cu_q.clone()
    elif case == "deterministic-empty":
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec, lengths=MIXED_EMPTY_SELF_LENGTHS
        )
        cu_k = cu_q.clone()
    q.requires_grad_()

    def reject_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("out-of-scope ragged backward reached SDPA")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_sdpa)
    deterministic = case in {
        "deterministic",
        "deterministic-noncausal",
        "deterministic-empty",
        "deterministic-empty-query-cross",
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
            causal=spec.causal,
            deterministic=deterministic,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("dropout_p", 0.1, "dropout"),
        ("window_size", (1, 1), "sliding-window"),
        ("softcap", 1.0, "softcap"),
        ("alibi_slopes", "slopes", "ALiBi backward"),
        ("return_attn_probs", True, "grad-enabled"),
    ],
)
@pytest.mark.parametrize(
    "attention",
    ["self", "self-empty", "cross-empty-query"],
    ids=["self", "empty-self", "empty-query-cross"],
)
def test_ragged_causal_varlen_backward_rejects_incompatible_options(
    option: str,
    value: object,
    message: str,
    attention: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_ALIBI_CAUSAL
    if attention == "cross-empty-query":
        q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
            spec, lengths_q=MIXED_EMPTY_CROSS_QUERY_LENGTHS
        )
    else:
        requested_lengths = (
            MIXED_EMPTY_SELF_LENGTHS if attention == "self-empty" else None
        )
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec, lengths=requested_lengths
        )
        cu_k = cu_q
    q.requires_grad_()
    if option == "alibi_slopes":
        value = torch.ones(spec.nheads_q, device=q.device)
    kwargs = {option: value}

    def reject_sdpa(*args: object, **dispatch_kwargs: object) -> torch.Tensor:
        raise AssertionError("incompatible ragged backward reached SDPA")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_sdpa)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            shape=spec,
            **kwargs,
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
    ("attention", "entry_point"),
    [
        pytest.param("self", "unpacked", id="self-unpacked"),
        pytest.param("self", "qkv-packed", id="self-qkv-packed"),
        pytest.param("self", "kv-packed", id="self-kv-packed"),
        pytest.param("cross", "unpacked", id="cross-unpacked"),
        pytest.param("cross", "kv-packed", id="cross-kv-packed"),
    ],
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "alibi_layout",
    [None, "heads", "batch-heads"],
    ids=["slope-free", "head-slopes", "batch-head-slopes"],
)
def test_bert_base_varlen_ragged_attention_with_optional_alibi_matches_fp32_and_fa2(
    attention: str,
    entry_point: str,
    softmax_scale: float | None,
    alibi_layout: str | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = BERT_BASE_VARLEN_INFERENCE
    if attention == "self":
        q, k, v, cu_q, lengths_q = make_ragged_self_packed(
            spec,
            lengths=BERT_BASE_RAGGED_SELF_LENGTHS,
            seed=20260810,
        )
        cu_k = cu_q
        lengths_k = lengths_q
    else:
        q, k, v, cu_q, cu_k, lengths_q, lengths_k = (
            make_ragged_cross_packed(
                spec,
                lengths_q=BERT_BASE_RAGGED_SELF_LENGTHS,
                lengths_k=BERT_BASE_RAGGED_CROSS_KEY_LENGTHS,
                seed=20260810,
            )
        )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    if alibi_layout is None:
        slopes = None
    elif alibi_layout == "heads":
        slopes = head_slopes
    else:
        slopes = torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )

    with torch.no_grad():
        if entry_point == "unpacked":
            got = helion_attention.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=slopes,
                shape=spec,
            )
            expected_fa2 = flash_attn.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=slopes,
            )
        elif entry_point == "qkv-packed":
            qkv = torch.stack((q, k, v), dim=1)
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_q,
                spec.seqlen_q,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=slopes,
                shape=spec,
            )
            expected_fa2 = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_q,
                spec.seqlen_q,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=slopes,
            )
        else:
            kv = torch.stack((k, v), dim=1)
            got = helion_attention.flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=slopes,
                shape=spec,
            )
            expected_fa2 = flash_attn.flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=slopes,
            )
        expected_fp32 = reference_packed(
            q,
            k,
            v,
            lengths_q,
            lengths_k,
            causal=False,
            scale=scale,
            alibi_slopes=slopes,
        )

    assert not helion_attention.is_varlen_shape_supported(
        spec, dtype=spec.dtype, causal=False
    )
    if attention == "cross":
        assert cu_q.data_ptr() != cu_k.data_ptr()
        assert q.shape[0] != k.shape[0]
    if slopes is not None:
        assert slopes.shape == (
            (spec.batch, spec.nheads_q)
            if alibi_layout == "batch-heads"
            else (spec.nheads_q,)
        )
    torch.testing.assert_close(
        got.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )


@requires_cuda
@pytest.mark.parametrize(
    "attention", ["self", "cross"], ids=["ragged-self", "ragged-cross"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "alibi_layout",
    [None, "heads", "batch-heads"],
    ids=["slope-free", "head-slopes", "batch-head-slopes"],
)
def test_bert_base_varlen_return_attn_probs_matches_fa2_and_fp32(
    attention: str,
    softmax_scale: float | None,
    alibi_layout: str | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = BERT_BASE_VARLEN_INFERENCE
    if attention == "self":
        q, k, v, cu_q, lengths_q = make_ragged_self_packed(
            spec,
            lengths=BERT_BASE_RAGGED_SELF_LENGTHS,
            seed=20260811,
        )
        cu_k = cu_q
        lengths_k = lengths_q
    else:
        q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_ragged_cross_packed(
            spec,
            lengths_q=BERT_BASE_RAGGED_SELF_LENGTHS,
            lengths_k=BERT_BASE_RAGGED_CROSS_KEY_LENGTHS,
            seed=20260811,
        )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    if alibi_layout is None:
        slopes = None
    elif alibi_layout == "heads":
        slopes = head_slopes
    else:
        slopes = torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )

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
            causal=False,
            alibi_slopes=slopes,
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
            causal=False,
            alibi_slopes=slopes,
            return_attn_probs=True,
        )
        expected_fp32 = reference_packed(
            q,
            k,
            v,
            lengths_q,
            lengths_k,
            causal=False,
            scale=scale,
            alibi_slopes=slopes,
        )

    assert isinstance(got, tuple) and len(got) == 3
    assert isinstance(expected, tuple) and len(expected) == 3
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
    if slopes is not None:
        assert slopes.shape == (
            (spec.batch, spec.nheads_q)
            if alibi_layout == "batch-heads"
            else (spec.nheads_q,)
        )
    torch.testing.assert_close(
        out.float(), expected_out.float(), atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        out.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        softmax_lse, expected_lse, atol=2e-3, rtol=1e-5
    )


@requires_cuda
@pytest.mark.parametrize(
    "entry_point", ["qkv-packed", "kv-packed"], ids=["qkv-packed", "kv-packed"]
)
@pytest.mark.parametrize(
    "alibi_layout",
    [None, "heads", "batch-heads"],
    ids=["slope-free", "head-slopes", "batch-head-slopes"],
)
def test_bert_base_varlen_packed_adapters_inherit_diagnostic_return(
    entry_point: str,
    alibi_layout: str | None,
) -> None:
    spec = BERT_BASE_VARLEN_INFERENCE
    if entry_point == "qkv-packed":
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec,
            lengths=BERT_BASE_RAGGED_SELF_LENGTHS,
            seed=271828,
        )
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
            spec,
            lengths_q=BERT_BASE_RAGGED_SELF_LENGTHS,
            lengths_k=BERT_BASE_RAGGED_CROSS_KEY_LENGTHS,
            seed=271828,
        )
    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    if alibi_layout is None:
        slopes = None
    elif alibi_layout == "heads":
        slopes = head_slopes
    else:
        slopes = torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )

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
            causal=False,
            alibi_slopes=slopes,
            return_attn_probs=True,
            shape=spec,
        )
        if entry_point == "qkv-packed":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=0.37,
                causal=False,
                alibi_slopes=slopes,
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
                causal=False,
                alibi_slopes=slopes,
                return_attn_probs=True,
                shape=spec,
            )

    assert isinstance(got, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_cuda
@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        pytest.param("dropout_p", 0.1, "dropout", id="dropout"),
        pytest.param(
            "window_size", (7, 7), "sliding-window", id="window"
        ),
        pytest.param("softcap", 50.0, "softcap", id="softcap"),
    ],
)
@pytest.mark.parametrize("with_alibi", [False, True], ids=["plain", "alibi"])
def test_bert_base_varlen_rejects_optional_features_before_generic_dispatch(
    option: str,
    value: object,
    message: str,
    with_alibi: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = BERT_BASE_VARLEN_INFERENCE
    q = torch.zeros(
        spec.batch,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros_like(q)
    v = torch.zeros_like(k)
    cu_seqlens = torch.arange(spec.batch + 1, device=q.device, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported BERT-base option reached dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_dispatch
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    kwargs = {option: value}
    if with_alibi:
        kwargs["alibi_slopes"] = torch.ones(spec.nheads_q, device=q.device)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=False,
            shape=spec,
            **kwargs,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("gradient", "grad-enabled", id="gradient"),
        pytest.param("dropout", "dropout", id="dropout"),
        pytest.param("window", "sliding-window", id="window"),
        pytest.param("softcap", "softcap", id="softcap"),
        pytest.param(
            "deterministic", "deterministic=False", id="deterministic"
        ),
        pytest.param("paging", "block_table", id="paging"),
        pytest.param("causal-profile", "implemented only", id="causal-profile"),
        pytest.param("fp16-profile", "implemented only", id="fp16-profile"),
        pytest.param("other-maxima", "implemented only", id="other-maxima"),
    ],
)
def test_bert_base_varlen_diagnostics_reject_out_of_scope_calls_before_dispatch(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case == "causal-profile":
        spec = AttnShape(16, 512, 512, 12, 12, 64, torch.bfloat16, True)
    elif case == "fp16-profile":
        spec = AttnShape(16, 512, 512, 12, 12, 64, torch.float16, False)
    elif case == "other-maxima":
        spec = AttnShape(16, 511, 511, 12, 12, 64, torch.bfloat16, False)
    else:
        spec = BERT_BASE_VARLEN_INFERENCE

    q = torch.zeros(
        spec.batch,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros_like(q)
    v = torch.zeros_like(k)
    cu_seqlens = torch.arange(spec.batch + 1, device=q.device, dtype=torch.int32)
    kwargs: dict[str, object] = {
        "causal": spec.causal,
        "return_attn_probs": True,
        "shape": spec,
    }
    if case == "gradient":
        q.requires_grad_()
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "window":
        kwargs["window_size"] = (7, 7)
    elif case == "softcap":
        kwargs["softcap"] = 50.0
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "paging":
        kwargs["block_table"] = torch.zeros(
            spec.batch, 1, device=q.device, dtype=torch.int32
        )

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError("out-of-scope BERT-base diagnostic reached dispatch")

    for dispatch_name in (
        "_generic_varlen_diagnostic_forward",
        "_generic_varlen_forward",
        "_generic_varlen_alibi_forward",
        "_generic_paged_varlen_forward",
        "lookup_varlen",
        "lookup_paged",
    ):
        monkeypatch.setattr(helion_attention, dispatch_name, reject_dispatch)

    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    "entry_point",
    ["unpacked", "qkv-packed", "kv-packed"],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize("full_length", [False, True], ids=["ragged", "full"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_bert_base_varlen_no_grad_retains_generic_inference_dispatch(
    entry_point: str,
    full_length: bool,
    deterministic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = BERT_BASE_VARLEN_INFERENCE
    sequence_length = spec.seqlen_q if full_length else 1
    q = torch.empty(
        spec.batch * sequence_length,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.empty_like(q)
    v = torch.empty_like(k)
    cu_seqlens = torch.arange(
        0,
        (spec.batch + 1) * sequence_length,
        sequence_length,
        device=q.device,
        dtype=torch.int32,
    )
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generic(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("slope-free BERT-base call changed dispatch")

    monkeypatch.setattr(helion_attention, "_generic_varlen_forward", generic)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_dispatch
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "dense_attention_math_sdpa", reject_dispatch
    )

    with torch.no_grad():
        if entry_point == "unpacked":
            got = helion_attention.flash_attn_varlen_func(
                q.requires_grad_(),
                k.requires_grad_(),
                v.requires_grad_(),
                cu_seqlens,
                cu_seqlens,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=False,
                alibi_slopes=None,
                deterministic=deterministic,
                shape=spec,
            )
        elif entry_point == "qkv-packed":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1).requires_grad_(),
                cu_seqlens,
                spec.seqlen_q,
                causal=False,
                alibi_slopes=None,
                deterministic=deterministic,
                shape=spec,
            )
        else:
            got = helion_attention.flash_attn_varlen_kvpacked_func(
                q.requires_grad_(),
                torch.stack((k, v), dim=1).requires_grad_(),
                cu_seqlens,
                cu_seqlens,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=False,
                alibi_slopes=None,
                deterministic=deterministic,
                shape=spec,
            )

    assert got is sentinel
    assert len(calls) == 1


@requires_cuda
@pytest.mark.parametrize(
    "entry_point",
    ["unpacked", "qkv-packed", "kv-packed"],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    ("deterministic", "message"),
    [
        pytest.param(False, "ragged varlen backward", id="ordinary"),
        pytest.param(True, "deterministic=True", id="deterministic"),
    ],
)
def test_bert_base_varlen_rejects_ragged_backward_before_dispatch(
    entry_point: str,
    deterministic: bool,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = BERT_BASE_VARLEN_INFERENCE
    q = torch.zeros(
        spec.batch,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros_like(q)
    v = torch.zeros_like(k)
    cu_seqlens = torch.arange(spec.batch + 1, device=q.device, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("gradient-bearing BERT-base input reached dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "dense_attention_math_sdpa", reject_dispatch
    )
    with pytest.raises(NotImplementedError, match=message):
        if entry_point == "unpacked":
            helion_attention.flash_attn_varlen_func(
                q.requires_grad_(),
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=False,
                deterministic=deterministic,
                shape=spec,
            )
        elif entry_point == "qkv-packed":
            qkv = torch.stack((q, k, v), dim=1).requires_grad_()
            helion_attention.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                spec.seqlen_q,
                causal=False,
                deterministic=deterministic,
                shape=spec,
            )
        else:
            kv = torch.stack((k, v), dim=1).requires_grad_()
            helion_attention.flash_attn_varlen_kvpacked_func(
                q.requires_grad_(),
                kv,
                cu_seqlens,
                cu_seqlens,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=False,
                deterministic=deterministic,
                shape=spec,
            )


@requires_cuda
@pytest.mark.parametrize(
    ("with_alibi", "exception", "message"),
    [
        pytest.param(
            False,
            helion_attention.UnsupportedShapeError,
            "block_table currently supports",
            id="plain",
        ),
        pytest.param(True, NotImplementedError, "ALiBi", id="alibi"),
    ],
)
def test_bert_base_varlen_rejects_paging_before_dispatch(
    with_alibi: bool,
    exception: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = BERT_BASE_VARLEN_INFERENCE
    q = torch.zeros(
        spec.batch,
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros(
        1,
        16,
        spec.nheads_kv,
        spec.head_dim,
        device=q.device,
        dtype=spec.dtype,
    )
    v = torch.zeros_like(k)
    cu_seqlens = torch.arange(spec.batch + 1, device=q.device, dtype=torch.int32)
    block_table = torch.zeros(spec.batch, 32, device=q.device, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("BERT-base paging reached attention dispatch")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_dispatch
    )
    slopes = (
        torch.ones(spec.nheads_q, device=q.device) if with_alibi else None
    )
    with pytest.raises(exception, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=False,
            alibi_slopes=slopes,
            block_table=block_table,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            AttnShape(16, 511, 511, 12, 12, 64, torch.bfloat16, False),
            id="other-maxima",
        ),
        pytest.param(
            AttnShape(16, 512, 512, 12, 12, 64, torch.bfloat16, True),
            id="causal",
        ),
        pytest.param(
            AttnShape(16, 512, 512, 12, 12, 64, torch.float16, False),
            id="fp16",
        ),
    ],
)
def test_bert_base_varlen_neighboring_shapes_remain_unsupported(
    spec: AttnShape, monkeypatch: pytest.MonkeyPatch
) -> None:
    q = torch.zeros(1, spec.nheads_q, spec.head_dim, device="cuda", dtype=spec.dtype)
    k = torch.zeros_like(q)
    v = torch.zeros_like(k)
    cu_seqlens = torch.tensor(
        [0, 1, *([1] * (spec.batch - 1))],
        device=q.device,
        dtype=torch.int32,
    )

    def reject_generic(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("neighboring shape reached BERT-base dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_generic
    )
    with pytest.raises(helion_attention.UnsupportedShapeError):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=spec.causal,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("lengths", "softmax_scale"),
    [
        pytest.param([256, 0, 73, 0], None, id="default-scale"),
        pytest.param([0, 19, 256, 7], 0.37, id="custom-scale"),
    ],
)
def test_llama3_varlen_ragged_self_attention_matches_fp32_and_fa2(
    lengths: list[int], softmax_scale: float | None
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_seqlens, actual_lengths = make_ragged_self_packed(
        spec, lengths=lengths, seed=20260809
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        shape=spec,
    )
    expected_fp32 = reference_packed(
        q,
        k,
        v,
        actual_lengths,
        actual_lengths,
        causal=True,
        scale=scale,
    )
    expected_fa2 = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
    )

    assert not helion_attention.is_varlen_shape_supported(
        spec, dtype=spec.dtype, causal=True
    )
    torch.testing.assert_close(
        got.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )


@requires_cuda
@pytest.mark.parametrize("softmax_scale", [None, 0.29], ids=["default", "custom"])
def test_llama3_varlen_kvpacked_inherits_cross_attention_support(
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=0, seed=271828
    )
    kv = torch.stack((k, v), dim=1)
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    got = helion_attention.flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        shape=spec,
    )
    expected = flash_attn.flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
    )
    expected_fp32 = reference_packed(
        q,
        k,
        v,
        lengths_q,
        lengths_k,
        causal=True,
        scale=scale,
    )
    assert cu_q.data_ptr() != cu_k.data_ptr()
    assert q.shape[0] != k.shape[0]
    torch.testing.assert_close(
        got.float(), expected.float(), atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        got.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )


@requires_cuda
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_llama3_varlen_ragged_cross_attention_matches_fp32_and_fa2(
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=0, seed=20260809
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
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
        shape=spec,
    )
    expected_fp32 = reference_packed(
        q,
        k,
        v,
        lengths_q,
        lengths_k,
        causal=True,
        scale=scale,
    )
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
    )

    assert all(length > 0 for length in lengths_q + lengths_k)
    assert cu_q.data_ptr() != cu_k.data_ptr()
    assert q.shape[0] != k.shape[0]
    torch.testing.assert_close(
        got.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )


@requires_cuda
@pytest.mark.parametrize(
    "attention", ["self", "cross"], ids=["ragged-self", "ragged-cross"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_llama3_varlen_return_attn_probs_matches_fa2(
    attention: str,
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = LLAMA3_VARLEN_INFERENCE
    if attention == "self":
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec, lengths=[256, 0, 73, 11], seed=20260810
        )
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
            spec, variant=0, seed=20260810
        )
        assert all(length > 0 for length in lengths_q + lengths_k)
        assert cu_q.data_ptr() != cu_k.data_ptr()
        assert q.shape[0] != k.shape[0]

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
            causal=True,
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
            causal=True,
            return_attn_probs=True,
        )

    assert isinstance(got, tuple) and len(got) == 3
    assert isinstance(expected, tuple) and len(expected) == 3
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
        out.float(), expected_out.float(), atol=5e-2, rtol=2e-2
    )
    finite_lse = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(softmax_lse), finite_lse)
    torch.testing.assert_close(
        softmax_lse[finite_lse],
        expected_lse[finite_lse],
        atol=2e-3,
        rtol=1e-5,
    )
    assert torch.equal(softmax_lse[~finite_lse], expected_lse[~finite_lse])


@requires_cuda
@pytest.mark.parametrize(
    "attention", ["self", "cross"], ids=["ragged-self", "ragged-cross"]
)
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_llama3_varlen_alibi_diagnostics_match_fa2(
    attention: str,
    batched_slopes: bool,
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = LLAMA3_VARLEN_INFERENCE
    if attention == "self":
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec, lengths=[256, 0, 73, 11], seed=20260810
        )
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
            spec, variant=0, seed=20260810
        )
        assert all(length > 0 for length in lengths_q + lengths_k)
        assert cu_q.data_ptr() != cu_k.data_ptr()
        assert q.shape[0] != k.shape[0]

    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )
        if batched_slopes
        else head_slopes
    )

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
            causal=True,
            alibi_slopes=slopes,
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
            causal=True,
            alibi_slopes=slopes,
            return_attn_probs=True,
        )

    assert isinstance(got, tuple) and len(got) == 3
    assert isinstance(expected, tuple) and len(expected) == 3
    out, softmax_lse, s_dmask = got
    expected_out, expected_lse, expected_s_dmask = expected
    assert slopes.shape == (
        (spec.batch, spec.nheads_q) if batched_slopes else (spec.nheads_q,)
    )
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
        out.float(), expected_out.float(), atol=5e-2, rtol=2e-2
    )
    finite_lse = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(softmax_lse), finite_lse)
    torch.testing.assert_close(
        softmax_lse[finite_lse],
        expected_lse[finite_lse],
        atol=2e-3,
        rtol=1e-5,
    )
    assert torch.equal(softmax_lse[~finite_lse], expected_lse[~finite_lse])


@requires_cuda
def test_llama3_varlen_kvpacked_inherits_diagnostic_return() -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, cu_k, *_ = make_packed(
        spec, variant=1, seed=271828
    )
    kv = torch.stack((k, v), dim=1)

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
            causal=True,
            return_attn_probs=True,
            shape=spec,
        )
        got = helion_attention.flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=0.37,
            causal=True,
            return_attn_probs=True,
            shape=spec,
        )

    assert isinstance(got, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_cuda
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
def test_llama3_varlen_kvpacked_inherits_alibi_diagnostics(
    batched_slopes: bool,
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, cu_k, *_ = make_packed(
        spec, variant=1, seed=271828
    )
    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )
        if batched_slopes
        else head_slopes
    )
    kv = torch.stack((k, v), dim=1)

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
            causal=True,
            alibi_slopes=slopes,
            return_attn_probs=True,
            shape=spec,
        )
        got = helion_attention.flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=0.37,
            causal=True,
            alibi_slopes=slopes,
            return_attn_probs=True,
            shape=spec,
        )

    assert isinstance(got, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_cuda
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_llama3_varlen_alibi_matches_fp32_and_fa2(
    batched_slopes: bool,
    softmax_scale: float | None,
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, lengths_q = make_ragged_self_packed(
        spec, lengths=[256, 0, 73, 0], seed=20260809
    )
    cu_k = cu_q
    lengths_k = lengths_q

    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )
        if batched_slopes
        else head_slopes
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

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
            causal=True,
            alibi_slopes=slopes,
            shape=spec,
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

    assert slopes.shape == (
        (spec.batch, spec.nheads_q) if batched_slopes else (spec.nheads_q,)
    )
    assert got.shape == q.shape
    assert got.dtype == q.dtype
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

    try:
        import flash_attn
    except ImportError:
        return
    with torch.no_grad():
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
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )


@requires_cuda
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.29], ids=["default-scale", "custom-scale"]
)
def test_llama3_varlen_kvpacked_inherits_alibi_support(
    batched_slopes: bool, softmax_scale: float | None
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, cu_k, *_ = make_packed(
        spec, variant=0, seed=271828
    )
    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )
        if batched_slopes
        else head_slopes
    )
    kv = torch.stack((k, v), dim=1)

    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            alibi_slopes=slopes,
            shape=spec,
        )
        expected_unpacked = helion_attention.flash_attn_varlen_func(
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

    torch.testing.assert_close(got, expected_unpacked)


@requires_cuda
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(BERT_BASE_VARLEN_INFERENCE, id="bert-base"),
        pytest.param(LLAMA3_VARLEN_INFERENCE, id="llama3-gqa"),
        pytest.param(VARLEN_DIAGNOSTIC, id="shipped-causal"),
    ],
)
@pytest.mark.parametrize("features", ["alibi", "diagnostics", "combined"])
def test_varlen_feature_dispatch_stays_separate(
    spec: AttnShape, features: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    slopes = torch.ones(spec.nheads_q, device=q.device)
    out = torch.empty_like(q)
    diagnostics = (out, torch.empty(1, device=q.device), q.new_empty((0,)))
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def alibi_forward(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(("alibi", args, kwargs))
        return out

    def diagnostic_forward(
        *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(("diagnostics", args, kwargs))
        return diagnostics

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("feature call reached unrelated varlen dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", alibi_forward
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", diagnostic_forward
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)

    kwargs: dict[str, object] = {}
    if features in ("alibi", "combined"):
        kwargs["alibi_slopes"] = slopes
    if features in ("diagnostics", "combined"):
        kwargs["return_attn_probs"] = True
    with torch.no_grad():
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
            **kwargs,
        )

    expected_dispatch = "alibi" if features == "alibi" else "diagnostics"
    assert got is (out if features == "alibi" else diagnostics)
    assert len(calls) == 1
    dispatch, args, dispatch_kwargs = calls[0]
    assert dispatch == expected_dispatch
    if features == "alibi":
        assert len(args) == 8 and args[-1] is slopes
        assert dispatch_kwargs == {}
    else:
        assert len(args) == 7
        assert dispatch_kwargs == (
            {"alibi_slopes": slopes} if features == "combined" else {}
        )


@requires_cuda
@pytest.mark.parametrize(
    "entry_point", ["unpacked", "kvpacked"], ids=["unpacked", "kv-packed"]
)
def test_llama3_varlen_without_alibi_retains_existing_dispatch(
    entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generic(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("slope-free Llama-3 call changed dispatch")

    monkeypatch.setattr(helion_attention, "_generic_varlen_forward", generic)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_dispatch
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)

    with torch.no_grad():
        if entry_point == "unpacked":
            got = helion_attention.flash_attn_varlen_func(
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
        else:
            got = helion_attention.flash_attn_varlen_kvpacked_func(
                q,
                torch.stack((k, v), dim=1),
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=True,
                alibi_slopes=None,
                shape=spec,
            )

    assert got is sentinel
    assert len(calls) == 1


@requires_cuda
def test_llama3_varlen_rejects_gradients_before_generic_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_seqlens, _ = make_ragged_self_packed(
        spec, lengths=[256, 0, 37, 11]
    )
    q.requires_grad_()

    def reject_generic(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("gradient-bearing input reached generic dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_generic
    )
    with pytest.raises(NotImplementedError, match="varlen backward"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        pytest.param("dropout_p", 0.1, "dropout", id="dropout"),
        pytest.param("window_size", (7, 7), "sliding-window", id="window"),
        pytest.param("softcap", 50.0, "softcap", id="softcap"),
        pytest.param(
            "deterministic", True, "deterministic=True", id="deterministic"
        ),
    ],
)
@pytest.mark.parametrize("with_alibi", [False, True], ids=["plain", "alibi"])
def test_llama3_varlen_rejects_optional_features_before_generic_dispatch(
    option: str,
    value: object,
    message: str,
    with_alibi: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_seqlens, _ = make_ragged_self_packed(
        spec, lengths=[256, 0, 37, 11]
    )

    def reject_generic(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("unsupported option reached generic dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_generic
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_generic
    )
    kwargs = {option: value}
    if with_alibi:
        kwargs["alibi_slopes"] = torch.ones(spec.nheads_q, device=q.device)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            shape=spec,
            **kwargs,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("gradient", "grad-enabled", id="gradient"),
        pytest.param("paging", "block_table", id="paging"),
        pytest.param("deterministic", "deterministic=True", id="deterministic"),
        pytest.param("dropout", "dropout", id="dropout"),
        pytest.param("window", "sliding-window", id="window"),
        pytest.param("softcap", "softcap", id="softcap"),
    ],
)
def test_llama3_varlen_diagnostics_reject_incompatible_features_before_dispatch(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    kwargs: dict[str, object] = {
        "causal": True,
        "return_attn_probs": True,
        "shape": spec,
    }
    if case == "gradient":
        q.requires_grad_()
    elif case == "paging":
        kwargs["block_table"] = torch.zeros(
            spec.batch, 1, device=q.device, dtype=torch.int32
        )
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "window":
        kwargs["window_size"] = (7, 7)
    else:
        kwargs["softcap"] = 50.0

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError("incompatible diagnostic call reached dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_dispatch
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_dispatch
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("q-gradient", "ALiBi backward", id="q-gradient"),
        pytest.param("slope-gradient", "ALiBi backward", id="slope-gradient"),
        pytest.param("paging", "block_table", id="paging"),
        pytest.param("deterministic", "deterministic", id="deterministic"),
        pytest.param("dropout", "dropout", id="dropout"),
        pytest.param("window", "sliding-window", id="window"),
        pytest.param("softcap", "softcap", id="softcap"),
    ],
)
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(BERT_BASE_VARLEN_INFERENCE, id="bert-base"),
        pytest.param(LLAMA3_VARLEN_INFERENCE, id="llama3-gqa"),
        pytest.param(VARLEN_DIAGNOSTIC, id="shipped-causal"),
    ],
)
def test_varlen_alibi_diagnostics_reject_incompatible_features(
    spec: AttnShape,
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    slopes = torch.ones(spec.nheads_q, device=q.device, dtype=torch.float32)
    kwargs: dict[str, object] = {
        "causal": spec.causal,
        "alibi_slopes": slopes,
        "return_attn_probs": True,
        "shape": spec,
    }
    if case == "q-gradient":
        q.requires_grad_()
    elif case == "slope-gradient":
        slopes.requires_grad_()
    elif case == "paging":
        kwargs["block_table"] = torch.zeros(
            spec.batch, 1, device=q.device, dtype=torch.int32
        )
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "window":
        kwargs["window_size"] = (7, 7)
    else:
        kwargs["softcap"] = 50.0

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError("incompatible ALiBi diagnostic call reached dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_dispatch
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    ("with_alibi", "exception", "message"),
    [
        pytest.param(
            False,
            helion_attention.UnsupportedShapeError,
            "supports only",
            id="plain",
        ),
        pytest.param(True, NotImplementedError, "ALiBi slopes", id="alibi"),
    ],
)
def test_llama3_varlen_rejects_paging_before_dispatch(
    with_alibi: bool,
    exception: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = LLAMA3_VARLEN_INFERENCE
    q = torch.zeros(4, 32, 128, device="cuda", dtype=spec.dtype)
    k = torch.zeros(1, 256, 8, 128, device="cuda", dtype=spec.dtype)
    v = torch.zeros_like(k)
    cu_seqlens = torch.arange(5, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(4, 1, device="cuda", dtype=torch.int32)

    def reject_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported profile reached paged dispatch")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_lookup)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_alibi_forward", reject_lookup
    )
    slopes = (
        torch.ones(spec.nheads_q, device=q.device) if with_alibi else None
    )
    with pytest.raises(exception, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            alibi_slopes=slopes,
            block_table=block_table,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            AttnShape(4, 255, 255, 32, 8, 128, torch.bfloat16, True),
            id="other-maxima",
        ),
        pytest.param(
            AttnShape(4, 256, 256, 32, 8, 128, torch.bfloat16, False),
            id="noncausal",
        ),
        pytest.param(
            AttnShape(4, 256, 256, 32, 8, 128, torch.float16, True),
            id="fp16",
        ),
    ],
)
def test_llama3_varlen_neighboring_shapes_remain_unsupported(
    spec: AttnShape, monkeypatch: pytest.MonkeyPatch
) -> None:
    q = torch.zeros(1, spec.nheads_q, spec.head_dim, device="cuda", dtype=spec.dtype)
    k = torch.zeros(1, spec.nheads_kv, spec.head_dim, device="cuda", dtype=spec.dtype)
    v = torch.zeros_like(k)
    cu_seqlens = torch.tensor([0, 1, 1, 1, 1], device="cuda", dtype=torch.int32)

    def reject_generic(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("neighboring shape reached Llama-3 dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_forward", reject_generic
    )
    with pytest.raises(helion_attention.UnsupportedShapeError):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=spec.causal,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    "entry_point",
    ["unpacked", "qkvpacked", "kvpacked"],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_full_causal_varlen_left_window_adapters_match_fp32_and_fa2(
    entry_point: str, softmax_scale: float | None
) -> None:
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=2, seed=20260810
    )
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    with torch.no_grad():
        if entry_point == "unpacked":
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
                window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
                shape=spec,
            )
        elif entry_point == "qkvpacked":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
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
                softmax_scale=softmax_scale,
                causal=True,
                window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
                shape=spec,
            )
        expected_fp32 = reference_packed(
            q,
            k,
            v,
            lengths_q,
            lengths_k,
            causal=True,
            scale=scale,
            window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
        )

    assert got.shape == q.shape
    assert got.dtype == torch.bfloat16
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

    try:
        import flash_attn
    except ImportError:
        return
    with torch.no_grad():
        if entry_point == "unpacked":
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
                window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
            )
        elif entry_point == "qkvpacked":
            expected_fa2 = flash_attn.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
            )
        else:
            expected_fa2 = flash_attn.flash_attn_varlen_kvpacked_func(
                q,
                torch.stack((k, v), dim=1),
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
            )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=2e-2, rtol=1e-2
    )


@requires_cuda
@pytest.mark.parametrize(
    "entry_point",
    ["unpacked", "qkvpacked", "kvpacked"],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_full_causal_varlen_left_window_backward_matches_fp32_and_fa2(
    entry_point: str, softmax_scale: float | None
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    base_q, base_k, base_v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=2, seed=20260810
    )
    generator = torch.Generator(device=base_q.device).manual_seed(20260811)
    grad_out = torch.randn(
        base_q.shape,
        device=base_q.device,
        dtype=base_q.dtype,
        generator=generator,
    )

    def run(
        package: object,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        kwargs: dict[str, object] = {
            "softmax_scale": softmax_scale,
            "causal": True,
            "window_size": VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
        }
        if package is helion_attention:
            kwargs["shape"] = spec

        if entry_point == "unpacked":
            q = base_q.detach().requires_grad_()
            k = base_k.detach().requires_grad_()
            v = base_v.detach().requires_grad_()
            out = package.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                **kwargs,
            )
            grads = torch.autograd.grad(out, (q, k, v), grad_out)
        elif entry_point == "qkvpacked":
            qkv = torch.stack((base_q, base_k, base_v), dim=1).requires_grad_()
            out = package.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_q,
                spec.seqlen_q,
                **kwargs,
            )
            packed_grad = torch.autograd.grad(out, qkv, grad_out)[0]
            grads = tuple(packed_grad[:, index] for index in range(3))
        else:
            q = base_q.detach().requires_grad_()
            kv = torch.stack((base_k, base_v), dim=1).requires_grad_()
            out = package.flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                **kwargs,
            )
            q_grad, packed_grad = torch.autograd.grad(out, (q, kv), grad_out)
            grads = (q_grad, packed_grad[:, 0], packed_grad[:, 1])

        assert isinstance(out, torch.Tensor)
        return out, grads

    got, got_grads = run(helion_attention)
    expected_fa2, expected_fa2_grads = run(flash_attn)

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    q_ref = base_q.float().requires_grad_()
    k_ref = base_k.float().requires_grad_()
    v_ref = base_v.float().requires_grad_()
    expected_fp32 = reference_packed(
        q_ref,
        k_ref,
        v_ref,
        lengths_q,
        lengths_k,
        causal=True,
        scale=scale,
        window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
    )
    expected_fp32_grads = torch.autograd.grad(
        expected_fp32, (q_ref, k_ref, v_ref), grad_out.float()
    )

    assert got.shape == base_q.shape
    assert got.dtype == torch.bfloat16
    assert got.is_contiguous()
    torch.testing.assert_close(
        got.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )
    gradient_atol = 8e-2 if softmax_scale is None else 1.5e-1
    for actual, reference, reference_fa2 in zip(
        got_grads, expected_fp32_grads, expected_fa2_grads
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=gradient_atol, rtol=5e-2
        )
        torch.testing.assert_close(
            actual.float(),
            reference_fa2.float(),
            atol=gradient_atol,
            rtol=5e-2,
        )


@requires_cuda
@pytest.mark.parametrize(
    "entry_point",
    ["unpacked", "qkvpacked", "kvpacked"],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
def test_causal_varlen_global_window_adapters_retain_generated_dispatch(
    entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2, seed=161803)
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generated(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_generic(*args: object, **kwargs: object) -> object:
        raise AssertionError("global-window call reached generic dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", lambda _spec: generated)
    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_causal_left_window_forward",
        reject_generic,
    )
    if entry_point == "unpacked":
        out = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            shape=spec,
        )
    elif entry_point == "qkvpacked":
        out = helion_attention.flash_attn_varlen_qkvpacked_func(
            torch.stack((q, k, v), dim=1),
            cu_q,
            spec.seqlen_q,
            causal=True,
            shape=spec,
        )
    else:
        out = helion_attention.flash_attn_varlen_kvpacked_func(
            q,
            torch.stack((k, v), dim=1),
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            shape=spec,
        )

    assert out is sentinel
    assert len(calls) == 1
    assert len(calls[0]) == 9


@requires_cuda
def test_full_causal_varlen_left_window_uses_fixed_generic_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2)
    sentinel = torch.empty_like(q)
    seen: list[tuple[float, AttnShape]] = []

    def generic(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        cu_q_arg: torch.Tensor,
        cu_k_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
    ) -> torch.Tensor:
        assert q_arg is q
        assert k_arg is k
        assert v_arg is v
        assert cu_q_arg is cu_q
        assert cu_k_arg is cu_k
        seen.append((scale_arg, spec_arg))
        return sentinel

    def reject_generated(*args: object, **kwargs: object) -> object:
        raise AssertionError("causal left window reached generated dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_causal_left_window_forward", generic
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_generated)
    out = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=0.29,
        causal=True,
        window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
        shape=spec,
    )

    assert out is sentinel
    assert seen == [(0.29, spec)]


@requires_cuda
@pytest.mark.parametrize(
    "offset_case",
    ["short-total", "noncanonical-full-total"],
    ids=["short-total", "noncanonical-full-total"],
)
def test_causal_varlen_left_window_rejects_ragged_offsets_before_dispatch(
    offset_case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    variant = 0 if offset_case == "short-total" else 2
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=variant)
    if offset_case == "noncanonical-full-total":
        cu_q = cu_q.clone()
        cu_k = cu_k.clone()
        cu_q[1] -= 1
        cu_k[3] -= 1

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("ragged causal window reached dispatch")

    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_causal_left_window_forward",
        reject_dispatch,
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    with pytest.raises(NotImplementedError, match="full-length|ragged offsets"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("window_size", "expected_kwargs"),
    [
        pytest.param(
            VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
            {"causal_window_left": VARLEN_CAUSAL_LEFT_WINDOW_SIZE[0]},
            id="finite-window",
        ),
        pytest.param((-1, -1), {}, id="global-window"),
    ],
)
def test_causal_varlen_backward_uses_narrow_dense_sdpa_dispatch(
    window_size: tuple[int, int],
    expected_kwargs: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2)
    q.requires_grad_()
    sentinel = torch.empty_like(q)
    seen: list[dict[str, object]] = []

    def sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        q_arg = args[0]
        assert isinstance(q_arg, torch.Tensor)
        assert args[4] == spec
        seen.append(kwargs)
        return sentinel.view_as(q_arg)

    def reject_forward_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("causal backward reached forward dispatch")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", sdpa)
    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_causal_left_window_forward",
        reject_forward_dispatch,
    )
    monkeypatch.setattr(
        helion_attention, "lookup_varlen", reject_forward_dispatch
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
        window_size=window_size,
        shape=spec,
    )

    assert out.data_ptr() == sentinel.data_ptr()
    assert seen == [expected_kwargs]


@pytest.mark.parametrize(
    "window_size",
    [(126, 0), (128, 0), (127, 1), (31, 31), (-1, 0)],
    ids=["smaller-left", "larger-left", "positive-right", "symmetric", "open-left"],
)
def test_causal_varlen_left_window_rejects_every_other_window(
    window_size: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    q = torch.zeros(1, spec.nheads_q, spec.head_dim, dtype=spec.dtype)
    cu_seqlens = torch.zeros(spec.batch + 1, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported causal window reached dispatch")

    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_causal_left_window_forward",
        reject_dispatch,
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    with pytest.raises(NotImplementedError, match="implemented only"):
        helion_attention.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            window_size=window_size,
            shape=spec,
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("dropout_p", 0.1, "dropout"),
        ("return_attn_probs", True, "return_attn_probs"),
        ("alibi_slopes", torch.ones(16), "ALiBi"),
        ("softcap", VARLEN_SOFTCAP_VALUE, "softcap"),
        ("deterministic", True, "deterministic"),
        ("block_table", torch.zeros(1, 1, dtype=torch.int32), "block_table"),
    ],
    ids=["dropout", "diagnostics", "alibi", "softcap", "determinism", "paging"],
)
def test_causal_varlen_left_window_rejects_optional_features_before_dispatch(
    option: str,
    value: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_CAUSAL_LEFT_WINDOW
    q = torch.zeros(1, spec.nheads_q, spec.head_dim, dtype=spec.dtype)
    cu_seqlens = torch.zeros(spec.batch + 1, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("optional causal-window feature reached dispatch")

    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_causal_left_window_forward",
        reject_dispatch,
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    kwargs = {
        option: value,
        "causal": True,
        "window_size": VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
        "shape": spec,
    }
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(VARLEN_ALIBI_NONCAUSAL, id="noncausal"),
        pytest.param(
            AttnShape(8, 512, 512, 16, 16, 64, torch.float16, True),
            id="fp16",
        ),
        pytest.param(
            AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, True),
            id="other-maxima",
        ),
    ],
)
def test_causal_varlen_left_window_rejects_other_profiles_before_dispatch(
    spec: AttnShape, monkeypatch: pytest.MonkeyPatch
) -> None:
    q = torch.zeros(1, spec.nheads_q, spec.head_dim, dtype=spec.dtype)
    cu_seqlens = torch.zeros(spec.batch + 1, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported causal-window profile reached dispatch")

    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_causal_left_window_forward",
        reject_dispatch,
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    with pytest.raises(NotImplementedError, match="implemented only"):
        helion_attention.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=spec.causal,
            window_size=VARLEN_CAUSAL_LEFT_WINDOW_SIZE,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("radius", "softmax_scale"),
    [
        pytest.param(31, None, id="default-scale"),
        pytest.param(127, 0.37, id="custom-scale"),
    ],
)
def test_full_varlen_symmetric_window_backward_matches_fp32_and_fa2(
    radius: int, softmax_scale: float | None
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=2, seed=20260809
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    generator = torch.Generator(device=q.device).manual_seed(20260810)
    grad_out = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=generator,
    )
    window_size = (radius, radius)

    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        window_size=window_size,
        shape=spec,
    )
    got_grads = torch.autograd.grad(got, (q, k, v), grad_out)

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    expected = reference_packed(
        q_ref,
        k_ref,
        v_ref,
        lengths_q,
        lengths_k,
        causal=False,
        scale=scale,
        window_size=window_size,
    )
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out.float()
    )

    q_fa2 = q.detach().requires_grad_()
    k_fa2 = k.detach().requires_grad_()
    v_fa2 = v.detach().requires_grad_()
    expected_fa2 = flash_attn.flash_attn_varlen_func(
        q_fa2,
        k_fa2,
        v_fa2,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        window_size=window_size,
    )
    expected_fa2_grads = torch.autograd.grad(
        expected_fa2, (q_fa2, k_fa2, v_fa2), grad_out
    )

    assert got.shape == q.shape
    assert got.dtype == q.dtype
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=5e-2, rtol=2e-2
    )
    for actual, reference, reference_fa2 in zip(
        got_grads, expected_grads, expected_fa2_grads
    ):
        torch.testing.assert_close(
            actual.float(), reference, atol=8e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(), reference_fa2.float(), atol=8e-2, rtol=2e-2
        )


@requires_cuda
def test_full_varlen_qkvpacked_inherits_symmetric_window_backward() -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_q, _, *_ = make_packed(spec, variant=2, seed=314159)
    window_size = (63, 63)
    grad_generator = torch.Generator(device=q.device).manual_seed(271828)
    grad_out = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=grad_generator,
    )

    qkv = torch.stack((q, k, v), dim=1).requires_grad_()
    got = helion_attention.flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_q,
        spec.seqlen_q,
        softmax_scale=0.29,
        window_size=window_size,
        shape=spec,
    )
    got_grad = torch.autograd.grad(got, qkv, grad_out)[0]

    q_ref = q.detach().requires_grad_()
    k_ref = k.detach().requires_grad_()
    v_ref = v.detach().requires_grad_()
    expected = helion_attention.flash_attn_varlen_func(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_q,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=0.29,
        window_size=window_size,
        shape=spec,
    )
    expected_grads = torch.autograd.grad(
        expected, (q_ref, k_ref, v_ref), grad_out
    )

    torch.testing.assert_close(got, expected)
    for index, expected_grad in enumerate(expected_grads):
        torch.testing.assert_close(got_grad[:, index], expected_grad)


@requires_cuda
@pytest.mark.parametrize(
    ("radius", "softmax_scale"),
    [
        pytest.param(0, None, id="radius-0-default-scale"),
        pytest.param(31, 0.37, id="radius-31-custom-scale"),
        pytest.param(127, None, id="radius-127-default-scale"),
        pytest.param(511, 0.19, id="radius-511-custom-scale"),
    ],
)
def test_noncausal_varlen_symmetric_window_matches_fa2_and_fp32_for_ragged_calls(
    radius: int, softmax_scale: float | None
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_seqlens, lengths = make_ragged_self_packed(
        spec, seed=20260809
    )
    window_size = (radius, radius)

    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=window_size,
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
        lengths,
        lengths,
        causal=False,
        scale=scale,
        window_size=window_size,
    )

    assert got.shape == q.shape
    assert got.dtype == torch.bfloat16
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

    try:
        import flash_attn
    except ImportError:
        return
    expected_fa2 = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=window_size,
    )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=1e-2, rtol=1e-2
    )


@requires_cuda
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.29], ids=["default-scale", "custom-scale"]
)
def test_varlen_qkvpacked_inherits_symmetric_window_support(
    softmax_scale: float | None,
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_seqlens, _ = make_ragged_self_packed(spec, seed=314159)
    window_size = (63, 63)

    expected = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=softmax_scale,
        window_size=window_size,
        shape=spec,
    )
    got = helion_attention.flash_attn_varlen_qkvpacked_func(
        torch.stack((q, k, v), dim=1),
        cu_seqlens,
        spec.seqlen_q,
        softmax_scale=softmax_scale,
        window_size=window_size,
        shape=spec,
    )

    torch.testing.assert_close(got, expected)


@requires_cuda
@pytest.mark.parametrize(
    ("window_size", "expected_radius"),
    [
        pytest.param((47, 47), 47, id="finite-window"),
        pytest.param((-1, -1), None, id="global-window"),
    ],
)
def test_full_varlen_window_backward_uses_existing_dense_sdpa_dispatch(
    window_size: tuple[int, int],
    expected_radius: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_q, _, *_ = make_packed(spec, variant=2)
    q.requires_grad_()
    sentinel = torch.empty_like(q)
    seen: list[dict[str, object]] = []

    def sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        q_arg = args[0]
        assert isinstance(q_arg, torch.Tensor)
        assert args[4] == spec
        seen.append(kwargs)
        return sentinel.view_as(q_arg)

    def reject_other_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("full-length backward reached forward dispatch")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", sdpa)
    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_symmetric_window_forward",
        reject_other_dispatch,
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_other_dispatch)
    out = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_q,
        spec.seqlen_q,
        spec.seqlen_k,
        window_size=window_size,
        shape=spec,
    )

    assert out.data_ptr() == sentinel.data_ptr()
    expected_kwargs = (
        {"symmetric_window_radius": expected_radius}
        if expected_radius is not None
        else {}
    )
    assert seen == [expected_kwargs]


@requires_cuda
def test_varlen_symmetric_window_bypasses_generated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_seqlens, _ = make_ragged_self_packed(spec)
    sentinel = torch.empty_like(q)
    seen: list[tuple[float, AttnShape, tuple[int, int]]] = []

    def reject_generated(*args: object, **kwargs: object) -> object:
        raise AssertionError("symmetric-window call reached generated dispatch")

    def generic(
        q_arg: torch.Tensor,
        k_arg: torch.Tensor,
        v_arg: torch.Tensor,
        cu_q_arg: torch.Tensor,
        cu_k_arg: torch.Tensor,
        scale_arg: float,
        spec_arg: AttnShape,
        window_arg: tuple[int, int],
    ) -> torch.Tensor:
        del q_arg, k_arg, v_arg, cu_q_arg, cu_k_arg
        seen.append((scale_arg, spec_arg, window_arg))
        return sentinel

    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_generated)
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_symmetric_window_forward", generic
    )
    out = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens.clone(),
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=0.37,
        window_size=(47, 47),
        shape=spec,
    )

    assert out is sentinel
    assert seen == [(0.37, spec, (47, 47))]


@requires_cuda
@pytest.mark.parametrize("entry_point", ["unpacked", "qkvpacked"])
def test_varlen_global_window_retains_generated_dispatch(
    entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_seqlens, _ = make_ragged_self_packed(spec)
    sentinel = torch.empty_like(q)
    calls: list[tuple[object, ...]] = []

    def generated(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    def reject_generic(*args: object, **kwargs: object) -> object:
        raise AssertionError("global-window call reached generic dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", lambda _spec: generated)
    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_symmetric_window_forward",
        reject_generic,
    )
    monkeypatch.setattr(
        helion_attention, "_validate_varlen_self_attention_offsets", reject_generic
    )
    if entry_point == "unpacked":
        out = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            window_size=(-1, -1),
            shape=spec,
        )
    else:
        out = helion_attention.flash_attn_varlen_qkvpacked_func(
            torch.stack((q, k, v), dim=1),
            cu_seqlens,
            spec.seqlen_q,
            window_size=(-1, -1),
            shape=spec,
        )

    assert out is sentinel
    assert len(calls) == 1
    assert len(calls[0]) == 9


@requires_cuda
@pytest.mark.parametrize(
    "requires_grad", [False, True], ids=["forward", "backward"]
)
def test_varlen_symmetric_window_rejects_unequal_query_key_offsets(
    requires_grad: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_q, _ = make_ragged_self_packed(spec)
    if requires_grad:
        q.requires_grad_()
    cu_k = cu_q.clone()
    cu_k[1] -= 1

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("cross-attention offsets reached window dispatch")

    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_symmetric_window_forward",
        reject_dispatch,
    )
    with pytest.raises(NotImplementedError, match="identical cu_seqlens"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            window_size=(31, 31),
            shape=spec,
        )


@pytest.mark.parametrize(
    ("window_size", "error"),
    [
        (None, TypeError),
        (1, TypeError),
        ((1,), ValueError),
        ((1, 1, 1), ValueError),
        ((True, True), TypeError),
        ((1.5, 1.5), TypeError),
        ((-2, -2), ValueError),
        ((2**31, 2**31), ValueError),
    ],
    ids=[
        "none",
        "scalar",
        "short",
        "long",
        "bool-bounds",
        "float-bounds",
        "below-sentinel",
        "int32-overflow",
    ],
)
def test_varlen_symmetric_window_rejects_malformed_bounds(
    window_size: object, error: type[Exception]
) -> None:
    q = torch.zeros(1, 16, 64, dtype=torch.bfloat16)
    cu_seqlens = torch.zeros(9, dtype=torch.int32)
    with pytest.raises(error, match="window_size"):
        helion_attention.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            VARLEN_SYMMETRIC_WINDOW.seqlen_q,
            VARLEN_SYMMETRIC_WINDOW.seqlen_k,
            window_size=window_size,  # type: ignore[arg-type]
            shape=VARLEN_SYMMETRIC_WINDOW,
        )


@pytest.mark.parametrize(
    "window_size",
    [(-1, 0), (31, -1), (31, 0), (0, 31), (31, 30), (-1, 31)],
    ids=[
        "unbounded-left",
        "unbounded-right",
        "left-only",
        "right-only",
        "unequal-finite",
        "right-with-unbounded-left",
    ],
)
def test_varlen_symmetric_window_rejects_asymmetric_forms(
    window_size: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    q = torch.zeros(1, 16, 64, dtype=torch.bfloat16)
    cu_seqlens = torch.zeros(9, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("asymmetric window reached dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_symmetric_window_forward",
        reject_dispatch,
    )
    with pytest.raises(NotImplementedError, match="radius, radius"):
        helion_attention.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            VARLEN_SYMMETRIC_WINDOW.seqlen_q,
            VARLEN_SYMMETRIC_WINDOW.seqlen_k,
            window_size=window_size,
            shape=VARLEN_SYMMETRIC_WINDOW,
        )


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(VARLEN_ALIBI_CAUSAL, id="causal"),
        pytest.param(
            AttnShape(8, 512, 512, 16, 16, 64, torch.float16, False),
            id="fp16",
        ),
        pytest.param(
            AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, False),
            id="other-maxima",
        ),
    ],
)
def test_varlen_symmetric_window_rejects_other_profiles(
    spec: AttnShape, monkeypatch: pytest.MonkeyPatch
) -> None:
    q = torch.zeros(1, spec.nheads_q, spec.head_dim, dtype=spec.dtype)
    cu_seqlens = torch.zeros(spec.batch + 1, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported window profile reached dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_symmetric_window_forward",
        reject_dispatch,
    )
    with pytest.raises(NotImplementedError, match="implemented only"):
        helion_attention.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=spec.causal,
            window_size=(31, 31),
            shape=spec,
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("dropout_p", 0.1, "dropout"),
        ("softcap", 1.0, "softcap"),
        ("alibi_slopes", torch.ones(16), "ALiBi"),
        ("return_attn_probs", True, "return_attn_probs"),
        ("block_table", torch.zeros(1), "block_table"),
        ("deterministic", True, "deterministic=True"),
    ],
)
def test_varlen_symmetric_window_rejects_incompatible_options_before_dispatch(
    option: str,
    value: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q = torch.zeros(1, spec.nheads_q, spec.head_dim, dtype=spec.dtype)
    cu_seqlens = torch.zeros(spec.batch + 1, dtype=torch.int32)

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("incompatible window call reached dispatch")

    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(
        helion_attention,
        "_generic_varlen_symmetric_window_forward",
        reject_dispatch,
    )
    kwargs = {
        option: value,
        "window_size": (31, 31),
        "shape": spec,
    }
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
def test_ragged_varlen_symmetric_window_rejects_backward_but_allows_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_SYMMETRIC_WINDOW
    q, k, v, cu_seqlens, _ = make_ragged_self_packed(spec)
    q.requires_grad_()
    sentinel = torch.empty_like(q)

    def generic(*args: object, **kwargs: object) -> torch.Tensor:
        return sentinel

    def reject_generated(*args: object, **kwargs: object) -> object:
        raise AssertionError("symmetric-window call reached generated dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_symmetric_window_forward", generic
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_generated)
    with pytest.raises(NotImplementedError, match="sliding-window backward"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            window_size=(31, 31),
            shape=spec,
        )

    with torch.no_grad():
        out = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            window_size=(31, 31),
            shape=spec,
        )
    assert out is sentinel


@requires_cuda
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_causal_varlen_return_attn_probs_matches_fa2_for_ragged_masked_rows(
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_DIAGNOSTIC
    q, k, v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=0, seed=20260808
    )

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
            causal=True,
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
            causal=True,
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

    finite_lse = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(softmax_lse), finite_lse)
    torch.testing.assert_close(
        softmax_lse[finite_lse],
        expected_lse[finite_lse],
        atol=2e-3,
        rtol=1e-5,
    )
    assert torch.equal(softmax_lse[~finite_lse], expected_lse[~finite_lse])

    fully_masked = lengths_q[0] - lengths_k[0]
    assert fully_masked > 0
    assert torch.isposinf(softmax_lse[:, :fully_masked]).all()
    torch.testing.assert_close(
        out[:fully_masked], torch.zeros_like(out[:fully_masked])
    )


@requires_cuda
@pytest.mark.parametrize(
    "attention", ["self", "cross"], ids=["ragged-self", "ragged-cross"]
)
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_causal_varlen_alibi_diagnostics_match_fa2(
    attention: str,
    batched_slopes: bool,
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_DIAGNOSTIC
    if attention == "self":
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec, seed=20260810
        )
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, lengths_q, lengths_k = (
            make_ragged_cross_packed(spec, seed=20260810)
        )
        assert all(length > 0 for length in lengths_q + lengths_k)
        assert cu_q.data_ptr() != cu_k.data_ptr()
        assert q.shape[0] != k.shape[0]

    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )
        if batched_slopes
        else head_slopes
    )

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
            causal=True,
            alibi_slopes=slopes,
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
            causal=True,
            alibi_slopes=slopes,
            return_attn_probs=True,
        )

    assert isinstance(got, tuple) and len(got) == 3
    assert isinstance(expected, tuple) and len(expected) == 3
    out, softmax_lse, s_dmask = got
    expected_out, expected_lse, expected_s_dmask = expected
    assert slopes.shape == (
        (spec.batch, spec.nheads_q) if batched_slopes else (spec.nheads_q,)
    )
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
        out.float(), expected_out.float(), atol=5e-2, rtol=2e-2
    )
    finite_lse = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(softmax_lse), finite_lse)
    torch.testing.assert_close(
        softmax_lse[finite_lse],
        expected_lse[finite_lse],
        atol=2e-3,
        rtol=1e-5,
    )
    assert torch.equal(softmax_lse[~finite_lse], expected_lse[~finite_lse])


@requires_cuda
@pytest.mark.parametrize("entry_point", ["qkvpacked", "kvpacked"])
def test_causal_varlen_packed_entry_points_inherit_diagnostic_return(
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
            causal=True,
            return_attn_probs=True,
            shape=spec,
        )
        if entry_point == "qkvpacked":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=0.37,
                causal=True,
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
                causal=True,
                return_attn_probs=True,
                shape=spec,
            )

    assert isinstance(got, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_cuda
@pytest.mark.parametrize(
    "entry_point", ["qkvpacked", "kvpacked"], ids=["qkv-packed", "kv-packed"]
)
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
def test_causal_varlen_packed_entry_points_inherit_alibi_diagnostics(
    entry_point: str,
    batched_slopes: bool,
) -> None:
    spec = VARLEN_DIAGNOSTIC
    if entry_point == "qkvpacked":
        q, k, v, cu_q, _ = make_ragged_self_packed(spec, seed=271828)
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
            spec, seed=271828
        )
    head_slopes = torch.linspace(0.01, 0.2, spec.nheads_q, device=q.device)
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(spec.batch)], dim=0
        )
        if batched_slopes
        else head_slopes
    )

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
            causal=True,
            alibi_slopes=slopes,
            return_attn_probs=True,
            shape=spec,
        )
        if entry_point == "qkvpacked":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=0.37,
                causal=True,
                alibi_slopes=slopes,
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
                causal=True,
                alibi_slopes=slopes,
                return_attn_probs=True,
                shape=spec,
            )

    assert isinstance(got, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_cuda
@pytest.mark.parametrize(
    "entry_point",
    ["unpacked", "qkvpacked", "kvpacked"],
    ids=["unpacked", "qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_full_causal_varlen_diagnostic_backward_matches_fp32_and_fa2(
    entry_point: str,
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_DIAGNOSTIC
    base_q, base_k, base_v, cu_q, cu_k, lengths_q, lengths_k = make_packed(
        spec, variant=2, seed=223606
    )
    generator = torch.Generator(device=base_q.device).manual_seed(244949)
    grad_out = torch.randn(
        base_q.shape,
        device=base_q.device,
        dtype=base_q.dtype,
        generator=generator,
    )
    grad_lse = torch.randn(
        (spec.nheads_q, base_q.shape[0]),
        device=base_q.device,
        dtype=torch.float32,
        generator=generator,
    )

    def run(
        package: object,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        api_kwargs: dict[str, object] = {
            "softmax_scale": softmax_scale,
            "causal": True,
            "return_attn_probs": True,
        }
        if package is helion_attention:
            api_kwargs["shape"] = spec

        if entry_point == "unpacked":
            q = base_q.detach().requires_grad_()
            k = base_k.detach().requires_grad_()
            v = base_v.detach().requires_grad_()
            result = package.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                **api_kwargs,
            )
            grad_inputs = (q, k, v)
        elif entry_point == "qkvpacked":
            packed = torch.stack(
                (base_q, base_k, base_v), dim=1
            ).requires_grad_()
            result = package.flash_attn_varlen_qkvpacked_func(
                packed,
                cu_q,
                spec.seqlen_q,
                **api_kwargs,
            )
            grad_inputs = (packed,)
        else:
            q = base_q.detach().requires_grad_()
            packed = torch.stack((base_k, base_v), dim=1).requires_grad_()
            result = package.flash_attn_varlen_kvpacked_func(
                q,
                packed,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                **api_kwargs,
            )
            grad_inputs = (q, packed)

        assert isinstance(result, tuple) and len(result) == 3
        assert all(tensor.requires_grad for tensor in result)
        out, softmax_lse, s_dmask = result
        raw_grads = torch.autograd.grad(
            result,
            grad_inputs,
            grad_outputs=(
                grad_out,
                grad_lse,
                torch.empty_like(s_dmask),
            ),
        )
        if entry_point == "unpacked":
            grads = raw_grads
        elif entry_point == "qkvpacked":
            grads = tuple(raw_grads[0][:, index] for index in range(3))
        else:
            grads = (
                raw_grads[0],
                raw_grads[1][:, 0],
                raw_grads[1][:, 1],
            )
        return (out, softmax_lse, s_dmask), grads

    got, got_grads = run(helion_attention)
    expected_fa2, expected_fa2_grads = run(flash_attn)

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    q_ref = base_q.float().requires_grad_()
    k_ref = base_k.float().requires_grad_()
    v_ref = base_v.float().requires_grad_()
    expected_fp32 = reference_packed(
        q_ref,
        k_ref,
        v_ref,
        lengths_q,
        lengths_k,
        causal=True,
        scale=scale,
    )
    expected_fp32_grads = torch.autograd.grad(
        expected_fp32, (q_ref, k_ref, v_ref), grad_out.float()
    )

    out, softmax_lse, s_dmask = got
    expected_out, expected_lse, expected_s_dmask = expected_fa2
    assert out.shape == expected_out.shape == base_q.shape
    assert out.dtype == expected_out.dtype == spec.dtype
    assert softmax_lse.shape == expected_lse.shape == (
        spec.nheads_q,
        spec.batch * spec.seqlen_q,
    )
    assert softmax_lse.dtype == expected_lse.dtype == torch.float32
    assert s_dmask.shape == expected_s_dmask.shape == (0,)
    assert s_dmask.dtype == expected_s_dmask.dtype == spec.dtype
    torch.testing.assert_close(
        out.float(), expected_out.float(), atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        out.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        softmax_lse, expected_lse, atol=2e-3, rtol=1e-5
    )
    for actual, reference_fa2, reference_fp32 in zip(
        got_grads, expected_fa2_grads, expected_fp32_grads
    ):
        # The nonzero LSE gradient passed above must contribute exactly zero,
        # just as it does in FA2; the fp32 reference differentiates only out.
        torch.testing.assert_close(
            actual.float(), reference_fa2.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual.float(), reference_fp32, atol=8e-2, rtol=2e-2
        )


@requires_cuda
def test_full_causal_varlen_diagnostic_auxiliary_gradients_are_exactly_zero(
) -> None:
    spec = VARLEN_DIAGNOSTIC
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2, seed=282842)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    _, softmax_lse, s_dmask = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        spec.seqlen_q,
        spec.seqlen_k,
        causal=True,
        return_attn_probs=True,
        shape=spec,
    )

    grads = torch.autograd.grad(
        (softmax_lse, s_dmask),
        (q, k, v),
        grad_outputs=(torch.ones_like(softmax_lse), torch.empty_like(s_dmask)),
    )

    assert all(torch.count_nonzero(grad).item() == 0 for grad in grads)


@requires_cuda
def test_causal_varlen_return_attn_probs_false_retains_generated_dispatch(
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
        causal=True,
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
        ("noncausal", "causal"),
        ("paged", "block_table"),
        ("grad", "grad-enabled"),
        ("deterministic", "deterministic=False"),
        ("dropout", "dropout"),
        ("window", "sliding-window"),
        ("softcap", "softcap"),
        ("other-shape", "causal"),
    ],
)
def test_causal_varlen_return_attn_probs_rejects_out_of_scope_calls(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_DIAGNOSTIC
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    kwargs: dict[str, object] = {
        "causal": True,
        "return_attn_probs": True,
        "shape": spec,
    }
    if case == "noncausal":
        spec = VARLEN_ALIBI_NONCAUSAL
        kwargs.update(causal=False, shape=spec)
    elif case == "paged":
        kwargs["block_table"] = torch.zeros(1, device=q.device)
    elif case == "grad":
        q.requires_grad_()
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "window":
        kwargs["window_size"] = (128, 0)
    elif case == "softcap":
        kwargs["softcap"] = 1.0
    else:
        spec = AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, True)
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
    "attention", ["self", "cross"], ids=["ragged-self", "ragged-cross"]
)
@pytest.mark.parametrize(
    ("softcap", "softmax_scale"),
    [
        pytest.param(
            VARLEN_NON_DIAGNOSTIC_SOFTCAP,
            None,
            id="cap-30-default-scale",
        ),
        pytest.param(80.0, 0.37, id="cap-80-custom-scale"),
    ],
)
def test_causal_varlen_positive_softcap_matches_fa2_and_fp32_for_ragged_calls(
    attention: str,
    softcap: float,
    softmax_scale: float | None,
) -> None:
    spec = VARLEN_SOFTCAP
    if attention == "self":
        q, k, v, cu_q, lengths_q = make_ragged_self_packed(
            spec, seed=20260809
        )
        cu_k = cu_q
        lengths_k = lengths_q
    else:
        q, k, v, cu_q, cu_k, lengths_q, lengths_k = (
            make_ragged_cross_packed(spec, seed=20260809)
        )
    # Exercise the nonlinear region; unscaled random inputs would make capped
    # and ordinary attention nearly identical.
    q.mul_(4.0)
    k.mul_(4.0)
    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

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
            causal=True,
            softcap=softcap,
            shape=spec,
        )
        expected_fp32 = reference_softcap_packed(
            q,
            k,
            v,
            lengths_q,
            lengths_k,
            causal=True,
            scale=scale,
            softcap=softcap,
        )

    assert got.shape == q.shape
    assert got.dtype == torch.bfloat16
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

    # FlashAttention is an optional benchmark dependency. Keep the independent
    # fp32 assertion above active in environments with only the dev extras.
    try:
        import flash_attn
    except ImportError:
        return
    with torch.no_grad():
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
            softcap=softcap,
        )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=1e-2, rtol=1e-2
    )


@requires_cuda
@pytest.mark.parametrize(
    "softcap",
    [math.ulp(0.0), 1e-50, 1e40, sys.float_info.max],
    ids=["min-positive", "below-fp32", "above-fp32", "max-finite"],
)
def test_causal_varlen_softcap_finite_range_boundaries_preserve_attention(
    softcap: float,
) -> None:
    spec = VARLEN_SOFTCAP
    lengths = [1] * spec.batch
    q = torch.zeros(
        sum(lengths),
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    cu_seqlens = _cumulative(lengths, q.device)

    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            softcap=softcap,
            shape=spec,
        )

    assert torch.equal(got, torch.ones_like(q))


@requires_cuda
@pytest.mark.parametrize(
    "softcap",
    [
        float.fromhex("0x1.0p+125"),
        float.fromhex("0x1.0p+126"),
        float.fromhex("0x1.fffffep+127"),
        sys.float_info.max,
    ],
    ids=["fp32-large", "fp32-fast-limit", "fp32-max", "fp64-max"],
)
@pytest.mark.parametrize(
    "logit", [1.0, 0.5, 0.015625], ids=["one", "half", "one-over-64"]
)
def test_causal_varlen_large_softcap_preserves_multi_key_logits(
    softcap: float,
    logit: float,
) -> None:
    spec = VARLEN_SOFTCAP
    lengths_q = [1] * spec.batch
    lengths_k = [2] * spec.batch
    q = torch.zeros(
        sum(lengths_q),
        spec.nheads_q,
        spec.head_dim,
        device="cuda",
        dtype=spec.dtype,
    )
    k = torch.zeros(
        sum(lengths_k),
        spec.nheads_kv,
        spec.head_dim,
        device=q.device,
        dtype=spec.dtype,
    )
    v = torch.zeros_like(k)
    q[:, :, 0] = 1.0
    k[1::2, :, 0] = logit
    v[1::2] = 1.0

    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            _cumulative(lengths_q, q.device),
            _cumulative(lengths_k, q.device),
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=1.0,
            causal=True,
            softcap=softcap,
            shape=spec,
        )

    expected_weight = torch.softmax(
        torch.tensor([0.0, logit], dtype=torch.float32), dim=0
    )[1].item()
    assert torch.equal(got, torch.full_like(got, expected_weight))


@requires_cuda
@pytest.mark.parametrize(
    "attention", ["self", "cross"], ids=["ragged-self", "ragged-cross"]
)
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_causal_varlen_softcap_diagnostics_match_fa2(
    attention: str,
    softmax_scale: float | None,
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    spec = VARLEN_SOFTCAP
    if attention == "self":
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec, seed=20260810
        )
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, lengths_q, lengths_k = (
            make_ragged_cross_packed(spec, seed=20260810)
        )
        assert all(length > 0 for length in lengths_q + lengths_k)
        assert cu_q.data_ptr() != cu_k.data_ptr()
        assert q.shape[0] != k.shape[0]

    # Exercise the nonlinear region of the cap rather than only its near-zero
    # approximation.
    q.mul_(4.0)
    k.mul_(4.0)
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
            causal=True,
            softcap=VARLEN_SOFTCAP_VALUE,
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
            causal=True,
            softcap=VARLEN_SOFTCAP_VALUE,
            return_attn_probs=True,
        )

    assert isinstance(got, tuple) and len(got) == 3
    assert isinstance(expected, tuple) and len(expected) == 3
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
    finite_lse = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(softmax_lse), finite_lse)
    torch.testing.assert_close(
        softmax_lse[finite_lse],
        expected_lse[finite_lse],
        atol=2e-3,
        rtol=1e-5,
    )
    assert torch.equal(softmax_lse[~finite_lse], expected_lse[~finite_lse])


@requires_cuda
@pytest.mark.parametrize(
    "name",
    ["flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func"],
    ids=["qkv-packed", "kv-packed"],
)
@pytest.mark.parametrize(
    ("softcap", "softmax_scale"),
    [
        pytest.param(
            VARLEN_NON_DIAGNOSTIC_SOFTCAP,
            None,
            id="cap-30-default-scale",
        ),
        pytest.param(80.0, 0.37, id="cap-80-custom-scale"),
    ],
)
def test_varlen_packed_entry_points_inherit_softcap_support(
    name: str, softcap: float, softmax_scale: float | None
) -> None:
    spec = VARLEN_SOFTCAP
    if name == "flash_attn_varlen_qkvpacked_func":
        q, k, v, cu_q, _ = make_ragged_self_packed(spec, seed=314159)
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, *_ = make_packed(
            spec, variant=1, seed=271828
        )

    with torch.no_grad():
        expected = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            softcap=softcap,
            shape=spec,
        )
        if name == "flash_attn_varlen_qkvpacked_func":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=softmax_scale,
                causal=True,
                softcap=softcap,
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
                softmax_scale=softmax_scale,
                causal=True,
                softcap=softcap,
                shape=spec,
            )

    torch.testing.assert_close(got, expected)


@requires_cuda
@pytest.mark.parametrize(
    "name",
    ["flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func"],
    ids=["qkv-packed", "kv-packed"],
)
def test_varlen_packed_entry_points_inherit_softcap_diagnostics(
    name: str,
) -> None:
    spec = VARLEN_SOFTCAP
    if name == "flash_attn_varlen_qkvpacked_func":
        q, k, v, cu_q, _ = make_ragged_self_packed(spec, seed=173205)
        cu_k = cu_q
    else:
        q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
            spec, seed=223607
        )

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
            causal=True,
            softcap=VARLEN_SOFTCAP_VALUE,
            return_attn_probs=True,
            shape=spec,
        )
        if name == "flash_attn_varlen_qkvpacked_func":
            got = helion_attention.flash_attn_varlen_qkvpacked_func(
                torch.stack((q, k, v), dim=1),
                cu_q,
                spec.seqlen_q,
                softmax_scale=0.37,
                causal=True,
                softcap=VARLEN_SOFTCAP_VALUE,
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
                causal=True,
                softcap=VARLEN_SOFTCAP_VALUE,
                return_attn_probs=True,
                shape=spec,
            )

    assert isinstance(got, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_cuda
@pytest.mark.parametrize(
    "features", ["softcap", "diagnostics", "combined"]
)
def test_varlen_softcap_and_diagnostic_dispatch_stays_separate(
    features: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_SOFTCAP
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    out = torch.empty_like(q)
    diagnostics = (out, torch.empty(1, device=q.device), q.new_empty((0,)))
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def softcap_forward(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(("softcap", args, kwargs))
        return out

    def diagnostic_forward(
        *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(("diagnostics", args, kwargs))
        return diagnostics

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError("feature call reached unrelated varlen dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_softcap_forward", softcap_forward
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", diagnostic_forward
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)

    kwargs: dict[str, object] = {}
    if features == "softcap":
        kwargs["softcap"] = VARLEN_NON_DIAGNOSTIC_SOFTCAP
    elif features == "combined":
        kwargs["softcap"] = VARLEN_SOFTCAP_VALUE
    if features in ("diagnostics", "combined"):
        kwargs["return_attn_probs"] = True
    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            shape=spec,
            **kwargs,
        )

    expected_dispatch = "softcap" if features == "softcap" else "diagnostics"
    assert got is (out if features == "softcap" else diagnostics)
    assert len(calls) == 1
    dispatch, args, dispatch_kwargs = calls[0]
    assert dispatch == expected_dispatch
    assert len(args) == 7
    expected_kwargs = {
        "softcap": (
            VARLEN_NON_DIAGNOSTIC_SOFTCAP
            if features == "softcap"
            else VARLEN_SOFTCAP_VALUE
        )
    }
    assert dispatch_kwargs == (
        {} if features == "diagnostics" else expected_kwargs
    )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("negative-cap", "finite positive"),
        ("nonfinite-cap", "finite positive"),
        ("nan-cap", "finite positive"),
        ("noncausal", "only.*causal varlen"),
        ("other-shape", "only.*causal varlen"),
        ("fp16", "only.*causal varlen"),
        ("gradient", "forward-only"),
        ("dropout", "dropout"),
        ("alibi", "softcap combined with ALiBi"),
        ("window", "sliding-window"),
        ("deterministic", "deterministic=False"),
        ("paging", "only as softcap=50.0"),
    ],
)
def test_varlen_softcap_rejects_out_of_scope_calls_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = VARLEN_SOFTCAP
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0, seed=223607)
    kwargs: dict[str, object] = {
        "causal": True,
        "softcap": VARLEN_NON_DIAGNOSTIC_SOFTCAP,
        "shape": spec,
    }
    if case == "negative-cap":
        kwargs["softcap"] = -1.0
    elif case == "nonfinite-cap":
        kwargs["softcap"] = float("inf")
    elif case == "nan-cap":
        kwargs["softcap"] = float("nan")
    elif case == "noncausal":
        spec = VARLEN_ALIBI_NONCAUSAL
        kwargs.update(causal=False, shape=spec)
    elif case == "other-shape":
        spec = AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, True)
        kwargs["shape"] = spec
    elif case == "fp16":
        spec = AttnShape(8, 512, 512, 16, 16, 64, torch.float16, True)
        q, k, v = (tensor.to(torch.float16) for tensor in (q, k, v))
        kwargs["shape"] = spec
    elif case == "gradient":
        q.requires_grad_()
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "alibi":
        kwargs["alibi_slopes"] = torch.ones(
            spec.nheads_q, device=q.device, dtype=torch.float32
        )
    elif case == "window":
        kwargs["window_size"] = (128, 0)
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "paging":
        kwargs["block_table"] = torch.zeros(1, device=q.device)

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> torch.Tensor:
        raise AssertionError("out-of-scope varlen softcap call reached dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_softcap_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("other-cap", "only as softcap=50.0"),
        ("noncausal", "only.*causal varlen"),
        ("other-shape", "only.*causal varlen"),
        ("fp16", "only.*causal varlen"),
        ("gradient", "forward-only"),
        ("dropout", "dropout"),
        ("alibi", "softcap combined with ALiBi"),
        ("window", "sliding-window"),
        ("deterministic", "deterministic=False"),
        ("paging", "block_table"),
    ],
)
def test_varlen_softcap_diagnostics_reject_out_of_scope_calls_before_dispatch(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_SOFTCAP
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0, seed=244949)
    kwargs: dict[str, object] = {
        "causal": True,
        "softcap": VARLEN_SOFTCAP_VALUE,
        "return_attn_probs": True,
        "shape": spec,
    }
    if case == "other-cap":
        kwargs["softcap"] = 49.0
    elif case == "noncausal":
        spec = VARLEN_ALIBI_NONCAUSAL
        kwargs.update(causal=False, shape=spec)
    elif case == "other-shape":
        spec = AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, True)
        kwargs["shape"] = spec
    elif case == "fp16":
        spec = AttnShape(8, 512, 512, 16, 16, 64, torch.float16, True)
        q, k, v = (tensor.to(torch.float16) for tensor in (q, k, v))
        kwargs["shape"] = spec
    elif case == "gradient":
        q.requires_grad_()
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "alibi":
        kwargs["alibi_slopes"] = torch.ones(
            spec.nheads_q, device=q.device, dtype=torch.float32
        )
    elif case == "window":
        kwargs["window_size"] = (128, 0)
    elif case == "deterministic":
        kwargs["deterministic"] = True
    elif case == "paging":
        kwargs["block_table"] = torch.zeros(1, device=q.device)

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError("out-of-scope softcap diagnostic reached dispatch")

    monkeypatch.setattr(
        helion_attention, "_generic_varlen_softcap_forward", reject_dispatch
    )
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_dispatch
    )
    monkeypatch.setattr(helion_attention, "lookup_varlen", reject_dispatch)
    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_dispatch)
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    "spec",
    (*VARLEN_ALIBI_PROFILES, LLAMA3_VARLEN_INFERENCE),
    ids=["causal", "noncausal", "llama3-gqa"],
)
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
    spec: AttnShape,
    batched_slopes: bool,
    softmax_scale: float | None,
    variant: int,
) -> None:
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
        causal=spec.causal,
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
        causal=spec.causal,
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
        causal=spec.causal,
        alibi_slopes=slopes,
    )
    torch.testing.assert_close(
        got.float(), expected_fa2.float(), atol=1e-2, rtol=1e-2
    )


@requires_cuda
def test_varlen_packed_entry_points_inherit_alibi_support() -> None:
    spec = VARLEN_ALIBI_NONCAUSAL
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
        causal=spec.causal,
        alibi_slopes=slopes,
        shape=spec,
    )
    qkv = torch.stack((q, k, v), dim=1)
    qkvpacked = helion_attention.flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_q,
        spec.seqlen_q,
        causal=spec.causal,
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
        causal=spec.causal,
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
        causal=spec.causal,
        alibi_slopes=batched_slopes,
        shape=spec,
    )
    torch.testing.assert_close(kvpacked, unpacked)


@requires_cuda
def test_varlen_without_alibi_retains_generated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_ALIBI_NONCAUSAL
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
        causal=spec.causal,
        alibi_slopes=None,
        shape=spec,
    )

    assert out is sentinel
    assert seen == [spec]


@requires_cuda
def test_varlen_alibi_bypasses_generated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_ALIBI_NONCAUSAL
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
        causal=spec.causal,
        alibi_slopes=slopes,
        shape=spec,
    )

    assert out is sentinel
    assert seen == [(0.37, spec, slopes)]


@requires_cuda
@pytest.mark.parametrize(
    "return_attn_probs",
    [False, True],
    ids=["alibi", "alibi-diagnostics"],
)
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            AttnShape(8, 512, 512, 16, 16, 64, torch.float16, False),
            id="other-dtype",
        ),
        pytest.param(
            AttnShape(8, 256, 256, 16, 16, 64, torch.bfloat16, True),
            id="other-causal-shape",
        ),
        pytest.param(
            AttnShape(4, 255, 255, 32, 8, 128, torch.bfloat16, True),
            id="llama3-other-maxima",
        ),
        pytest.param(
            AttnShape(4, 256, 256, 32, 8, 128, torch.bfloat16, False),
            id="llama3-noncausal",
        ),
        pytest.param(
            AttnShape(4, 256, 256, 32, 8, 128, torch.float16, True),
            id="llama3-fp16",
        ),
        pytest.param(
            AttnShape(16, 511, 511, 12, 12, 64, torch.bfloat16, False),
            id="bert-other-maxima",
        ),
        pytest.param(
            AttnShape(16, 512, 512, 12, 12, 64, torch.bfloat16, True),
            id="bert-causal",
        ),
        pytest.param(
            AttnShape(16, 512, 512, 12, 12, 64, torch.float16, False),
            id="bert-fp16",
        ),
    ],
)
def test_varlen_alibi_rejects_other_profiles(
    spec: AttnShape, return_attn_probs: bool
) -> None:
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
            return_attn_probs=return_attn_probs,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize("gradient_source", ["q", "slopes"])
@pytest.mark.parametrize(
    "spec",
    (*VARLEN_ALIBI_PROFILES, LLAMA3_VARLEN_INFERENCE, BERT_BASE_VARLEN_INFERENCE),
    ids=["causal", "noncausal", "llama3-gqa", "bert-base"],
)
def test_varlen_alibi_rejects_gradients_before_dispatch(
    gradient_source: str, spec: AttnShape, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            causal=spec.causal,
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
    spec = VARLEN_ALIBI_NONCAUSAL
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=0)
    kwargs = {
        option: value,
        "causal": spec.causal,
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
@pytest.mark.parametrize(
    ("spec", "return_attn_probs"),
    [
        pytest.param(VARLEN_ALIBI_NONCAUSAL, False, id="mha-alibi"),
        pytest.param(LLAMA3_VARLEN_INFERENCE, False, id="llama3-alibi"),
        pytest.param(
            LLAMA3_VARLEN_INFERENCE, True, id="llama3-alibi-diagnostics"
        ),
    ],
)
def test_varlen_alibi_rejects_malformed_slopes_before_dispatch(
    case: str,
    message: str,
    spec: AttnShape,
    return_attn_probs: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        helion_attention, "_generic_varlen_diagnostic_forward", reject_dispatch
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
            causal=spec.causal,
            alibi_slopes=slopes,  # type: ignore[arg-type]
            return_attn_probs=return_attn_probs,
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
@pytest.mark.parametrize(
    "spec", VARLEN_ALIBI_PROFILES, ids=["causal", "noncausal"]
)
def test_full_varlen_grad_enabled_supports_cuda_graph_capture(
    spec: AttnShape,
) -> None:
    q, k, v, cu_q, cu_k, *_ = make_packed(spec, variant=2)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    def run() -> torch.Tensor:
        return helion_attention.flash_attn_varlen_func(
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

    expected = run().detach()
    torch.cuda.synchronize(q.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(q.device)

    assert captured.grad_fn is not None
    torch.testing.assert_close(captured, expected)


@requires_cuda
@pytest.mark.parametrize(
    "attention",
    ["self", "self-empty", "cross", "cross-empty-query"],
    ids=[
        "self-attention",
        "empty-self-attention",
        "cross-attention",
        "empty-query-cross-attention",
    ],
)
def test_ragged_causal_backward_rejects_cuda_graph_capture(
    attention: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = VARLEN_ALIBI_CAUSAL
    if attention in {"self", "self-empty"}:
        requested_lengths = (
            MIXED_EMPTY_SELF_LENGTHS
            if attention == "self-empty"
            else None
        )
        q, k, v, cu_q, _ = make_ragged_self_packed(
            spec, lengths=requested_lengths
        )
        cu_k = cu_q
    else:
        requested_lengths_q = (
            MIXED_EMPTY_CROSS_QUERY_LENGTHS
            if attention == "cross-empty-query"
            else None
        )
        q, k, v, cu_q, cu_k, *_ = make_ragged_cross_packed(
            spec, lengths_q=requested_lengths_q
        )
    q.requires_grad_()

    def reject_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("graph-captured ragged backward reached SDPA")

    monkeypatch.setattr(helion_attention, "dense_attention_sdpa", reject_sdpa)
    torch.cuda.synchronize(q.device)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(NotImplementedError, match="CUDA graph capture"):
        with torch.cuda.graph(graph):
            torch.empty(1, device=q.device).zero_()
            helion_attention.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                spec.seqlen_q,
                spec.seqlen_k,
                causal=True,
                shape=spec,
            )


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
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize("page_size", [256, 512], ids=["page-256", "page-512"])
def test_core_varlen_generic_paged_decode_matches_fa2_and_fp32_for_ragged_permuted_pages(
    causal: bool, softmax_scale: float | None, page_size: int
) -> None:
    q, k, v, cu_q, cu_k, block_table, _, request_kv = make_paged_decode(
        page_size=page_size, seed=20260809
    )
    scale = (
        1.0 / math.sqrt(PAGED_DECODE.head_dim)
        if softmax_scale is None
        else softmax_scale
    )
    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        block_table=block_table,
        shape=(4, 1, 1024, 8, 2, 128),
    )
    expected_fp32 = torch.cat(
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

    assert got.shape == q.shape
    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

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
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        block_table=block_table,
    )
    torch.testing.assert_close(got, expected_fa2, atol=2e-2, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize(
    "page_size", [256, 512], ids=["page-256", "page-512"]
)
def test_core_varlen_large_page_diagnostics_match_fa2_for_ragged_permuted_pages(
    causal: bool, softmax_scale: float | None, page_size: int
) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    q, k, v, cu_q, cu_k, block_table, _, request_kv = make_paged_decode(
        page_size=page_size, seed=20260810
    )
    scale = (
        1.0 / math.sqrt(PAGED_DECODE.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=True,
            block_table=block_table,
            shape=(4, 1, 1024, 8, 2, 128),
        )
        expected_fa2 = flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=True,
            block_table=block_table,
        )

    assert isinstance(got, tuple) and len(got) == 3
    assert isinstance(expected_fa2, tuple) and len(expected_fa2) == 3
    out, softmax_lse, s_dmask = got
    expected_out, expected_lse, expected_s_dmask = expected_fa2
    expected_fp32 = torch.cat(
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

    assert out.shape == expected_out.shape == q.shape
    assert out.dtype == expected_out.dtype == torch.bfloat16
    assert softmax_lse.shape == expected_lse.shape == (
        PAGED_DECODE.nheads_q,
        q.shape[0],
    )
    assert softmax_lse.dtype == expected_lse.dtype == torch.float32
    assert s_dmask.shape == expected_s_dmask.shape == (0,)
    assert s_dmask.dtype == expected_s_dmask.dtype == torch.bfloat16
    assert s_dmask.device == expected_s_dmask.device == q.device
    torch.testing.assert_close(out, expected_out, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(
        out.float(), expected_fp32, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        softmax_lse, expected_lse, atol=2e-3, rtol=1e-5
    )


@requires_cuda
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize("page_size", [256, 512], ids=["page-256", "page-512"])
def test_core_varlen_large_page_softcap_matches_fa2_and_fp32_for_ragged_permuted_pages(
    causal: bool, softmax_scale: float | None, page_size: int
) -> None:
    q, k, v, cu_q, cu_k, block_table, lengths_k, request_kv = (
        make_paged_decode(page_size=page_size, seed=20260809)
    )
    # Exercise the nonlinear part of a cap as large as 50. The logical and
    # physical copies are independent, so scale both representations.
    q.mul_(4.0)
    k.mul_(4.0)
    for logical_k, _ in request_kv:
        logical_k.mul_(4.0)
    scale = (
        1.0 / math.sqrt(PAGED_DECODE.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=CORE_PAGED_SOFTCAP_VALUE,
            block_table=block_table,
            shape=(4, 1, 1024, 8, 2, 128),
        )
        expected_fp32 = reference_softcap_packed(
            q,
            torch.cat([logical_k for logical_k, _ in request_kv]),
            torch.cat([logical_v for _, logical_v in request_kv]),
            [1] * PAGED_DECODE.batch,
            lengths_k,
            causal=causal,
            scale=scale,
            softcap=CORE_PAGED_SOFTCAP_VALUE,
        )

    assert got.shape == q.shape
    assert got.dtype == torch.bfloat16
    assert got.is_contiguous()
    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

    try:
        import flash_attn
    except ImportError:
        return
    with torch.no_grad():
        expected_fa2 = flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=CORE_PAGED_SOFTCAP_VALUE,
            block_table=block_table,
        )
    torch.testing.assert_close(got, expected_fa2, atol=2e-2, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
def test_core_varlen_page_256_softcap_diagnostics_match_fa2_and_fp32_for_ragged_permuted_pages(
    causal: bool, softmax_scale: float | None
) -> None:
    q, k, v, cu_q, cu_k, block_table, lengths_k, request_kv = (
        make_paged_decode(page_size=256, seed=20260811)
    )
    # Exercise the nonlinear part of the cap in both physical and logical K.
    q.mul_(4.0)
    k.mul_(4.0)
    for logical_k, _ in request_kv:
        logical_k.mul_(4.0)
    scale = (
        1.0 / math.sqrt(PAGED_DECODE.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    with torch.no_grad():
        got = helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=CORE_PAGED_SOFTCAP_VALUE,
            return_attn_probs=True,
            block_table=block_table,
            shape=(4, 1, 1024, 8, 2, 128),
        )
        expected_fp32_out, expected_fp32_lse = (
            reference_softcap_packed_diagnostics(
                q,
                torch.cat([logical_k for logical_k, _ in request_kv]),
                torch.cat([logical_v for _, logical_v in request_kv]),
                [1] * PAGED_DECODE.batch,
                lengths_k,
                causal=causal,
                scale=scale,
                softcap=CORE_PAGED_SOFTCAP_VALUE,
            )
        )

    assert isinstance(got, tuple) and len(got) == 3
    out, softmax_lse, s_dmask = got
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert softmax_lse.shape == (PAGED_DECODE.nheads_q, q.shape[0])
    assert softmax_lse.dtype == torch.float32
    assert s_dmask.shape == (0,)
    assert s_dmask.dtype == torch.bfloat16
    assert s_dmask.device == q.device
    torch.testing.assert_close(
        out.float(), expected_fp32_out, atol=5e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        softmax_lse, expected_fp32_lse, atol=5e-2, rtol=2e-3
    )

    try:
        import flash_attn
    except ImportError:
        return
    with torch.no_grad():
        expected_fa2 = flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=CORE_PAGED_SOFTCAP_VALUE,
            return_attn_probs=True,
            block_table=block_table,
        )
    expected_out, expected_lse, expected_s_dmask = expected_fa2
    assert expected_s_dmask.shape == s_dmask.shape
    assert expected_s_dmask.dtype == s_dmask.dtype
    torch.testing.assert_close(out, expected_out, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(
        softmax_lse, expected_lse, atol=2e-3, rtol=1e-5
    )


@requires_cuda
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "softmax_scale", [None, 0.37], ids=["default-scale", "custom-scale"]
)
@pytest.mark.parametrize("page_size", [256, 512], ids=["page-256", "page-512"])
@pytest.mark.parametrize(
    "batched_slopes", [False, True], ids=["head-slopes", "batch-head-slopes"]
)
def test_core_varlen_large_page_alibi_matches_fa2_and_fp32_for_ragged_permuted_pages(
    causal: bool,
    softmax_scale: float | None,
    page_size: int,
    batched_slopes: bool,
) -> None:
    q, k, v, cu_q, cu_k, block_table, lengths_k, request_kv = (
        make_paged_decode(page_size=page_size, seed=20260809)
    )
    head_slopes = torch.linspace(
        0.01,
        0.2,
        PAGED_DECODE.nheads_q,
        device=q.device,
        dtype=torch.float32,
    )
    slopes = (
        torch.stack(
            [head_slopes.roll(batch) for batch in range(PAGED_DECODE.batch)]
        )
        if batched_slopes
        else head_slopes
    )
    scale = (
        1.0 / math.sqrt(PAGED_DECODE.head_dim)
        if softmax_scale is None
        else softmax_scale
    )

    got = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        alibi_slopes=slopes,
        block_table=block_table,
        shape=(4, 1, 1024, 8, 2, 128),
    )
    expected_fp32 = reference_packed(
        q,
        torch.cat([key for key, _ in request_kv]),
        torch.cat([value for _, value in request_kv]),
        [1] * PAGED_DECODE.batch,
        lengths_k,
        causal=causal,
        scale=scale,
        alibi_slopes=slopes,
    )

    assert got.shape == q.shape
    torch.testing.assert_close(got.float(), expected_fp32, atol=5e-2, rtol=2e-2)

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
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        alibi_slopes=slopes,
        block_table=block_table,
    )
    torch.testing.assert_close(got, expected_fa2, atol=2e-2, rtol=1e-2)


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

    def reject_generic(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("page-16 decode reached generic paged dispatch")

    monkeypatch.setattr(helion_attention, "lookup_paged", fake_lookup)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_generic
    )
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
@pytest.mark.parametrize(
    ("page_size", "softcap"),
    [
        pytest.param(256, 0.0, id="page-256-no-cap"),
        pytest.param(
            256,
            CORE_PAGED_SOFTCAP_VALUE,
            id="page-256-softcap-50",
        ),
        pytest.param(512, 0.0, id="page-512-no-cap"),
        pytest.param(
            512,
            CORE_PAGED_SOFTCAP_VALUE,
            id="page-512-softcap-50",
        ),
    ],
)
def test_core_varlen_non_generated_decode_uses_generic_paged_runtime(
    monkeypatch: pytest.MonkeyPatch, page_size: int, softcap: float
) -> None:
    import helion_attention._paged_attention as generic_attention

    q, k, v, cu_q, cu_k, block_table, lengths_k, _ = make_paged_decode(
        page_size=page_size
    )
    sentinel = torch.full_like(q, 7.0)
    seen: dict[str, object] = {}

    def reject_generated(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"page-{page_size} decode reached generated page-16 dispatch"
        )

    def fake_generic(*args: object, **kwargs: object) -> torch.Tensor:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_generated)
    monkeypatch.setattr(generic_attention, "paged_attention", fake_generic)
    result = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=0.19,
        causal=False,
        softcap=softcap,
        block_table=block_table,
        shape=(4, 1, 1024, 8, 2, 128),
    )

    args = seen["args"]
    kwargs = seen["kwargs"]
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[:4] == (q, k, v, cu_q)
    assert args[5] is block_table
    seqused_k = args[4]
    assert isinstance(seqused_k, torch.Tensor)
    torch.testing.assert_close(
        seqused_k, torch.tensor(lengths_k, device=q.device, dtype=torch.int32)
    )
    assert kwargs["softmax_scale"] == 0.19
    assert kwargs["causal"] is False
    assert kwargs["softcap"] == softcap
    assert kwargs["alibi_slopes"] is None
    assert kwargs["return_softmax_lse"] is False
    assert result is sentinel


@requires_cuda
@pytest.mark.parametrize(
    ("page_size", "softcap"),
    [
        pytest.param(256, 0.0, id="page-256"),
        pytest.param(512, 0.0, id="page-512"),
        pytest.param(
            256,
            CORE_PAGED_SOFTCAP_VALUE,
            id="page-256-softcap-50",
        ),
    ],
)
def test_core_varlen_large_page_diagnostics_request_lse_from_generic_runtime(
    monkeypatch: pytest.MonkeyPatch, page_size: int, softcap: float
) -> None:
    import helion_attention._paged_attention as generic_attention

    q, k, v, cu_q, cu_k, block_table, lengths_k, _ = make_paged_decode(
        page_size=page_size
    )
    sentinel = torch.full_like(q, 7.0)
    sentinel_lse = torch.full(
        (PAGED_DECODE.nheads_q, q.shape[0]),
        11.0,
        device=q.device,
        dtype=torch.float32,
    )
    seen: dict[str, object] = {}

    def reject_generated(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"page-{page_size} diagnostics reached generated dispatch"
        )

    def fake_generic(*args: object, **kwargs: object) -> object:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return sentinel, sentinel_lse

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_generated)
    monkeypatch.setattr(generic_attention, "paged_attention", fake_generic)
    result = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=0.19,
        causal=False,
        softcap=softcap,
        return_attn_probs=True,
        block_table=block_table,
        shape=(4, 1, 1024, 8, 2, 128),
    )

    args = seen["args"]
    kwargs = seen["kwargs"]
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    torch.testing.assert_close(
        args[4], torch.tensor(lengths_k, device=q.device, dtype=torch.int32)
    )
    assert kwargs["softmax_scale"] == 0.19
    assert kwargs["causal"] is False
    assert kwargs["softcap"] == softcap
    assert kwargs["return_softmax_lse"] is True
    assert isinstance(result, tuple) and len(result) == 3
    assert result[0] is sentinel
    assert result[1] is sentinel_lse
    assert result[2].shape == (0,)
    assert result[2].dtype == torch.bfloat16
    assert result[2].device == q.device


@requires_cuda
@pytest.mark.parametrize("page_size", [256, 512], ids=["page-256", "page-512"])
def test_core_varlen_large_page_alibi_uses_generic_paged_runtime(
    page_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import helion_attention._paged_attention as generic_attention

    q, k, v, cu_q, cu_k, block_table, lengths_k, _ = make_paged_decode(
        page_size=page_size
    )
    slopes = torch.linspace(
        0.01,
        0.2,
        PAGED_DECODE.nheads_q,
        device=q.device,
        dtype=torch.float32,
    )
    sentinel = torch.full_like(q, 7.0)
    seen: dict[str, object] = {}

    def reject_generated(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"page-{page_size} ALiBi reached generated page-16 dispatch"
        )

    def fake_generic(*args: object, **kwargs: object) -> torch.Tensor:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_generated)
    monkeypatch.setattr(generic_attention, "paged_attention", fake_generic)
    result = helion_attention.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        PAGED_DECODE.seqlen_q,
        PAGED_DECODE.seqlen_k,
        softmax_scale=0.19,
        causal=False,
        alibi_slopes=slopes,
        block_table=block_table,
        shape=(4, 1, 1024, 8, 2, 128),
    )

    args = seen["args"]
    kwargs = seen["kwargs"]
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[:4] == (q, k, v, cu_q)
    assert args[5] is block_table
    torch.testing.assert_close(
        args[4], torch.tensor(lengths_k, device=q.device, dtype=torch.int32)
    )
    assert kwargs["softmax_scale"] == 0.19
    assert kwargs["causal"] is False
    assert kwargs["softcap"] == 0.0
    assert kwargs["alibi_slopes"] is slopes
    assert kwargs["return_softmax_lse"] is False
    assert result is sentinel


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
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(page_size=128)
    with pytest.raises(
        helion_attention.UnsupportedShapeError, match="page_size=128"
    ):
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

    for page_size in (256, 512):
        q, k, v, cu_q, cu_k, block_table, *_ = make_paged_chunked_prefill(
            page_size=page_size
        )
        with pytest.raises(
            helion_attention.UnsupportedShapeError,
            match=f"page_size={page_size}",
        ):
            helion_attention.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                PAGED_CHUNKED_PREFILL.seqlen_q,
                PAGED_CHUNKED_PREFILL.seqlen_k,
                causal=True,
                block_table=block_table,
                shape=PAGED_CHUNKED_PREFILL,
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
@pytest.mark.parametrize(
    ("spec", "factory", "page_size"),
    [
        pytest.param(PAGED_DECODE, make_paged_decode, 16, id="decode-page-16"),
        pytest.param(
            PAGED_DECODE, make_paged_decode, 128, id="decode-other-page"
        ),
        pytest.param(
            PAGED_CHUNKED_PREFILL,
            make_paged_chunked_prefill,
            16,
            id="chunked-prefill-page-16",
        ),
        pytest.param(
            PAGED_CHUNKED_PREFILL,
            make_paged_chunked_prefill,
            512,
            id="chunked-prefill-page-512",
        ),
    ],
)
def test_core_varlen_unsupported_paged_alibi_rejected_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    spec: AttnShape,
    factory: object,
    page_size: int,
) -> None:
    assert callable(factory)
    q, k, v, cu_q, cu_k, block_table, *_ = factory(page_size=page_size)

    def reject_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("paged ALiBi call reached kernel lookup")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_lookup)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_lookup
    )
    with pytest.raises(NotImplementedError, match="ALiBi"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            spec.seqlen_q,
            spec.seqlen_k,
            causal=True,
            alibi_slopes=torch.ones(spec.nheads_q, device=q.device),
            block_table=block_table,
            shape=spec,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("dropout", "dropout", id="dropout"),
        pytest.param("window", "sliding-window", id="window"),
        pytest.param("deterministic", "deterministic=True", id="deterministic"),
    ],
)
@pytest.mark.parametrize(
    ("page_size", "with_alibi"),
    [
        pytest.param(256, False, id="page-256-slope-free"),
        pytest.param(256, True, id="page-256-alibi"),
        pytest.param(512, False, id="page-512-slope-free"),
        pytest.param(512, True, id="page-512-alibi"),
    ],
)
def test_core_varlen_generic_paged_decode_rejects_optional_features_before_dispatch(
    case: str,
    message: str,
    page_size: int,
    with_alibi: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(
        page_size=page_size
    )
    kwargs: dict[str, object] = {
        "causal": True,
        "block_table": block_table,
        "shape": PAGED_DECODE,
    }
    if with_alibi:
        kwargs["alibi_slopes"] = torch.ones(
            PAGED_DECODE.nheads_q, device=q.device, dtype=torch.float32
        )
    if case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "window":
        kwargs["window_size"] = (1, 1)
    else:
        kwargs["deterministic"] = True

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError(
            f"unsupported page-{page_size} feature reached dispatch"
        )

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_dispatch
    )
    with pytest.raises(NotImplementedError, match=message):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param(
            "page-16", "page-size-256 or page-size-512", id="page-16"
        ),
        pytest.param(
            "page-128", "page-size-256 or page-size-512", id="other-page"
        ),
        pytest.param("other-profile", "no-backward bf16", id="other-profile"),
        pytest.param("fp16", "no-backward bf16", id="other-dtype"),
        pytest.param("alibi", "ALiBi", id="alibi"),
        pytest.param("page-512-alibi", "ALiBi", id="page-512-alibi"),
        pytest.param("gradient", "paged-cache backward", id="gradient"),
        pytest.param(
            "page-512-gradient",
            "paged-cache backward",
            id="page-512-gradient",
        ),
        pytest.param("dropout", "dropout", id="dropout"),
        pytest.param("window", "sliding-window", id="window"),
        pytest.param("deterministic", "deterministic=False", id="deterministic"),
        pytest.param(
            "page-512-softcap", "page-size-256", id="page-512-softcap"
        ),
    ],
)
def test_core_varlen_large_page_diagnostics_reject_out_of_scope_calls_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(page_size=256)
    spec = PAGED_DECODE
    kwargs: dict[str, object] = {
        "causal": True,
        "return_attn_probs": True,
        "block_table": block_table,
        "shape": spec,
    }

    scoped_case = case
    if case.startswith("page-512-"):
        q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(
            page_size=512
        )
        kwargs["block_table"] = block_table
        scoped_case = case.removeprefix("page-512-")
    elif case in {"page-16", "page-128"}:
        page_size = int(case.removeprefix("page-"))
        q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(
            page_size=page_size
        )
        kwargs["block_table"] = block_table

    if scoped_case == "other-profile":
        spec = PAGED_CHUNKED_PREFILL
        q, k, v, cu_q, cu_k, block_table, *_ = make_paged_chunked_prefill()
        kwargs.update(block_table=block_table, shape=spec)
    elif scoped_case == "fp16":
        spec = AttnShape(4, 1, 1024, 8, 2, 128, torch.float16, True)
        q, k, v = (tensor.to(torch.float16) for tensor in (q, k, v))
        kwargs["shape"] = spec
    elif scoped_case == "alibi":
        kwargs["alibi_slopes"] = torch.ones(
            spec.nheads_q, device=q.device, dtype=torch.float32
        )
    elif scoped_case == "gradient":
        q.requires_grad_()
    elif scoped_case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif scoped_case == "window":
        kwargs["window_size"] = (1, 1)
    elif scoped_case == "deterministic":
        kwargs["deterministic"] = True
    elif scoped_case == "softcap":
        kwargs["softcap"] = CORE_PAGED_SOFTCAP_VALUE

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError("out-of-scope paged diagnostics reached dispatch")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_dispatch
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
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("other-cap", "only as softcap=50.0", id="other-cap"),
        pytest.param(
            "page-512-other-cap",
            "only as softcap=50.0",
            id="page-512-other-cap",
        ),
        pytest.param(
            "page-16", "page-size-256 or page-size-512", id="page-16"
        ),
        pytest.param(
            "other-page",
            "page-size-256 or page-size-512",
            id="other-page",
        ),
        pytest.param("other-profile", "decode profile", id="other-profile"),
        pytest.param("fp16", "bf16", id="fp16"),
        pytest.param("gradient", "softcap backward", id="gradient"),
        pytest.param(
            "page-512-gradient",
            "softcap backward",
            id="page-512-gradient",
        ),
        pytest.param("dropout", "dropout", id="dropout"),
        pytest.param(
            "alibi", "softcap combined with ALiBi", id="alibi"
        ),
        pytest.param(
            "page-512-alibi",
            "softcap combined with ALiBi",
            id="page-512-alibi",
        ),
        pytest.param("window", "sliding-window", id="window"),
        pytest.param(
            "page-512-diagnostic",
            "page-size-256",
            id="page-512-diagnostic",
        ),
        pytest.param(
            "diagnostic-gradient", "softcap backward", id="diagnostic-gradient"
        ),
        pytest.param(
            "deterministic", "deterministic=True", id="deterministic"
        ),
    ],
)
def test_core_varlen_large_page_softcap_rejects_out_of_scope_calls_before_dispatch(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(page_size=256)
    spec = PAGED_DECODE
    kwargs: dict[str, object] = {
        "causal": True,
        "softcap": CORE_PAGED_SOFTCAP_VALUE,
        "block_table": block_table,
        "shape": spec,
    }
    scoped_case = case
    if case.startswith("page-512-"):
        q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(
            page_size=512
        )
        kwargs["block_table"] = block_table
        scoped_case = case.removeprefix("page-512-")

    if scoped_case == "other-cap":
        kwargs["softcap"] = 49.0
    elif scoped_case in {"page-16", "other-page"}:
        page_size = {"page-16": 16, "other-page": 128}[scoped_case]
        q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(
            page_size=page_size
        )
        kwargs["block_table"] = block_table
    elif scoped_case == "other-profile":
        spec = PAGED_CHUNKED_PREFILL
        q, k, v, cu_q, cu_k, block_table, *_ = make_paged_chunked_prefill()
        kwargs.update(block_table=block_table, shape=spec)
    elif scoped_case == "fp16":
        spec = AttnShape(4, 1, 1024, 8, 2, 128, torch.float16, True)
        q, k, v = (tensor.to(torch.float16) for tensor in (q, k, v))
        kwargs["shape"] = spec
    elif scoped_case == "gradient":
        q.requires_grad_()
    elif scoped_case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif scoped_case == "alibi":
        kwargs["alibi_slopes"] = torch.ones(
            spec.nheads_q, device=q.device, dtype=torch.float32
        )
    elif scoped_case == "window":
        kwargs["window_size"] = (1, 1)
    elif scoped_case == "diagnostic":
        kwargs["return_attn_probs"] = True
    elif scoped_case == "diagnostic-gradient":
        q.requires_grad_()
        kwargs["return_attn_probs"] = True
    else:
        kwargs["deterministic"] = True

    def reject_dispatch(*args: object, **dispatch_kwargs: object) -> object:
        raise AssertionError("out-of-scope core paged softcap reached dispatch")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_dispatch
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
            **kwargs,  # type: ignore[arg-type]
        )


@requires_cuda
@pytest.mark.parametrize("gradient_source", ["q", "slopes"])
@pytest.mark.parametrize("page_size", [256, 512], ids=["page-256", "page-512"])
def test_core_varlen_large_page_alibi_rejects_gradients_before_dispatch(
    gradient_source: str,
    page_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, cu_q, cu_k, block_table, *_ = make_paged_decode(
        page_size=page_size
    )
    slopes = torch.ones(
        PAGED_DECODE.nheads_q, device=q.device, dtype=torch.float32
    )
    if gradient_source == "q":
        q.requires_grad_()
    else:
        slopes.requires_grad_()

    def reject_dispatch(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"gradient-bearing page-{page_size} ALiBi reached dispatch"
        )

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_dispatch)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_dispatch
    )
    with pytest.raises(NotImplementedError, match="ALiBi backward"):
        helion_attention.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            PAGED_DECODE.seqlen_q,
            PAGED_DECODE.seqlen_k,
            causal=True,
            alibi_slopes=slopes,
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
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    ("spec", "factory", "page_size"),
    [
        pytest.param(PAGED_DECODE, make_paged_decode, 16, id="decode-page-16"),
        pytest.param(PAGED_DECODE, make_paged_decode, 256, id="decode-page-256"),
        pytest.param(PAGED_DECODE, make_paged_decode, 512, id="decode-page-512"),
        pytest.param(
            PAGED_CHUNKED_PREFILL,
            make_paged_chunked_prefill,
            16,
            id="chunked-prefill-page-16",
        ),
    ],
)
def test_core_varlen_paged_rejects_gradients_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    spec: AttnShape,
    factory: object,
    page_size: int,
    deterministic: bool,
) -> None:
    assert callable(factory)
    q, k, v, cu_q, cu_k, block_table, *_ = factory(page_size=page_size)

    def reject_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("gradient-bearing input reached paged kernel lookup")

    monkeypatch.setattr(helion_attention, "lookup_paged", reject_lookup)
    monkeypatch.setattr(
        helion_attention, "_generic_paged_varlen_forward", reject_lookup
    )
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
            deterministic=deterministic,
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
