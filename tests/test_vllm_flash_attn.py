"""Regression tests for the vLLM FlashAttention compatibility module."""

from __future__ import annotations

import inspect
import time

import pytest
import torch

from helion_attention import vllm_flash_attn as compat
from helion_attention import AttnShape

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VLLM_KEYWORDS = {
    "q",
    "k",
    "v",
    "max_seqlen_q",
    "cu_seqlens_q",
    "max_seqlen_k",
    "cu_seqlens_k",
    "seqused_k",
    "q_v",
    "dropout_p",
    "softmax_scale",
    "causal",
    "window_size",
    "softcap",
    "alibi_slopes",
    "deterministic",
    "return_attn_probs",
    "block_table",
    "return_softmax_lse",
    "out",
    "scheduler_metadata",
    "q_descale",
    "k_descale",
    "v_descale",
    "num_splits",
    "output_scale",
    "fa_version",
    "s_aux",
    "cp_world_size",
    "cp_rank",
    "cp_tot_seqused_k",
    "dynamic_causal",
    "mask_mod",
    "block_sparse_tensors",
    "aux_tensors",
    "aux_tensor_leading_dims",
}


def _reference_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window: tuple[int, int],
    softcap: float,
    slopes: torch.Tensor,
    sink: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    nheads_q = q.shape[1]
    group = nheads_q // k.shape[1]
    query = q.float().transpose(0, 1)
    key = k.float().transpose(0, 1).repeat_interleave(group, dim=0)
    value = v.float().transpose(0, 1).repeat_interleave(group, dim=0)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    scores = softcap * torch.tanh(scores / softcap)

    rows = torch.arange(q.shape[0])[:, None]
    columns = torch.arange(k.shape[0])[None, :]
    aligned_rows = rows + k.shape[0] - q.shape[0]
    scores -= slopes[:, None, None] * (aligned_rows - columns).abs()[None]
    keep = (
        columns <= aligned_rows
        if causal
        else torch.ones_like(columns, dtype=torch.bool)
    )
    if window[0] >= 0:
        keep &= columns >= aligned_rows - window[0]
    if window[1] >= 0:
        keep &= columns <= aligned_rows + window[1]
    scores.masked_fill_(~keep[None], float("-inf"))

    augmented = torch.cat(
        (sink[:, None, None].expand(-1, q.shape[0], 1), scores), dim=-1
    )
    probabilities = torch.softmax(augmented, dim=-1)[..., 1:]
    result = torch.matmul(probabilities, value).transpose(0, 1)
    return result, torch.logsumexp(augmented, dim=-1)


def test_vllm_surface_has_no_shape_argument() -> None:
    parameters = inspect.signature(compat.flash_attn_varlen_func).parameters
    assert VLLM_KEYWORDS <= parameters.keys()
    assert "shape" not in parameters
    assert parameters["dropout_p"].default == 0.0
    assert parameters["deterministic"].default is False
    assert parameters["return_attn_probs"].default is False
    assert parameters["output_scale"].default is None
    assert parameters["fa_version"].default == 2
    assert parameters["cp_world_size"].default == 1
    assert parameters["cp_rank"].default == 0
    assert parameters["block_sparse_tensors"].default is None
    assert parameters["aux_tensor_leading_dims"].default is None
    assert compat.compile_flash_attn_varlen_func_from_specs is None

    assert compat.is_fa_version_supported(2)
    assert compat.is_fa_version_supported(3)
    assert not compat.is_fa_version_supported(4)
    assert compat.fa_version_unsupported_reason(3) is None
    assert compat.fa_version_unsupported_reason(4)
    compat.get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=4,
        num_heads_q=2,
        num_heads_kv=1,
        headdim=8,
        cache_seqlens=torch.tensor([4], dtype=torch.int32),
    )


def test_nonzero_dropout_is_rejected_explicitly() -> None:
    tensor = torch.zeros(1, 1, 1)
    cumulative = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="dropout_p=0.0"):
        compat.flash_attn_varlen_func(
            q=tensor,
            k=tensor,
            v=tensor,
            max_seqlen_q=1,
            cu_seqlens_q=cumulative,
            max_seqlen_k=1,
            cu_seqlens_k=cumulative,
            dropout_p=0.1,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("deterministic", True, id="deterministic"),
        pytest.param("return_attn_probs", True, id="attention-probabilities"),
        pytest.param("output_scale", torch.tensor(1.0), id="output-scale"),
        pytest.param("block_sparse_tensors", object(), id="block-sparse"),
        pytest.param("aux_tensor_leading_dims", (0,), id="aux-leading-dims"),
    ],
)
def test_unsupported_upstream_optional_values_are_rejected(
    name: str, value: object
) -> None:
    tensor = torch.zeros(1, 1, 1)
    cumulative = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        compat.flash_attn_varlen_func(
            q=tensor,
            k=tensor,
            v=tensor,
            max_seqlen_q=1,
            cu_seqlens_q=cumulative,
            max_seqlen_k=1,
            cu_seqlens_k=cumulative,
            **{name: value},
        )


def test_fa3_mla_prefill_accepts_current_optional_surface() -> None:
    generator = torch.Generator().manual_seed(911)
    query_length, key_length = 2, 3
    nheads, head_dim, value_dim = 2, 4, 6
    q = torch.randn(query_length, nheads, head_dim, generator=generator)
    q_v = torch.randn(query_length, nheads, value_dim, generator=generator)
    k = torch.randn(key_length, 1, head_dim, generator=generator)
    v = torch.randn(key_length, 1, value_dim, generator=generator)
    cu_q = torch.tensor([0, query_length], dtype=torch.int32)
    cu_k = torch.tensor([0, key_length], dtype=torch.int32)
    scale = 0.23

    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        q_v=q_v,
        max_seqlen_q=query_length,
        cu_seqlens_q=cu_q,
        max_seqlen_k=key_length,
        cu_seqlens_k=cu_k,
        dropout_p=0.0,
        softmax_scale=scale,
        deterministic=False,
        return_attn_probs=False,
        output_scale=None,
        fa_version=3,
        block_sparse_tensors=None,
        aux_tensors=None,
        aux_tensor_leading_dims=None,
        dynamic_causal=None,
    )
    assert isinstance(result, torch.Tensor)

    query = q.float().transpose(0, 1)
    query_v = q_v.float().transpose(0, 1)
    key = k.float().transpose(0, 1).expand(nheads, -1, -1)
    value = v.float().transpose(0, 1).expand(nheads, -1, -1)
    scores = (
        torch.matmul(query, key.transpose(-1, -2))
        + torch.matmul(query_v, value.transpose(-1, -2))
    ) * scale
    expected = torch.matmul(torch.softmax(scores, dim=-1), value).transpose(0, 1)
    torch.testing.assert_close(result, expected)


def test_default_fa_version_uses_fa2_alibi_lse_convention() -> None:
    tokens, nheads, head_dim = 4, 2, 8
    q = torch.zeros(tokens, nheads, head_dim)
    k = torch.zeros_like(q)
    v = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)
    cumulative = torch.tensor([0, tokens], dtype=torch.int32)
    slopes = torch.tensor([0.1, 0.4])
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "max_seqlen_q": tokens,
        "cu_seqlens_q": cumulative,
        "max_seqlen_k": tokens,
        "cu_seqlens_k": cumulative,
        "causal": True,
        "alibi_slopes": slopes,
        "return_softmax_lse": True,
    }

    implicit = compat.flash_attn_varlen_func(**kwargs)
    explicit_fa2 = compat.flash_attn_varlen_func(**kwargs, fa_version=2)
    explicit_fa3 = compat.flash_attn_varlen_func(**kwargs, fa_version=3)
    assert isinstance(implicit, tuple)
    assert isinstance(explicit_fa2, tuple)
    assert isinstance(explicit_fa3, tuple)
    implicit_out, implicit_lse = implicit
    fa2_out, fa2_lse = explicit_fa2
    fa3_out, fa3_lse = explicit_fa3

    torch.testing.assert_close(implicit_out, fa2_out)
    torch.testing.assert_close(implicit_lse, fa2_lse)
    torch.testing.assert_close(implicit_out, fa3_out)
    expected_shift = slopes[:, None] * torch.arange(tokens, dtype=torch.float32)
    torch.testing.assert_close(fa2_lse - fa3_lse, expected_shift)


@requires_cuda
def test_fa3_mla_qv_cp_and_scalar_tensor_maxima() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1601)
    query_length, local_key_length = 2, 2
    nheads, head_dim, value_dim, page_size = 2, 16, 32, 4
    scale = 0.17
    q = torch.randn(
        query_length,
        nheads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_v = torch.randn(
        query_length,
        nheads,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        1,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        1,
        page_size,
        1,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    cu_q = torch.tensor([0, query_length], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor(
        [local_key_length], device="cuda", dtype=torch.int32
    )
    total_seqused_k = torch.tensor([5], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)

    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        q_v=q_v,
        max_seqlen_q=torch.tensor(query_length, device="cuda", dtype=torch.int32),
        cu_seqlens_q=cu_q,
        max_seqlen_k=torch.tensor(page_size, device="cuda", dtype=torch.int64),
        seqused_k=seqused_k,
        block_table=block_table,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=True,
        fa_version=3,
        cp_world_size=2,
        cp_rank=1,
        cp_tot_seqused_k=total_seqused_k,
    )
    assert isinstance(result, tuple)
    actual, actual_lse = result

    query = q.float().transpose(0, 1)
    query_v = q_v.float().transpose(0, 1)
    key = k[0, :local_key_length, 0].float()
    value = v[0, :local_key_length, 0].float()
    scores = (
        torch.matmul(query, key.transpose(0, 1))
        + torch.matmul(query_v, value.transpose(0, 1))
    ) * scale
    global_keys = torch.arange(local_key_length, device="cuda") * 2 + 1
    aligned_queries = torch.arange(query_length, device="cuda") + 5 - query_length
    keep = global_keys[None, :] <= aligned_queries[:, None]
    scores = scores.masked_fill(~keep[None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probabilities, value).transpose(0, 1)
    expected_lse = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(actual.float(), expected, atol=4e-2, rtol=3e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=3e-2, rtol=2e-2)


@requires_cuda
def test_fp8_without_out_returns_bfloat16() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("PyTorch build does not expose float8_e4m3fn")
    generator = torch.Generator(device="cuda").manual_seed(741)

    def fp8_random(tokens: int) -> torch.Tensor:
        return torch.randn(
            tokens, 2, 16, device="cuda", dtype=torch.float32, generator=generator
        ).to(fp8_dtype)

    q = fp8_random(2)
    k = fp8_random(3)
    v = fp8_random(3)
    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=2,
        cu_seqlens_q=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        max_seqlen_k=3,
        cu_seqlens_k=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
        fa_version=3,
    )

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.bfloat16


def test_nonpaged_fallback_features_and_out_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force both loops to cross tile boundaries with a small test problem.
    monkeypatch.setattr(compat, "_QUERY_TILE_SIZE", 1)
    monkeypatch.setattr(compat, "_KEY_TILE_SIZE", 2)
    generator = torch.Generator().manual_seed(123)
    query_lengths = [2, 1]
    key_lengths = [3, 2]
    q = torch.randn(sum(query_lengths), 2, 4, generator=generator)
    k = torch.randn(sum(key_lengths), 1, 4, generator=generator)
    v = torch.randn(sum(key_lengths), 1, 4, generator=generator)
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)
    q_descale = torch.tensor([[0.75], [1.25]])
    k_descale = torch.tensor([[1.1], [0.9]])
    v_descale = torch.tensor([[0.8], [1.2]])
    slopes = torch.tensor([0.1, 0.2])
    sink = torch.tensor([-0.3, 0.4])
    scale = 0.37
    softcap = 1.5
    window = (2, 0)
    out = torch.empty_like(q)

    returned, lse = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=2,
        cu_seqlens_q=cu_q,
        max_seqlen_k=3,
        cu_seqlens_k=cu_k,
        softmax_scale=scale,
        causal=True,
        window_size=window,
        softcap=softcap,
        alibi_slopes=slopes,
        return_softmax_lse=True,
        out=out,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=sink,
        dynamic_causal=None,
        mask_mod=None,
        aux_tensors=None,
    )

    expected_outputs = []
    expected_lse = []
    q_start = 0
    k_start = 0
    for request, (query_length, key_length) in enumerate(
        zip(query_lengths, key_lengths)
    ):
        expected, expected_request_lse = _reference_one(
            q[q_start : q_start + query_length] * q_descale[request],
            k[k_start : k_start + key_length] * k_descale[request],
            v[k_start : k_start + key_length] * v_descale[request],
            scale=scale,
            causal=True,
            window=window,
            softcap=softcap,
            slopes=slopes,
            sink=sink,
        )
        expected_outputs.append(expected)
        expected_lse.append(expected_request_lse)
        q_start += query_length
        k_start += key_length

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.cat(expected_outputs))
    torch.testing.assert_close(lse, torch.cat(expected_lse, dim=1))


def test_paged_nonfinite_cache_tail_is_masked() -> None:
    generator = torch.Generator().manual_seed(987)
    q = torch.randn(2, 2, 4, generator=generator)
    clean_k = torch.zeros(2, 4, 1, 4)
    clean_v = torch.zeros_like(clean_k)
    clean_k[0, :1] = torch.randn(1, 1, 4, generator=generator)
    clean_v[0, :1] = torch.randn(1, 1, 4, generator=generator)
    clean_k[1, :3] = torch.randn(3, 1, 4, generator=generator)
    clean_v[1, :3] = torch.randn(3, 1, 4, generator=generator)

    poisoned_k = clean_k.clone()
    poisoned_v = clean_v.clone()
    poisoned_k[0, 1:3] = float("nan")
    poisoned_v[0, 1] = float("nan")
    poisoned_v[0, 2] = float("inf")
    cumulative_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    seqused_k = torch.tensor([1, 3], dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    kwargs = {
        "q": q,
        "max_seqlen_q": 1,
        "cu_seqlens_q": cumulative_q,
        "max_seqlen_k": 3,
        "seqused_k": seqused_k,
        "block_table": block_table,
        "causal": True,
        "fa_version": 3,
    }

    expected = compat.flash_attn_varlen_func(k=clean_k, v=clean_v, **kwargs)
    actual = compat.flash_attn_varlen_func(k=poisoned_k, v=poisoned_v, **kwargs)
    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("fa_version", "positive_infinity"), [(2, True), (3, False)]
)
def test_zero_key_lse_uses_version_specific_sentinel(
    fa_version: int, positive_infinity: bool
) -> None:
    q = torch.zeros(2, 2, 4)
    k = torch.empty(0, 1, 4)
    v = torch.empty_like(k)
    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=2,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        max_seqlen_k=0,
        cu_seqlens_k=torch.tensor([0, 0], dtype=torch.int32),
        causal=True,
        return_softmax_lse=True,
        fa_version=fa_version,
    )
    assert isinstance(result, tuple)
    output, lse = result
    torch.testing.assert_close(output, torch.zeros_like(output))
    assert torch.isposinf(lse).all() if positive_infinity else torch.isneginf(lse).all()


@requires_cuda
@pytest.mark.parametrize("paged", [False, True], ids=["packed", "paged"])
@pytest.mark.parametrize(
    ("fa_version", "positive_infinity"), [(2, True), (3, False)]
)
def test_cuda_fully_masked_lse_uses_version_specific_sentinel(
    paged: bool, fa_version: int, positive_infinity: bool
) -> None:
    q = torch.zeros(3, 2, 16, device="cuda", dtype=torch.bfloat16)
    if paged:
        k = torch.zeros(1, 4, 1, 16, device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        key_kwargs = {
            "seqused_k": torch.tensor([1], device="cuda", dtype=torch.int32),
            "block_table": torch.tensor([[0]], device="cuda", dtype=torch.int32),
        }
    else:
        k = torch.zeros(1, 1, 16, device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        key_kwargs = {
            "cu_seqlens_k": torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        }

    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=3,
        cu_seqlens_q=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
        max_seqlen_k=1,
        causal=True,
        return_softmax_lse=True,
        fa_version=fa_version,
        **key_kwargs,
    )
    assert isinstance(result, tuple)
    output, lse = result
    torch.testing.assert_close(output[:2], torch.zeros_like(output[:2]))
    empty_lse = lse[:, :2]
    assert (
        torch.isposinf(empty_lse).all()
        if positive_infinity
        else torch.isneginf(empty_lse).all()
    )
    assert torch.isfinite(lse[:, 2:]).all()


@pytest.mark.parametrize("fa_version", [4, 999])
def test_unsupported_fa_version_is_rejected(fa_version: int) -> None:
    tensor = torch.zeros(1, 1, 1)
    cumulative = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match=rf"unsupported fa_version={fa_version}"):
        compat.flash_attn_varlen_func(
            q=tensor,
            k=tensor,
            v=tensor,
            max_seqlen_q=1,
            cu_seqlens_q=cumulative,
            max_seqlen_k=1,
            cu_seqlens_k=cumulative,
            fa_version=fa_version,
        )


@requires_cuda
def test_paged_fallback_supports_cuda_graph_capture() -> None:
    generator = torch.Generator(device="cuda").manual_seed(321)
    query_lengths = [2, 1]
    key_lengths = [5, 3]
    nheads_q, nheads_kv, head_dim, page_size = 4, 2, 8, 2
    q = torch.randn(
        sum(query_lengths),
        nheads_q,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    per_request_k = [
        torch.randn(
            length,
            nheads_kv,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for length in key_lengths
    ]
    per_request_v = [
        torch.randn(
            key.shape,
            device=key.device,
            dtype=key.dtype,
            generator=generator,
        )
        for key in per_request_k
    ]
    k_cache = torch.zeros(
        7,
        page_size,
        nheads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.tensor([[3, 0, 5], [2, 4, 6]], device="cuda", dtype=torch.int32)
    for request, (key, value) in enumerate(zip(per_request_k, per_request_v)):
        for logical_block in range((key_lengths[request] + page_size - 1) // page_size):
            physical_block = int(block_table[request, logical_block])
            start = logical_block * page_size
            stop = min(start + page_size, key_lengths[request])
            k_cache[physical_block, : stop - start] = key[start:stop]
            v_cache[physical_block, : stop - start] = value[start:stop]

    cu_q = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor(key_lengths, device="cuda", dtype=torch.int32)
    kwargs = {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "max_seqlen_q": torch.tensor(
            max(query_lengths), device="cuda", dtype=torch.int32
        ),
        "cu_seqlens_q": cu_q,
        "max_seqlen_k": torch.tensor(
            max(key_lengths), device="cuda", dtype=torch.int32
        ),
        "seqused_k": seqused_k,
        "block_table": block_table,
        "causal": True,
        "fa_version": 3,
    }

    expected_result = compat.flash_attn_varlen_func(**kwargs)
    assert isinstance(expected_result, torch.Tensor)
    expected = expected_result.clone()
    captured_out = torch.empty_like(q)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = compat.flash_attn_varlen_func(**kwargs, out=captured_out)
    graph.replay()
    torch.cuda.synchronize()

    assert isinstance(returned, torch.Tensor)
    assert returned.data_ptr() == captured_out.data_ptr()
    torch.testing.assert_close(captured_out, expected)


@requires_cuda
def test_fa2_effective_global_alibi_lse_is_stable_during_capture() -> None:
    tokens, nheads, head_dim, page_size = 5, 2, 16, 4
    q = torch.zeros(tokens, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(2, page_size, nheads, head_dim, device=q.device, dtype=q.dtype)
    v = torch.randn(
        k.shape,
        device=k.device,
        dtype=k.dtype,
        generator=torch.Generator(device="cuda").manual_seed(505),
    )
    max_seqlen_k = torch.tensor(tokens, device="cuda", dtype=torch.int32)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "max_seqlen_q": torch.tensor(tokens, device="cuda", dtype=torch.int32),
        "cu_seqlens_q": torch.tensor(
            [0, tokens], device="cuda", dtype=torch.int32
        ),
        "max_seqlen_k": max_seqlen_k,
        "seqused_k": torch.tensor([tokens], device="cuda", dtype=torch.int32),
        "block_table": torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
        "causal": True,
        "window_size": (tokens, 0),
        "alibi_slopes": torch.tensor([0.2, 0.4], device="cuda"),
        "return_softmax_lse": True,
        "fa_version": 2,
    }

    eager_result = compat.flash_attn_varlen_func(**kwargs)
    assert isinstance(eager_result, tuple)
    eager_out, eager_lse = (tensor.clone() for tensor in eager_result)
    captured_out = torch.empty_like(eager_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = compat.flash_attn_varlen_func(**kwargs, out=captured_out)
    assert isinstance(captured_result, tuple)
    returned_out, captured_lse = captured_result
    graph.replay()
    torch.cuda.synchronize()

    assert returned_out.data_ptr() == captured_out.data_ptr()
    torch.testing.assert_close(captured_out, eager_out)
    torch.testing.assert_close(captured_lse, eager_lse)


@requires_cuda
def test_paged_2048_has_bounded_launch_and_memory_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens, nheads_q, nheads_kv, head_dim, page_size = 2048, 4, 2, 64, 16
    num_blocks = tokens // page_size
    generator = torch.Generator(device="cuda").manual_seed(2048)
    q = torch.randn(
        tokens,
        nheads_q,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        num_blocks,
        page_size,
        nheads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        k.shape,
        device=k.device,
        dtype=k.dtype,
        generator=generator,
    )
    cu_q = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).reshape(1, -1)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "max_seqlen_q": tokens,
        "cu_seqlens_q": cu_q,
        "max_seqlen_k": tokens,
        "seqused_k": seqused_k,
        "block_table": block_table,
        "causal": True,
        "fa_version": 3,
    }

    # Compile before measuring. The production path must remain one adapter
    # dispatch, regardless of the O(sequence^2) attention work done in-kernel.
    original = compat._paged_attention
    calls = 0

    def counted(*args: object, **call_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **call_kwargs)

    monkeypatch.setattr(compat, "_paged_attention", counted)
    compat.flash_attn_varlen_func(**kwargs)
    torch.cuda.synchronize()
    calls = 0
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.memory_allocated()
    started = time.perf_counter()
    result = compat.flash_attn_varlen_func(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    extra_peak_memory = torch.cuda.max_memory_allocated() - starting_memory

    assert isinstance(result, torch.Tensor)
    assert calls == 1
    assert elapsed < 2.0
    assert extra_peak_memory < 128 * 1024 * 1024


@requires_cuda
def test_nonpaged_2048_has_bounded_launch_and_memory_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens, nheads_q, nheads_kv, head_dim = 2048, 4, 2, 64
    generator = torch.Generator(device="cuda").manual_seed(2049)
    q = torch.randn(
        tokens,
        nheads_q,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        tokens,
        nheads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        k.shape,
        device=k.device,
        dtype=k.dtype,
        generator=generator,
    )
    cumulative = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "max_seqlen_q": tokens,
        "cu_seqlens_q": cumulative,
        "max_seqlen_k": tokens,
        "cu_seqlens_k": cumulative,
        "causal": True,
        "fa_version": 3,
    }

    original = compat._packed_attention
    calls = 0

    def counted(*args: object, **call_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **call_kwargs)

    monkeypatch.setattr(compat, "_packed_attention", counted)
    compat.flash_attn_varlen_func(**kwargs)
    torch.cuda.synchronize()
    calls = 0
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.memory_allocated()
    started = time.perf_counter()
    result = compat.flash_attn_varlen_func(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    extra_peak_memory = torch.cuda.max_memory_allocated() - starting_memory

    assert isinstance(result, torch.Tensor)
    assert calls == 1
    assert elapsed < 2.0
    assert extra_peak_memory < 128 * 1024 * 1024


@requires_cuda
@pytest.mark.parametrize(
    "window", [(-1, -1), (5, 0)], ids=["unbounded", "effective-global"]
)
def test_fa2_causal_alibi_lse_matches_upstream(window: tuple[int, int]) -> None:
    flash_attn = pytest.importorskip("flash_attn")
    generator = torch.Generator(device="cuda").manual_seed(456)
    query_lengths = [3, 2]
    key_lengths = [5, 4]
    nheads, head_dim = 2, 64

    def random(total: int) -> torch.Tensor:
        return torch.randn(
            total,
            nheads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

    q = random(sum(query_lengths))
    k = random(sum(key_lengths))
    v = random(sum(key_lengths))
    cu_q = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 9], device="cuda", dtype=torch.int32)
    slopes = torch.tensor([0.2, 0.4], device="cuda", dtype=torch.float32)

    expected_out, expected_lse, _ = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(query_lengths),
        max(key_lengths),
        causal=True,
        window_size=window,
        alibi_slopes=slopes,
        return_attn_probs=True,
    )
    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max(query_lengths),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(key_lengths),
        cu_seqlens_k=cu_k,
        causal=True,
        window_size=window,
        alibi_slopes=slopes,
        return_softmax_lse=True,
        fa_version=2,
    )
    assert isinstance(result, tuple)
    actual_out, actual_lse = result

    torch.testing.assert_close(actual_out, expected_out, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-3, rtol=1e-3)


@requires_cuda
def test_fa2_noncausal_one_sided_alibi_lse_matches_upstream() -> None:
    flash_attn = pytest.importorskip("flash_attn")
    tokens, nheads, head_dim = 4, 2, 64
    q = torch.zeros(tokens, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.randn(
        q.shape,
        device=q.device,
        dtype=q.dtype,
        generator=torch.Generator(device="cuda").manual_seed(654),
    )
    cumulative = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    slopes = torch.tensor([0.1, 0.4], device="cuda", dtype=torch.float32)
    window = (-1, 0)

    expected_out, expected_lse, _ = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cumulative,
        cumulative,
        tokens,
        tokens,
        causal=False,
        window_size=window,
        alibi_slopes=slopes,
        return_attn_probs=True,
    )
    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=tokens,
        cu_seqlens_q=cumulative,
        max_seqlen_k=tokens,
        cu_seqlens_k=cumulative,
        causal=False,
        window_size=window,
        alibi_slopes=slopes,
        return_softmax_lse=True,
        fa_version=2,
    )
    assert isinstance(result, tuple)
    actual_out, actual_lse = result

    torch.testing.assert_close(actual_out, expected_out, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-3, rtol=1e-3)


@requires_cuda
def test_exact_shape_dispatch_infers_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = torch.zeros(8, 16, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    cumulative = torch.arange(9, device="cuda", dtype=torch.int32)
    seen: dict[str, object] = {}

    def fake_has_kernel(spec: object) -> bool:
        seen["spec"] = spec
        return True

    def fake_specialized(*args: object, **kwargs: object) -> torch.Tensor:
        seen["shape"] = kwargs["shape"]
        return torch.full_like(q, 7)

    monkeypatch.setattr(compat, "has_varlen_kernel", fake_has_kernel)
    monkeypatch.setattr(compat, "_specialized_varlen_func", fake_specialized)
    result = compat.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=512,
        cu_seqlens_q=cumulative,
        max_seqlen_k=512,
        cu_seqlens_k=cumulative,
        causal=True,
    )

    spec = seen["spec"]
    assert spec is seen["shape"]
    assert isinstance(spec, AttnShape)
    assert spec.batch == 8
    assert spec.seqlen_q == spec.seqlen_k == 512
    assert spec.nheads_q == spec.nheads_kv == 16
    assert spec.head_dim == 64
    assert spec.dtype == torch.bfloat16
    assert spec.causal is True
    torch.testing.assert_close(result, torch.full_like(q, 7))
