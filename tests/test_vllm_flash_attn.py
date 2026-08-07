"""Regression tests for the vLLM FlashAttention compatibility module."""

from __future__ import annotations

import inspect

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
    "softmax_scale",
    "causal",
    "window_size",
    "softcap",
    "alibi_slopes",
    "block_table",
    "return_softmax_lse",
    "out",
    "scheduler_metadata",
    "q_descale",
    "k_descale",
    "v_descale",
    "num_splits",
    "fa_version",
    "s_aux",
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
        torch.randn_like(key, generator=generator) for key in per_request_k
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
        "max_seqlen_q": max(query_lengths),
        "cu_seqlens_q": cu_q,
        "max_seqlen_k": max(key_lengths),
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
