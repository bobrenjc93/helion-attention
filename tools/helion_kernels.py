"""Helion sources for the checked-in kernels.

This module is the *only* place Helion is used, and it is a development-time
dependency: ``tools/generate.py`` imports it to autotune and emit plain Triton
into ``helion_attention/kernels/``. Nothing under ``helion_attention/`` imports
this file, so the installed package never needs Helion.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def _sdpa_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=sm_scale,
        enable_gqa=q.size(2) != k.size(2),
    ).transpose(1, 2)


def _causal_sdpa_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    seqlen_q = q.size(1)
    seqlen_k = k.size(1)
    row = torch.arange(seqlen_q, device=q.device)[:, None]
    col = torch.arange(seqlen_k, device=q.device)[None, :]
    mask = col <= row + seqlen_k - seqlen_q
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=sm_scale,
        attn_mask=mask,
        enable_gqa=q.size(2) != k.size(2),
    ).transpose(1, 2)


def _varlen_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Packed-sequence oracle used only while autotuning."""
    del max_seqlen_q, max_seqlen_k
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    out = torch.empty_like(q)
    for q_start, q_end, k_start, k_end in zip(
        cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]
    ):
        q_seq = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_seq = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_seq = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
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
            scale=sm_scale,
            enable_gqa=q.size(1) != k.size(1),
        )
        out[q_start:q_end] = result.squeeze(0).transpose(0, 1)
    return out


def _paged_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Paged-KV oracle used only while autotuning."""
    del max_seqlen_q, max_seqlen_k
    cu_q = cu_seqlens_q.tolist()
    lengths_k = seqused_k.tolist()
    page_size = k.size(1)
    out = torch.empty_like(q)
    for request, (q_start, q_end, seqlen_k) in enumerate(
        zip(cu_q[:-1], cu_q[1:], lengths_k)
    ):
        physical_blocks = block_table[
            request, : (seqlen_k + page_size - 1) // page_size
        ].long()
        k_seq = k.index_select(0, physical_blocks).flatten(0, 1)[:seqlen_k]
        v_seq = v.index_select(0, physical_blocks).flatten(0, 1)[:seqlen_k]
        q_seq = q[q_start:q_end]
        seqlen_q = q_end - q_start
        mask = None
        if causal:
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            mask = col <= row + seqlen_k - seqlen_q
        result = torch.nn.functional.scaled_dot_product_attention(
            q_seq.transpose(0, 1).unsqueeze(0),
            k_seq.transpose(0, 1).unsqueeze(0),
            v_seq.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            scale=sm_scale,
            enable_gqa=q.size(1) != k.size(2),
        )
        out[q_start:q_end] = result.squeeze(0).transpose(0, 1)
    return out


def _attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 SDPA gradients used while autotuning the backward kernel."""
    with torch.enable_grad():
        q_ref = q.float().detach().requires_grad_()
        k_ref = k.float().detach().requires_grad_()
        v_ref = v.float().detach().requires_grad_()
        out_ref = _sdpa_reference(q_ref, k_ref, v_ref, sm_scale)
        grads = torch.autograd.grad(
            out_ref,
            (q_ref, k_ref, v_ref),
            grad_out.float(),
        )
        return (
            grads[0].to(q.dtype),
            grads[1].to(k.dtype), grads[2].to(v.dtype),
        )


@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=_sdpa_reference,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def attention_bshd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Non-causal attention over FlashAttention's ``[batch, seq, heads, dim]`` layout.

    ``k``/``v`` may carry fewer heads than ``q`` (grouped-query attention); each
    query head reads the key/value head ``h // (nheads_q // nheads_kv)``.
    """
    batch = q.size(0)
    m_dim = q.size(1)
    nheads_q = hl.specialize(q.size(2))
    head_dim = hl.specialize(q.size(3))
    n_dim = k.size(1)
    nheads_kv = hl.specialize(k.size(2))
    group = nheads_q // nheads_kv
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504088896340736
    # Scale the fp32 accumulator, not bf16 q: large runtime scales otherwise
    # lose enough precision to change the softmax distribution materially.
    for b, h in hl.grid([batch, nheads_q]):
        h_kv = h // group
        for tile_m in hl.tile(m_dim):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 1.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            q_blk = q[b, tile_m, h, :]
            for tile_n in hl.tile(n_dim):
                k_blk = k[b, tile_n, h_kv, :]
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                qk = qk - m_ij[:, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v_blk = v[b, tile_n, h_kv, :]
                acc = hl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                m_i = m_ij
            acc = acc / l_i[:, None]
            out[b, tile_m, h, :] = acc.to(out.dtype)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=_causal_sdpa_reference,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def causal_attention_bshd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Bottom-right causal attention over ``[batch, seq, heads, dim]``.

    Query row ``i`` attends through key ``i + seqlen_k - seqlen_q``, matching
    FlashAttention for both equal and unequal sequence lengths. Equal-length
    specializations retain the triangular key-loop optimization; unequal
    specializations mask the full key range.
    """
    batch = q.size(0)
    m_dim = hl.specialize(q.size(1))
    nheads_q = hl.specialize(q.size(2))
    head_dim = hl.specialize(q.size(3))
    n_dim = hl.specialize(k.size(1))
    nheads_kv = hl.specialize(k.size(2))
    group = nheads_q // nheads_kv
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504088896340736
    causal_offset = n_dim - m_dim
    for b, h in hl.grid([batch, nheads_q]):
        h_kv = h // group
        for tile_m in hl.tile(m_dim):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            q_blk = q[b, tile_m, h, :]
            # Keep the equal-length triangular bound as a direct tile value.
            # Combining it algebraically with the unequal-length bound can
            # make Helion lower the first tile's stop to zero.
            if m_dim == n_dim:
                key_stop = tile_m.end
            else:
                key_stop = n_dim
            for tile_n in hl.tile(0, key_stop):
                k_blk = k[b, tile_n, h_kv, :]
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                score_mask = tile_n.index[None, :] <= (
                    tile_m.index + causal_offset
                )[:, None]
                qk = torch.where(score_mask, qk, float("-inf"))
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                has_key = m_ij != float("-inf")
                p = torch.exp2(
                    torch.where(score_mask, qk - m_ij[:, None], float("-inf"))
                )
                alpha = torch.where(has_key, torch.exp2(m_i - m_ij), 1.0)
                l_i = l_i * alpha + torch.sum(p, -1)
                acc = acc * alpha[:, None]
                v_blk = v[b, tile_n, h_kv, :]
                acc = hl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                m_i = m_ij
            result = torch.where(
                (l_i > 0)[:, None],
                acc / torch.where(l_i > 0, l_i, 1.0)[:, None],
                0.0,
            )
            out[b, tile_m, h, :] = result.to(out.dtype)
    return out


@helion.kernel(
    config=helion.Config(
        block_sizes=[128],
        num_warps=4,
        num_stages=3,
        pid_type="flat",
        loop_orders=[[1, 2, 0]],
    ),
    static_shapes=True,
)
def decode_attention_bshd_split_kv_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    partial_acc: torch.Tensor,
    partial_stats: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent online-softmax partials for each GQA head group."""
    batch = q.size(0)
    nheads_q = hl.specialize(q.size(2))
    head_dim = hl.specialize(q.size(3))
    n_dim = hl.specialize(k.size(1))
    nheads_kv = hl.specialize(k.size(2))
    group = nheads_q // nheads_kv
    num_splits = hl.specialize(partial_acc.size(2))
    split_size = (n_dim + num_splits - 1) // num_splits
    qk_scale = sm_scale * 1.44269504088896340736

    for b, h_kv, split in hl.grid([batch, nheads_kv, num_splits]):
        # Process the complete GQA group together so its query heads share each
        # K/V load. More KV splits preserve a full launch grid for small batches.
        for tile_h in hl.tile(
            h_kv * group,
            (h_kv + 1) * group,
            block_size=group,
        ):
            m_i = hl.full([tile_h], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_h], dtype=torch.float32)
            acc = hl.zeros([tile_h, head_dim], dtype=torch.float32)
            q_rows = q[b, 0, tile_h, :]
            for tile_n in hl.tile(
                split * split_size,
                min((split + 1) * split_size, n_dim),
            ):
                k_blk = k[b, tile_n, h_kv, :]
                qk = hl.dot(q_rows, k_blk.T, out_dtype=torch.float32) * qk_scale
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                p = torch.exp2(qk - m_ij[:, None])
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + torch.sum(p, -1)
                acc = acc * alpha[:, None]
                v_blk = v[b, tile_n, h_kv, :]
                acc = hl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                m_i = m_ij
            partial_acc[b, tile_h, split, :, :] = acc[:, None, :]
            partial_stats[b, tile_h, 0, split, :] = m_i[:, None]
            partial_stats[b, tile_h, 1, split, :] = l_i[:, None]
    return partial_acc, partial_stats


@helion.kernel(
    config=helion.Config(
        num_warps=1,
        num_stages=3,
        pid_type="flat",
    ),
    static_shapes=True,
)
def decode_attention_bshd_split_kv_combine(
    partial_acc: torch.Tensor,
    partial_stats: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Renormalize and combine the per-split softmax numerators."""
    batch = out.size(0)
    nheads_q = hl.specialize(out.size(2))
    for b, h in hl.grid([batch, nheads_q]):
        split_max = partial_stats[b, h, 0, :, :]
        global_max = torch.amax(split_max, 0)
        renormalize = torch.exp2(split_max - global_max)
        denominator = torch.sum(
            partial_stats[b, h, 1, :, :] * renormalize,
            0,
        )
        numerator = torch.sum(
            partial_acc[b, h, :, :, :] * renormalize[:, :, None],
            0,
        )
        out[b, :, h, :] = (numerator / denominator[:, None]).to(out.dtype)
    return out


def decode_attention_bshd_split_kv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Single-query decode with sixteen KV-sequence programs per GQA head group.

    Long-cache decode otherwise exposes only ``batch * nheads_q`` programs.
    Each split computes an online-softmax numerator and statistics for its KV
    range, then a small second kernel renormalizes and combines the partials.
    A single bottom-right-aligned query can see the entire cache, so causal and
    non-causal decode have the same unmasked computation here.
    """
    num_splits = 16
    partial_acc = torch.empty(
        (q.size(0), q.size(2), num_splits, 1, q.size(3)),
        dtype=torch.float32,
        device=q.device,
    )
    partial_stats = torch.empty(
        (q.size(0), q.size(2), 2, num_splits, 1),
        dtype=torch.float32,
        device=q.device,
    )
    out = torch.empty_like(q)
    decode_attention_bshd_split_kv_partials(
        q,
        k,
        v,
        partial_acc,
        partial_stats,
        sm_scale,
    )
    return decode_attention_bshd_split_kv_combine(
        partial_acc,
        partial_stats,
        out,
    )


@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 128],
        num_warps=8,
        num_stages=3,
        pid_type="persistent_interleaved",
        num_sm_multiplier=1,
        # Use Hopper's descriptor path for K while retaining a masked block
        # pointer for V; making both descriptor loads exceeds shared memory.
        indexing=[
            "pointer",
            "tensor_descriptor",
            "block_ptr",
            "pointer",
            "pointer",
        ],
    ),
    static_shapes=True,
    autotune_baseline_fn=_causal_sdpa_reference,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def causal_attention_bshd_16k(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Fixed persistent causal attention for explicitly validated shapes."""
    batch = q.size(0)
    m_dim = hl.specialize(q.size(1))
    nheads_q = hl.specialize(q.size(2))
    head_dim = hl.specialize(q.size(3))
    n_dim = hl.specialize(k.size(1))
    nheads_kv = hl.specialize(k.size(2))
    group = nheads_q // nheads_kv
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504088896340736
    causal_offset = n_dim - m_dim
    # A one-CTA-per-SM persistent grid interleaves short and long causal rows
    # instead of assigning all query tiles to only sixteen batch/head CTAs.
    for tile_b, tile_h, tile_m in hl.tile(
        [batch, nheads_q, m_dim], block_size=[1, 1, None]
    ):
        b = tile_b.begin
        h = tile_h.begin
        h_kv = h // group
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([tile_m], dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_blk = q[b, tile_m, h, :]
        if m_dim == n_dim:
            key_stop = tile_m.end
        else:
            key_stop = n_dim
        for tile_n in hl.tile(0, key_stop):
            k_blk = k[b, tile_n, h_kv, :]
            qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
            score_mask = tile_n.index[None, :] <= (
                tile_m.index + causal_offset
            )[:, None]
            qk = torch.where(score_mask, qk, float("-inf"))
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            has_key = m_ij != float("-inf")
            p = torch.exp2(
                torch.where(score_mask, qk - m_ij[:, None], float("-inf"))
            )
            alpha = torch.where(has_key, torch.exp2(m_i - m_ij), 1.0)
            l_i = l_i * alpha + torch.sum(p, -1)
            acc = acc * alpha[:, None]
            v_blk = v[b, tile_n, h_kv, :]
            acc = hl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
            m_i = m_ij
        result = torch.where(
            (l_i > 0)[:, None],
            acc / torch.where(l_i > 0, l_i, 1.0)[:, None],
            0.0,
        )
        out[b, tile_m, h, :] = result.to(out.dtype)
    return out


@helion.kernel(
    static_shapes=False,
    autotune_effort="quick",
    autotune_baseline_fn=_varlen_sdpa_reference,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def varlen_attention_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: hl.constexpr,
    max_seqlen_k: hl.constexpr,
    sm_scale: float,
    causal: hl.constexpr,
) -> torch.Tensor:
    """Attention over FlashAttention's packed ``[total, heads, dim]`` layout.

    The first tensor dimension is deliberately dynamic.  The batch size,
    maximum sequence lengths, head counts, head dimension, and masking mode
    remain compile-time specializations, while each invocation reads the
    actual per-sequence bounds from the two device-resident cumulative-length
    arrays.
    """
    batch = hl.specialize(cu_seqlens_q.size(0) - 1)
    max_q = max_seqlen_q
    max_k = max_seqlen_k
    nheads_q = hl.specialize(q.size(1))
    nheads_kv = hl.specialize(k.size(1))
    head_dim = hl.specialize(q.size(2))
    is_causal = causal
    group = nheads_q // nheads_kv
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504088896340736

    for b, h in hl.grid([batch, nheads_q]):
        h_kv = h // group
        q_start = cu_seqlens_q[b]
        q_end = cu_seqlens_q[b + 1]
        k_start = cu_seqlens_k[b]
        k_end = cu_seqlens_k[b + 1]
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
        for tile_m in hl.tile(max_q):
            valid_m = tile_m.index < seqlen_q
            q_index = q_start + tile_m.index
            q_blk = hl.load(
                q,
                [q_index, h, slice(None)],
                extra_mask=valid_m[:, None],
            )
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            for tile_n in hl.tile(max_k):
                valid_n = tile_n.index < seqlen_k
                k_index = k_start + tile_n.index
                k_blk = hl.load(
                    k,
                    [k_index, h_kv, slice(None)],
                    extra_mask=valid_n[:, None],
                )
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                score_mask = valid_m[:, None] & valid_n[None, :]
                if is_causal:
                    causal_bound = tile_m.index + seqlen_k - seqlen_q
                    score_mask = score_mask & (
                        tile_n.index[None, :] <= causal_bound[:, None]
                    )
                qk = torch.where(score_mask, qk, float("-inf"))
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                has_key = m_ij != float("-inf")
                p = torch.exp2(
                    torch.where(score_mask, qk - m_ij[:, None], float("-inf"))
                )
                alpha = torch.where(
                    has_key,
                    torch.exp2(m_i - m_ij),
                    1.0,
                )
                l_i = l_i * alpha + torch.sum(p, -1)
                acc = acc * alpha[:, None]
                v_blk = hl.load(
                    v,
                    [k_index, h_kv, slice(None)],
                    extra_mask=valid_n[:, None],
                )
                acc = hl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                m_i = m_ij
            result = torch.where(
                (l_i > 0)[:, None],
                acc / torch.where(l_i > 0, l_i, 1.0)[:, None],
                0.0,
            )
            hl.store(
                out,
                [q_index, h, slice(None)],
                result.to(out.dtype),
                extra_mask=valid_m[:, None],
            )
    return out


@helion.kernel(
    static_shapes=False,
    autotune_effort="quick",
    autotune_baseline_fn=_paged_sdpa_reference,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def paged_attention_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: hl.constexpr,
    max_seqlen_k: hl.constexpr,
    sm_scale: float,
    causal: hl.constexpr,
) -> torch.Tensor:
    """Attention over packed queries and a block-table-addressed paged KV cache.

    Query token totals, physical cache allocation, block-table strides, and
    per-request cache lengths remain dynamic.  The generated specialization
    fixes the batch, maximum lengths, head geometry, page size, dtype, and
    masking mode.  Logical key positions are translated through ``block_table``
    before every K/V load, so physical pages need not be contiguous or ordered.
    """
    batch = hl.specialize(cu_seqlens_q.size(0) - 1)
    max_q = max_seqlen_q
    max_k = max_seqlen_k
    nheads_q = hl.specialize(q.size(1))
    nheads_kv = hl.specialize(k.size(2))
    head_dim = hl.specialize(q.size(2))
    page_size = hl.specialize(k.size(1))
    max_blocks = (max_k + page_size - 1) // page_size
    is_causal = causal
    group = nheads_q // nheads_kv
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504088896340736

    for b, h in hl.grid([batch, nheads_q]):
        h_kv = h // group
        q_start = cu_seqlens_q[b]
        q_end = cu_seqlens_q[b + 1]
        seqlen_q = q_end - q_start
        seqlen_k = seqused_k[b]
        for tile_m in hl.tile(max_q):
            valid_m = tile_m.index < seqlen_q
            q_index = q_start + tile_m.index
            q_blk = hl.load(
                q,
                [q_index, h, slice(None)],
                extra_mask=valid_m[:, None],
            )
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            # One tile is exactly one cache page.  Keeping the physical block
            # index scalar lets Helion emit a native four-dimensional load
            # with all runtime strides intact; flattening block/page would
            # copy vLLM's split K/V cache views.
            for tile_block in hl.tile(max_blocks, block_size=1):
                key_positions = (
                    tile_block.begin * page_size + hl.arange(page_size)
                )
                valid_n = key_positions < seqlen_k
                logical_block = tile_block.begin
                physical_block = block_table[b, logical_block]
                k_page = hl.load(
                    k,
                    [physical_block, slice(None), h_kv, slice(None)],
                    extra_mask=valid_n[:, None],
                )
                k_blk = k_page.reshape(page_size, head_dim)
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                score_mask = valid_m[:, None] & valid_n[None, :]
                if is_causal:
                    causal_bound = tile_m.index + seqlen_k - seqlen_q
                    score_mask = score_mask & (
                        key_positions[None, :] <= causal_bound[:, None]
                    )
                qk = torch.where(score_mask, qk, float("-inf"))
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                has_key = m_ij != float("-inf")
                p = torch.exp2(
                    torch.where(score_mask, qk - m_ij[:, None], float("-inf"))
                )
                alpha = torch.where(
                    has_key,
                    torch.exp2(m_i - m_ij),
                    1.0,
                )
                l_i = l_i * alpha + torch.sum(p, -1)
                acc = acc * alpha[:, None]
                v_page = hl.load(
                    v,
                    [physical_block, slice(None), h_kv, slice(None)],
                    extra_mask=valid_n[:, None],
                )
                v_blk = v_page.reshape(page_size, head_dim)
                acc = hl.dot(p.to(v_blk.dtype), v_blk, acc=acc)
                m_i = m_ij
            result = torch.where(
                (l_i > 0)[:, None],
                acc / torch.where(l_i > 0, l_i, 1.0)[:, None],
                0.0,
            )
            hl.store(
                out,
                [q_index, h, slice(None)],
                result.to(out.dtype),
                extra_mask=valid_m[:, None],
            )
    return out


@helion.kernel(
    static_shapes=True,
    dot_precision="ieee",
    autotune_effort="quick",
    # Grid-wide barriers require every CTA to be resident at once.  One CTA
    # per SM is portable to Ampere/Ada as well as Hopper; allowing Helion to
    # tune this above 1 can produce a cooperative grid that those GPUs reject.
    autotune_config_overrides={"num_sm_multiplier": 1},
    autotune_baseline_fn=_attention_backward_reference,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def attention_backward_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-causal MHA backward over ``[batch, seq, heads, dim]`` tensors.

    This follows the recompute strategy used by FlashAttention backward.  A
    first pass recreates the per-query log-sum-exp values that the existing
    forward-only modules do not expose.  A second pass computes the softmax
    Jacobian's row correction in fp32, and the final two passes produce dQ,
    then dK/dV, without materializing the quadratic attention matrix.

    The first checked-in specialization intentionally covers MHA with equal
    query/key lengths only.  The generator rejects causal, cross-attention, and
    GQA shapes until their masking and reduction rules are implemented here.
    """
    batch = q.size(0)
    m_dim = q.size(1)
    n_dim = k.size(1)
    nheads = hl.specialize(q.size(2))
    head_dim = hl.specialize(q.size(3))
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    lse = torch.empty(
        (batch, m_dim, nheads),
        dtype=torch.float32,
        device=q.device,
    )
    delta = torch.empty(
        (batch, m_dim, nheads),
        dtype=torch.float32,
        device=q.device,
    )
    qk_scale = sm_scale * 1.44269504088896340736
    # Keep the same fp32 logit scaling used by the matching forward kernel.

    # Recreate the forward log-sum-exp in base 2 once.  Both gradient passes
    # can then reconstruct each probability tile independently.
    for b, h in hl.grid([batch, nheads]):
        for tile_m in hl.tile(m_dim):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 1.0, dtype=torch.float32)
            q_blk = q[b, tile_m, h, :]
            for tile_n in hl.tile(n_dim):
                k_blk = k[b, tile_n, h, :]
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                p = torch.exp2(qk - m_ij[:, None])
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + torch.sum(p, -1)
                m_i = m_ij
            lse[b, tile_m, h] = m_i + torch.log2(l_i)

    hl.barrier()

    # Recompute delta = sum(p * dP) in fp32.  Deriving it from the bf16 forward
    # output (sum(out * grad_out)) amplifies output rounding at larger scales.
    for b, h in hl.grid([batch, nheads]):
        for tile_m in hl.tile(m_dim):
            q_blk = q[b, tile_m, h, :]
            grad_out_blk = grad_out[b, tile_m, h, :]
            lse_blk = lse[b, tile_m, h]
            delta_acc = hl.zeros([tile_m], dtype=torch.float32)
            for tile_n in hl.tile(n_dim):
                k_blk = k[b, tile_n, h, :]
                v_blk = v[b, tile_n, h, :]
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                p = torch.exp2(qk - lse_blk[:, None])
                dp = hl.dot(grad_out_blk, v_blk.T, out_dtype=torch.float32)
                delta_acc = delta_acc + torch.sum(p * dp, -1)
            delta[b, tile_m, h] = delta_acc

    hl.barrier()

    # dQ owns query-row tiles, so no atomics are needed.
    for b, h in hl.grid([batch, nheads]):
        for tile_m in hl.tile(m_dim):
            q_blk = q[b, tile_m, h, :]
            grad_out_blk = grad_out[b, tile_m, h, :]
            delta_blk = delta[b, tile_m, h]
            lse_blk = lse[b, tile_m, h]
            dq_acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            for tile_n in hl.tile(n_dim):
                k_blk = k[b, tile_n, h, :]
                v_blk = v[b, tile_n, h, :]
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                p = torch.exp2(qk - lse_blk[:, None])
                dp = hl.dot(grad_out_blk, v_blk.T, out_dtype=torch.float32)
                ds = p * (dp - delta_blk[:, None])
                dq_acc = hl.dot(ds, k_blk.to(torch.float32), acc=dq_acc)
            dq[b, tile_m, h, :] = (dq_acc * sm_scale).to(dq.dtype)

    hl.barrier()

    # dK and dV own key-row tiles and stream over all query rows.
    for b, h in hl.grid([batch, nheads]):
        for tile_n in hl.tile(n_dim):
            k_blk = k[b, tile_n, h, :]
            v_blk = v[b, tile_n, h, :]
            dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
            dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
            for tile_m in hl.tile(m_dim):
                q_blk = q[b, tile_m, h, :]
                grad_out_blk = grad_out[b, tile_m, h, :]
                delta_blk = delta[b, tile_m, h]
                lse_blk = lse[b, tile_m, h]
                qk = hl.dot(q_blk, k_blk.T, out_dtype=torch.float32) * qk_scale
                p = torch.exp2(qk - lse_blk[:, None])
                dp = hl.dot(grad_out_blk, v_blk.T, out_dtype=torch.float32)
                ds = p * (dp - delta_blk[:, None])
                dk_acc = hl.dot(ds.T, q_blk.to(torch.float32), acc=dk_acc)
                dv_acc = hl.dot(
                    p.T.to(grad_out_blk.dtype), grad_out_blk, acc=dv_acc
                )
            dk[b, tile_n, h, :] = (dk_acc * sm_scale).to(dk.dtype)
            dv[b, tile_n, h, :] = dv_acc.to(dv.dtype)
    return dq, dk, dv
