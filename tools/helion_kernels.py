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
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=sm_scale,
        is_causal=True,
        enable_gqa=q.size(2) != k.size(2),
    ).transpose(1, 2)


def _decode_sdpa_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Bottom-right causal attention when the single query is the newest token."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=sm_scale,
        enable_gqa=q.size(2) != k.size(2),
    ).transpose(1, 2)


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
    for b, h in hl.grid([batch, nheads_q]):
        h_kv = h // group
        for tile_m in hl.tile(m_dim):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 1.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            q_blk = q[b, tile_m, h, :]
            for tile_n in hl.tile(n_dim):
                k_blk = k[b, tile_n, h_kv, :]
                qk = hl.dot(q_blk * qk_scale, k_blk.T, out_dtype=torch.float32)
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
    """Causal attention over ``[batch, seq, heads, dim]`` with ``seqlen_q == seqlen_k``.

    The key loop stops at the diagonal instead of masking a full row of blocks,
    so the fully-masked upper triangle is never computed at all.
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
    for b, h in hl.grid([batch, nheads_q]):
        h_kv = h // group
        for tile_m in hl.tile(m_dim):
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 1.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            q_blk = q[b, tile_m, h, :]
            for tile_n in hl.tile(0, tile_m.end):
                k_blk = k[b, tile_n, h_kv, :]
                qk = hl.dot(q_blk * qk_scale, k_blk.T, out_dtype=torch.float32)
                qk = torch.where(
                    tile_m.index[:, None] >= tile_n.index[None, :], qk, float("-inf")
                )
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
    autotune_baseline_fn=_decode_sdpa_reference,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def decode_attention_bshd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Single-token decode over a full ``[batch, cache, heads, dim]`` KV cache.

    FlashAttention aligns unequal causal masks to the bottom right. With one
    query representing the newest token, every cache position is visible, so
    this is the decode-specific form of causal attention without a mask.
    """
    batch = q.size(0)
    nheads_q = hl.specialize(q.size(2))
    head_dim = hl.specialize(q.size(3))
    n_dim = k.size(1)
    nheads_kv = hl.specialize(k.size(2))
    group = nheads_q // nheads_kv
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504088896340736
    for b, h in hl.grid([batch, nheads_q]):
        h_kv = h // group
        m_i = hl.full([1], float("-inf"), dtype=torch.float32)
        l_i = hl.full([1], 1.0, dtype=torch.float32)
        acc = hl.zeros([1, head_dim], dtype=torch.float32)
        q_row = q[b, :, h, :]
        for tile_n in hl.tile(n_dim):
            k_blk = k[b, tile_n, h_kv, :]
            qk = hl.dot(q_row, k_blk.T, out_dtype=torch.float32)
            qk = qk * qk_scale
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
        out[b, :, h, :] = (acc / l_i[:, None]).to(out.dtype)
    return out


KERNELS: dict[bool, object] = {
    False: attention_bshd,
    True: causal_attention_bshd,
}


def select_kernel(*, causal: bool, seqlen_q: int, seqlen_k: int) -> object:
    """Select the source kernel whose masking matches the requested shape."""
    if causal and seqlen_q != seqlen_k:
        if seqlen_q != 1:
            raise ValueError(
                "unequal causal lengths are supported only for seqlen_q=1 decode"
            )
        return decode_attention_bshd
    return KERNELS[causal]
