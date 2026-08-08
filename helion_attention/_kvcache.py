"""Runtime helpers for the dense, read-only KV-cache entry point."""

from __future__ import annotations

import torch


def single_token_softmax_lse(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Return natural-log softmax normalizers for one-token GQA queries.

    The public cache path validates that the query length is one and that the
    dense cache is full before calling this helper.  Grouping query heads by
    their KV head avoids expanding the cache for GQA.  The only attention-sized
    temporary is ``[batch, nheads_q, cache_len]``, so both work and storage are
    linear in the cache length.
    """
    batch, _, nheads_q, head_dim = q.shape
    nheads_kv = k_cache.shape[2]
    queries_per_kv = nheads_q // nheads_kv

    grouped_q = q[:, 0].reshape(
        batch, nheads_kv, queries_per_kv, head_dim
    ).float()
    grouped_k_t = k_cache.float().permute(0, 2, 3, 1)
    scores = torch.matmul(grouped_q, grouped_k_t) * softmax_scale
    return torch.logsumexp(scores, dim=-1).reshape(batch, nheads_q, 1)
