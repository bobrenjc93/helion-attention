"""Single-launch generic packed and paged attention for the vLLM adapter."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _varlen_attention_kernel(
    q,
    k,
    v,
    q_v,
    out,
    lse_out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    block_table,
    max_seqlen_q_tensor,
    max_seqlen_k_tensor,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes,
    sinks,
    cp_tot_seqused_k,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_qvt,
    stride_qvh,
    stride_qvd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lseh,
    stride_lset,
    stride_bt_b,
    stride_bt_block,
    stride_qs_b,
    stride_qs_h,
    stride_ks_b,
    stride_ks_h,
    stride_vs_b,
    stride_vs_h,
    stride_alibi_b,
    stride_alibi_h,
    stride_sink_b,
    stride_sink_h,
    softmax_scale,
    softcap_ptr,
    window_left,
    window_right,
    max_seqlen_q_value,
    max_seqlen_k_value,
    query_blocks,
    NHEADS_Q: tl.constexpr,
    NHEADS_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    PAGED: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_Q_V: tl.constexpr,
    HAS_Q_DESCALE: tl.constexpr,
    HAS_K_DESCALE: tl.constexpr,
    HAS_V_DESCALE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_CP_TOTAL: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    STORE_LSE: tl.constexpr,
    SHIFT_FA2_LSE: tl.constexpr,
    HAS_DYNAMIC_MAX_Q: tl.constexpr,
    HAS_DYNAMIC_MAX_K: tl.constexpr,
    FA_VERSION_2: tl.constexpr,
    INPUT_FP16: tl.constexpr,
    CP_WORLD_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    # Keep all logical work in CUDA's large X grid dimension.  The Y and Z
    # dimensions are limited to 65,535, which valid large batches can exceed
    # once multiplied by the number of query heads.
    work_id = tl.program_id(0)
    query_block = work_id % query_blocks
    batch_head = work_id // query_blocks
    batch = batch_head // NHEADS_Q
    head_q = batch_head % NHEADS_Q
    group_size = NHEADS_Q // NHEADS_KV
    head_kv = head_q // group_size

    if HAS_DYNAMIC_MAX_Q:
        real_max_seqlen_q = tl.load(max_seqlen_q_tensor)
    else:
        real_max_seqlen_q = max_seqlen_q_value
    if HAS_DYNAMIC_MAX_K:
        real_max_seqlen_k = tl.load(max_seqlen_k_tensor)
    else:
        real_max_seqlen_k = max_seqlen_k_value
    if FA_VERSION_2:
        window_left = tl.where(
            window_left >= real_max_seqlen_k, -1, window_left
        )
        window_right = tl.where(
            window_right >= real_max_seqlen_k, -1, window_right
        )
    else:
        window_left = tl.where(
            window_left >= real_max_seqlen_k - 1, -1, window_left
        )
        window_right = tl.where(
            window_right >= real_max_seqlen_q - 1, -1, window_right
        )

    q_start = tl.load(cu_seqlens_q + batch)
    q_stop = tl.load(cu_seqlens_q + batch + 1)
    seqlen_q = q_stop - q_start
    if PAGED:
        packed_k_start = 0
        seqlen_k = tl.load(seqused_k + batch)
    else:
        packed_k_start = tl.load(cu_seqlens_k + batch)
        packed_k_stop = tl.load(cu_seqlens_k + batch + 1)
        seqlen_k = packed_k_stop - packed_k_start
    if HAS_CP_TOTAL:
        total_seqlen_k = tl.load(cp_tot_seqused_k + batch)
    else:
        total_seqlen_k = seqlen_k

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    valid_m = offs_m < seqlen_q
    q_indices = q_start + offs_m

    q_values = tl.load(
        q
        + q_indices[:, None] * stride_qt
        + head_q * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=valid_m[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    if HAS_Q_DESCALE:
        q_scale = tl.load(q_descale + batch * stride_qs_b + head_q * stride_qs_h)
        q_values *= q_scale

    if HAS_Q_V:
        qv_values = tl.load(
            q_v
            + q_indices[:, None] * stride_qvt
            + head_q * stride_qvh
            + offs_dv[None, :] * stride_qvd,
            mask=valid_m[:, None] & (offs_dv[None, :] < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)
        if HAS_Q_DESCALE:
            qv_values *= q_scale

    aligned_query = offs_m + total_seqlen_k - seqlen_q
    if HAS_SINK:
        sink = tl.load(sinks + batch * stride_sink_b + head_q * stride_sink_h)
        running_max = tl.full([BLOCK_M], sink, tl.float32)
        running_sum = tl.full([BLOCK_M], 1.0, tl.float32)
    else:
        running_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    for key_start in tl.range(0, seqlen_k, BLOCK_N):
        offs_n = key_start + tl.arange(0, BLOCK_N)
        valid_n = offs_n < seqlen_k
        global_key_positions = offs_n * CP_WORLD_SIZE + CP_RANK
        valid_storage = valid_n & (global_key_positions < total_seqlen_k)
        if PAGED:
            logical_blocks = offs_n // PAGE_SIZE
            physical_blocks = tl.load(
                block_table
                + batch * stride_bt_b
                + logical_blocks * stride_bt_block,
                mask=valid_storage,
                other=0,
            )
            physical_blocks = tl.maximum(
                0, tl.minimum(physical_blocks, NUM_BLOCKS - 1)
            )
            page_offsets = offs_n % PAGE_SIZE
            key_row_offsets = (
                physical_blocks * stride_kb + page_offsets * stride_ks
            )
            value_row_offsets = (
                physical_blocks * stride_vb + page_offsets * stride_vs
            )
        else:
            key_row_offsets = (packed_k_start + offs_n) * stride_ks
            value_row_offsets = (packed_k_start + offs_n) * stride_vs

        key_values = tl.load(
            k
            + key_row_offsets[:, None]
            + head_kv * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=valid_storage[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        if HAS_K_DESCALE:
            key_scale = tl.load(
                k_descale + batch * stride_ks_b + head_kv * stride_ks_h
            )
            key_values *= key_scale

        if INPUT_FP16:
            scores = tl.dot(
                q_values.to(tl.float16),
                tl.trans(key_values.to(tl.float16)),
                out_dtype=tl.float32,
            )
        else:
            scores = tl.dot(
                q_values.to(tl.bfloat16),
                tl.trans(key_values.to(tl.bfloat16)),
                out_dtype=tl.float32,
            )

        value_values = tl.load(
            v
            + value_row_offsets[:, None]
            + head_kv * stride_vh
            + offs_dv[None, :] * stride_vd,
            mask=valid_storage[:, None] & (offs_dv[None, :] < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)
        if HAS_V_DESCALE:
            value_scale = tl.load(
                v_descale + batch * stride_vs_b + head_kv * stride_vs_h
            )
            value_values *= value_scale

        if HAS_Q_V:
            if INPUT_FP16:
                scores += tl.dot(
                    qv_values.to(tl.float16),
                    tl.trans(value_values.to(tl.float16)),
                    out_dtype=tl.float32,
                )
            else:
                scores += tl.dot(
                    qv_values.to(tl.bfloat16),
                    tl.trans(value_values.to(tl.bfloat16)),
                    out_dtype=tl.float32,
                )
        scores *= softmax_scale
        if HAS_SOFTCAP:
            # Load from device storage because Triton 3.0-3.3 narrow Python
            # float kernel arguments to fp32 even with an fp64 annotation.
            softcap = tl.load(softcap_ptr)
            # Use tanh directly: the sigmoid identity loses scores when
            # scores / softcap is small. Attention accumulation remains fp32.
            scores = (
                softcap * libdevice.tanh(scores.to(tl.float64) / softcap)
            ).to(tl.float32)

        score_mask = (
            valid_m[:, None]
            & valid_storage[None, :]
        )
        if CAUSAL:
            score_mask &= global_key_positions[None, :] <= aligned_query[:, None]
        score_mask &= (window_left < 0) | (
            global_key_positions[None, :] >= aligned_query[:, None] - window_left
        )
        score_mask &= (window_right < 0) | (
            global_key_positions[None, :] <= aligned_query[:, None] + window_right
        )
        if HAS_ALIBI:
            slope = tl.load(
                alibi_slopes + batch * stride_alibi_b + head_q * stride_alibi_h
            )
            distance = tl.abs(
                aligned_query[:, None] - global_key_positions[None, :]
            )
            scores -= slope * distance
        scores = tl.where(score_mask, scores, float("-inf"))

        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        has_key = next_max != float("-inf")
        old_weight = tl.where(has_key, tl.exp(running_max - next_max), 1.0)
        weights = tl.where(
            score_mask, tl.exp(scores - next_max[:, None]), 0.0
        )
        running_sum = old_weight * running_sum + tl.sum(weights, axis=1)
        accumulator *= old_weight[:, None]
        # Unused cache rows are loaded as zero, preventing 0 * NaN/Inf from
        # poisoning ragged requests.
        if INPUT_FP16:
            accumulator += tl.dot(
                weights.to(tl.float16),
                value_values.to(tl.float16),
                out_dtype=tl.float32,
            )
        else:
            accumulator += tl.dot(
                weights.to(tl.bfloat16),
                value_values.to(tl.bfloat16),
                out_dtype=tl.float32,
            )
        running_max = next_max

    has_mass = running_sum > 0.0
    safe_sum = tl.where(has_mass, running_sum, 1.0)
    result = tl.where(has_mass[:, None], accumulator / safe_sum[:, None], 0.0)
    tl.store(
        out
        + q_indices[:, None] * stride_ot
        + head_q * stride_oh
        + offs_dv[None, :] * stride_od,
        result,
        mask=valid_m[:, None] & (offs_dv[None, :] < VALUE_DIM),
    )
    if STORE_LSE:
        empty_lse = float("inf") if FA_VERSION_2 else float("-inf")
        lse = tl.where(has_mass, running_max + tl.log(safe_sum), empty_lse)
        if SHIFT_FA2_LSE and HAS_ALIBI:
            slope = tl.load(
                alibi_slopes + batch * stride_alibi_b + head_q * stride_alibi_h
            )
            shifted_lse = lse + slope * aligned_query
            lse = tl.where(window_left < 0, shifted_lse, lse)
        tl.store(
            lse_out + head_q * stride_lseh + q_indices * stride_lset,
            lse,
            mask=valid_m,
        )


def _head_metadata_strides(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    batch: int,
    heads: int,
    grouped_heads: int | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Return a tensor plus strides for scalar-per-head metadata."""
    if value is None:
        dummy = torch.empty(1, device=device)
        return dummy, 0, 0
    if not isinstance(value, torch.Tensor):
        raise TypeError("head metadata must be a torch.Tensor or None")
    if value.device != device:
        raise ValueError(f"head metadata must be on device {device}")
    if value.ndim == 0:
        return value.reshape(1, 1), 0, 0
    if value.ndim == 1:
        if value.numel() not in (1, heads, grouped_heads):
            raise ValueError("head metadata has an incompatible number of values")
        head_stride = 0 if value.numel() == 1 else value.stride(0)
        if grouped_heads is not None and value.numel() == grouped_heads:
            expanded = value.repeat_interleave(heads // grouped_heads)
            return expanded, 0, expanded.stride(0)
        return value, 0, head_stride
    if value.ndim == 2 and value.shape[0] in (1, batch):
        if value.shape[1] not in (1, heads, grouped_heads):
            raise ValueError("head metadata has an incompatible head dimension")
        selected = value
        if grouped_heads is not None and value.shape[1] == grouped_heads:
            selected = value.repeat_interleave(heads // grouped_heads, dim=1)
        return (
            selected,
            0 if selected.shape[0] == 1 else selected.stride(0),
            0 if selected.shape[1] == 1 else selected.stride(1),
        )
    raise ValueError("head metadata must be scalar, [heads], or [batch, heads]")


def _softcap_device_value(softcap: float, device: torch.device) -> torch.Tensor:
    """Transport a Python softcap without Triton scalar-argument narrowing."""
    return torch.tensor(softcap, device=device, dtype=torch.float64)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    cu_seqlens_k: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    block_table: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dynamic_max_seqlen_q: torch.Tensor | None,
    dynamic_max_seqlen_k: torch.Tensor | None,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    s_aux: torch.Tensor | None,
    q_v: torch.Tensor | None,
    cp_world_size: int,
    cp_rank: int,
    cp_tot_seqused_k: torch.Tensor | None,
    out: torch.Tensor | None,
    return_softmax_lse: bool,
    shift_fa2_lse: bool,
    fa_version: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch packed or paged varlen attention in one GPU kernel."""
    paged = block_table is not None
    if paged:
        assert seqused_k is not None
    else:
        assert cu_seqlens_k is not None
    batch = cu_seqlens_q.numel() - 1
    nheads_q = q.shape[1]
    nheads_kv = k.shape[-2]
    head_dim = q.shape[2]
    value_dim = v.shape[-1]
    if out is None:
        output_dtype = (
            torch.bfloat16 if str(q.dtype).startswith("torch.float8_") else q.dtype
        )
        out = torch.empty(
            (q.shape[0], nheads_q, value_dim), device=q.device, dtype=output_dtype
        )
    lse = torch.empty(
        (nheads_q, q.shape[0]) if return_softmax_lse else (1,),
        device=q.device,
        dtype=torch.float32,
    )
    if q.shape[0] == 0:
        return (out, lse) if return_softmax_lse else out

    q_v_arg = q if q_v is None else q_v
    q_scale, qsb, qsh = _head_metadata_strides(
        q_descale,
        device=q.device,
        batch=batch,
        heads=nheads_q,
        grouped_heads=nheads_kv,
    )
    k_scale, ksb, ksh = _head_metadata_strides(
        k_descale, device=q.device, batch=batch, heads=nheads_kv
    )
    v_scale, vsb, vsh = _head_metadata_strides(
        v_descale, device=q.device, batch=batch, heads=nheads_kv
    )
    alibi, alb, alh = _head_metadata_strides(
        alibi_slopes, device=q.device, batch=batch, heads=nheads_q
    )
    sinks, skb, skh = _head_metadata_strides(
        s_aux, device=q.device, batch=batch, heads=nheads_q
    )
    cp_tot_arg = (
        torch.empty(1, device=q.device, dtype=torch.int32)
        if cp_tot_seqused_k is None
        else cp_tot_seqused_k
    )
    cu_k_arg = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    seqused_arg = cu_seqlens_q if seqused_k is None else seqused_k
    block_table_arg = (
        cu_seqlens_q.reshape(1, -1) if block_table is None else block_table
    )
    max_seqlen_q_arg = (
        cu_seqlens_q
        if dynamic_max_seqlen_q is None
        else dynamic_max_seqlen_q
    )
    max_seqlen_k_arg = (
        cu_seqlens_q
        if dynamic_max_seqlen_k is None
        else dynamic_max_seqlen_k
    )
    has_softcap = softcap > 0.0
    # HAS_SOFTCAP makes the pointer unused for ordinary attention. Reusing q
    # there avoids a device allocation and preserves its existing launch path.
    softcap_arg = (
        _softcap_device_value(softcap, q.device) if has_softcap else q
    )

    if paged:
        k_strides = k.stride()
        v_strides = v.stride()
        page_size = k.shape[1]
        num_blocks = max(1, k.shape[0])
    else:
        k_strides = (0, k.stride(0), k.stride(1), k.stride(2))
        v_strides = (0, v.stride(0), v.stride(1), v.stride(2))
        page_size = 1
        num_blocks = 1

    block_d = max(16, triton.next_power_of_2(head_dim))
    block_dv = max(16, triton.next_power_of_2(value_dim))
    block_m = 16
    block_n = 64
    query_blocks = triton.cdiv(max_seqlen_q, block_m)
    grid = (query_blocks * batch * nheads_q,)
    with torch.cuda.device(q.device):
        _varlen_attention_kernel[grid](
            q,
            k,
            v,
            q_v_arg,
            out,
            lse,
            cu_seqlens_q,
            cu_k_arg,
            seqused_arg,
            block_table_arg,
            max_seqlen_q_arg,
            max_seqlen_k_arg,
            q_scale,
            k_scale,
            v_scale,
            alibi,
            sinks,
            cp_tot_arg,
            *q.stride(),
            *k_strides,
            *v_strides,
            *q_v_arg.stride(),
            *out.stride(),
            *lse.stride() if lse.ndim == 2 else (0, 0),
            *block_table_arg.stride(),
            qsb,
            qsh,
            ksb,
            ksh,
            vsb,
            vsh,
            alb,
            alh,
            skb,
            skh,
            softmax_scale,
            softcap_arg,
            window_size[0],
            window_size[1],
            max_seqlen_q,
            max_seqlen_k,
            query_blocks,
            NHEADS_Q=nheads_q,
            NHEADS_KV=nheads_kv,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            PAGE_SIZE=page_size,
            NUM_BLOCKS=num_blocks,
            PAGED=paged,
            CAUSAL=causal,
            HAS_Q_V=q_v is not None,
            HAS_Q_DESCALE=q_descale is not None,
            HAS_K_DESCALE=k_descale is not None,
            HAS_V_DESCALE=v_descale is not None,
            HAS_ALIBI=alibi_slopes is not None,
            HAS_SINK=s_aux is not None,
            HAS_CP_TOTAL=cp_tot_seqused_k is not None,
            HAS_SOFTCAP=has_softcap,
            STORE_LSE=return_softmax_lse,
            SHIFT_FA2_LSE=shift_fa2_lse,
            HAS_DYNAMIC_MAX_Q=dynamic_max_seqlen_q is not None,
            HAS_DYNAMIC_MAX_K=dynamic_max_seqlen_k is not None,
            FA_VERSION_2=fa_version == 2,
            INPUT_FP16=q.dtype == torch.float16,
            CP_WORLD_SIZE=cp_world_size,
            CP_RANK=cp_rank,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
            num_warps=4,
            num_stages=2,
        )
    return (out, lse) if return_softmax_lse else out


def paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dynamic_max_seqlen_q: torch.Tensor | None,
    dynamic_max_seqlen_k: torch.Tensor | None,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    s_aux: torch.Tensor | None,
    q_v: torch.Tensor | None,
    cp_world_size: int,
    cp_rank: int,
    cp_tot_seqused_k: torch.Tensor | None,
    out: torch.Tensor | None,
    return_softmax_lse: bool,
    shift_fa2_lse: bool,
    fa_version: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch generic paged attention in one GPU kernel."""
    return _attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=block_table,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dynamic_max_seqlen_q=dynamic_max_seqlen_q,
        dynamic_max_seqlen_k=dynamic_max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
        q_v=q_v,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
        out=out,
        return_softmax_lse=return_softmax_lse,
        shift_fa2_lse=shift_fa2_lse,
        fa_version=fa_version,
    )


def packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dynamic_max_seqlen_q: torch.Tensor | None,
    dynamic_max_seqlen_k: torch.Tensor | None,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    s_aux: torch.Tensor | None,
    q_v: torch.Tensor | None,
    cp_world_size: int,
    cp_rank: int,
    cp_tot_seqused_k: torch.Tensor | None,
    out: torch.Tensor | None,
    return_softmax_lse: bool,
    shift_fa2_lse: bool,
    fa_version: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch generic packed varlen attention in one GPU kernel."""
    return _attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=None,
        block_table=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dynamic_max_seqlen_q=dynamic_max_seqlen_q,
        dynamic_max_seqlen_k=dynamic_max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
        q_v=q_v,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
        out=out,
        return_softmax_lse=return_softmax_lse,
        shift_fa2_lse=shift_fa2_lse,
        fa_version=fa_version,
    )
