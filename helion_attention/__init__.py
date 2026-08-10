"""helion-attention: FlashAttention's API, served by Triton kernels.

The kernels in :mod:`helion_attention.kernels` were generated and autotuned by
`Helion <https://github.com/pytorch/helion>`_, but they are plain Triton by the
time they are checked in: importing this package pulls in ``torch`` and
``triton`` and nothing else.

Every entry point takes a required ``shape`` argument. Registered shapes use an
exact generated specialization, except evidenced SM90 fast paths that use
direct PyTorch Flash or cuDNN SDPA; compatible unregistered dense shapes, dense
ALiBi calls, BERT-base and causal GPT-2 diagnostics, one causal Llama-3 GQA
varlen inference profile, ALiBi on that profile and both shipped varlen
profiles, symmetric windows on the shipped noncausal varlen profile, and
diagnostics on the shipped causal varlen profile use a generic Triton forward
kernel. That runtime also provides
``softcap=50.0`` for one forward-only Gemma-2 profile, the shipped causal
varlen profile, and, only through the KV-cache API, read-only page-256 paged
decode. The same runtime exposes page-256 decode without softcap through the
core varlen API. The KV-cache adapter also uses the generic paged runtime for
ALiBi on both exposed page-16 profiles and page-256 decode. Read-only 4K dense
decode likewise uses the generic runtime for ALiBi while retaining its
generated specialization when slopes are omitted.
Default-scale SM90 training on the shipped encoder-training profile keeps its
generated forward values and uses raw PyTorch BSHD Flash gradients, falling
back to its generated backward when Flash is unavailable. Positive dropout on
that profile, the checked-in BERT-base encoder, one shipped causal GPT-2
profile, and the shipped asymmetric causal GQA chunked-prefill profile,
grad-enabled dense calls without a generated backward, both
full-length varlen profiles, and ragged causal attention use PyTorch SDPA
autograd. Full-length symmetric-window training on the shipped noncausal
varlen profile uses the same bounded SDPA bridge. Deterministic zero-dropout
BERT-base, causal GPT-2, and full-length causal varlen training use the direct
math operator without changing process-wide SDPA backend state. The explicit
shape validates these paths and makes specialization introspection independent
of fallback coverage.
"""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

import torch

from ._autograd import attention_autograd
from ._autograd import attention_with_flash_gradients
from ._registry import UnsupportedShapeError
from ._registry import available_paged_shapes
from ._registry import available_shapes
from ._registry import available_varlen_shapes
from ._registry import has_backward
from ._registry import has_kernel
from ._registry import has_paged_kernel
from ._registry import has_varlen_kernel
from ._registry import lookup
from ._registry import lookup_backward
from ._registry import lookup_paged
from ._registry import lookup_varlen
from ._sdpa import dense_attention_cudnn_default_scale
from ._sdpa import dense_attention_flash_default_scale
from ._sdpa import dense_attention_math_sdpa
from ._sdpa import dense_attention_sdpa
from ._shape import AttnShape
from ._shape import ShapeLike
from ._shape import check_paged_varlen_tensors
from ._shape import check_tensors
from ._shape import check_varlen_tensors
from ._shape import normalize_shape

__all__ = [
    "AttnShape",
    "UnsupportedShapeError",
    "available_paged_shapes",
    "available_shapes",
    "available_varlen_shapes",
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
    "is_shape_supported",
    "is_paged_shape_supported",
    "is_varlen_shape_supported",
]

__version__ = "0.1.0"

_CORE_PAGED_CHUNKED_PREFILL_SHAPE = (
    2,
    200,
    320,
    8,
    2,
    128,
    torch.bfloat16,
)
_CORE_PAGED_KVCACHE_SHAPE = (4, 1, 1024, 8, 2, 128, torch.bfloat16)
_CORE_PAGED_VARLEN_SHAPES = frozenset(
    {_CORE_PAGED_CHUNKED_PREFILL_SHAPE, _CORE_PAGED_KVCACHE_SHAPE}
)
_CORE_PAGED_KVCACHE_SHAPES = frozenset(
    {_CORE_PAGED_CHUNKED_PREFILL_SHAPE, _CORE_PAGED_KVCACHE_SHAPE}
)
_CORE_PAGED_GENERATED_PAGE_SIZE = 16
_CORE_PAGED_DECODE_PAGE_SIZES = frozenset({16, 256})
_PAGE256_PAGED_KVCACHE_SOFTCAP = 50.0
_GENERIC_DENSE_MAX_HEAD_DIM = 256
_INT32_MAX = 2**31 - 1
_ENCODER_TRAINING_KEY = "b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal"
_BERT_DIAGNOSTIC_KEY = "b16_sq512_sk512_hq12_hkv12_d64_bf16_noncausal"
_CAUSAL_DROPOUT_KEY = "b2_sq1024_sk1024_hq32_hkv32_d64_bf16_causal"
_CHUNKED_PREFILL_DROPOUT_KEY = (
    "b1_sq64_sk320_hq8_hkv2_d128_bf16_causal"
)
_GENERIC_DENSE_DIAGNOSTIC_KEYS = frozenset(
    {_BERT_DIAGNOSTIC_KEY, _CAUSAL_DROPOUT_KEY}
)
_DROPOUT_SDPA_KEYS = frozenset(
    {
        _ENCODER_TRAINING_KEY,
        _BERT_DIAGNOSTIC_KEY,
        _CAUSAL_DROPOUT_KEY,
        _CHUNKED_PREFILL_DROPOUT_KEY,
    }
)
_DENSE_DETERMINISTIC_BACKWARD_KEYS = frozenset(
    {_BERT_DIAGNOSTIC_KEY, _CAUSAL_DROPOUT_KEY}
)
_GEMMA2_SOFTCAP_KEY = "b1_sq4096_sk4096_hq16_hkv8_d256_bf16_causal"
_GEMMA2_SOFTCAP = 50.0
_FLASH_SDPA_FAST_PATH_KEY = (
    "b2_sq1024_sk1024_hq16_hkv16_d256_bf16_noncausal"
)
_CUDNN_SDPA_FAST_PATH_KEYS = frozenset(
    {
        "b1_sq4096_sk4096_hq32_hkv8_d128_bf16_causal",
        "b1_sq8192_sk8192_hq28_hkv4_d128_bf16_causal",
        "b2_sq8192_sk8192_hq16_hkv16_d128_bf16_causal",
        "b4_sq2048_sk2048_hq28_hkv4_d128_bf16_causal",
        "b4_sq4096_sk4096_hq32_hkv32_d128_bf16_causal",
    }
)
_LLAMA3_VARLEN_INFERENCE_KEY = (
    "varlen_b4_sq256_sk256_hq32_hkv8_d128_bf16_causal"
)
_VARLEN_ALIBI_KEYS = frozenset(
    {
        _LLAMA3_VARLEN_INFERENCE_KEY,
        "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_causal",
        "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal",
    }
)
_VARLEN_DIAGNOSTIC_KEY = "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_causal"
_VARLEN_SYMMETRIC_WINDOW_KEY = (
    "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal"
)
_VARLEN_SOFTCAP_KEY = _VARLEN_DIAGNOSTIC_KEY
_VARLEN_SOFTCAP = 50.0
_VARLEN_SDPA_BACKWARD_KEYS = frozenset(
    {
        "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_causal",
        "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_noncausal",
    }
)
_VARLEN_DETERMINISTIC_BACKWARD_KEY = (
    "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_causal"
)
_RAGGED_VARLEN_SDPA_BACKWARD_KEY = (
    "varlen_b8_sq512_sk512_hq16_hkv16_d64_bf16_causal"
)
_TENSOR_LENGTH_DENSE_KVCACHE_KEY = (
    "b1_sq1_sk16384_hq32_hkv8_d128_bf16_causal"
)
_EXPLICIT_SPLIT_DENSE_KVCACHE_PROFILE = (
    1,
    1,
    16384,
    32,
    8,
    128,
    torch.bfloat16,
)
_EXPLICIT_SPLIT_DENSE_KVCACHE_NUM_SPLITS = 16
_TWO_TOKEN_DENSE_KVCACHE_KEY = (
    "b1_sq2_sk1024_hq32_hkv8_d128_bf16_causal"
)
_FOUR_TOKEN_DENSE_KVCACHE_KEY = (
    "b1_sq4_sk1024_hq32_hkv8_d128_bf16_causal"
)
_DENSE_KVCACHE_ALIBI_PROFILE = (
    1,
    1,
    4096,
    32,
    8,
    128,
    torch.bfloat16,
)
_DIAGNOSTIC_DECODE_PROFILES = frozenset(
    {
        (1, 1, cache_length, 32, 8, 128, torch.bfloat16)
        for cache_length in (1024, 4096, 16384)
    }
)


@lru_cache(maxsize=None)
def _is_sm90(device: torch.device) -> bool:
    """Whether ``device`` is exactly an SM90 CUDA GPU."""
    return torch.cuda.get_device_capability(device) == (9, 0)


def _reject_unsupported(
    dropout_p: float,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_attn_probs: bool,
    *,
    allow_dropout: bool = False,
    allow_alibi: bool = False,
    allow_return_attn_probs: bool = False,
    allow_softcap_return_attn_probs: bool = False,
    allowed_softcap: float | None = None,
) -> float:
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, Real):
        raise TypeError("dropout_p must be a real number")
    probability = float(dropout_p)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout_p must satisfy 0.0 <= dropout_p < 1.0")
    if probability != 0.0 and not allow_dropout:
        raise NotImplementedError("dropout is not implemented; pass dropout_p=0.0")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("sliding-window attention is not implemented")
    has_softcap = softcap != 0.0
    if has_softcap and softcap != allowed_softcap:
        raise NotImplementedError(
            "softcap is implemented only as softcap=50.0 for the supported "
            "inference profiles"
        )
    if alibi_slopes is not None and not allow_alibi:
        raise NotImplementedError("ALiBi slopes are not implemented")
    if probability != 0.0 and alibi_slopes is not None:
        raise NotImplementedError(
            "dropout combined with ALiBi slopes is not implemented"
        )
    if probability != 0.0 and return_attn_probs:
        raise NotImplementedError(
            "return_attn_probs=True is not implemented with dropout"
        )
    if has_softcap and probability != 0.0:
        raise NotImplementedError("softcap combined with dropout is not implemented")
    if has_softcap and alibi_slopes is not None:
        raise NotImplementedError(
            "softcap combined with ALiBi slopes is not implemented"
        )
    if (
        has_softcap
        and return_attn_probs
        and not allow_softcap_return_attn_probs
    ):
        raise NotImplementedError(
            "return_attn_probs=True is not implemented with softcap"
        )
    if return_attn_probs and not allow_return_attn_probs:
        raise NotImplementedError("return_attn_probs is not implemented")
    return probability


def _supports_diagnostic_return(spec: AttnShape) -> bool:
    """Whether ``spec`` has a validated dense diagnostic path."""
    profile = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    # Causal and non-causal are equivalent for a bottom-right single-token
    # query, and the registry maps both modes to the same checked-in kernel.
    return (
        spec.key in _GENERIC_DENSE_DIAGNOSTIC_KEYS
        or profile in _DIAGNOSTIC_DECODE_PROFILES
    ) and has_kernel(spec)


def _contiguous_tensors_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Whether two validated contiguous tensors share any device memory."""
    first_start = first.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_start = second.data_ptr()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def _validate_kvcache_rotary(
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    q: torch.Tensor,
    spec: AttnShape,
    *,
    rotary_interleaved: bool,
    full_head_interleaved_only: bool = False,
) -> None:
    """Validate the narrow rotary contract for a final-cache append."""
    for name, tensor in (("rotary_cos", rotary_cos), ("rotary_sin", rotary_sin)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use torch.strided layout")
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have shape [seqlen_ro, rotary_dim / 2], got "
                f"{tuple(tensor.shape)}"
            )

    if rotary_cos.shape != rotary_sin.shape:
        raise ValueError(
            "rotary_cos and rotary_sin must have the same shape, got "
            f"{tuple(rotary_cos.shape)} and {tuple(rotary_sin.shape)}"
        )
    rotary_dim = rotary_cos.shape[1] * 2
    if full_head_interleaved_only and (
        not rotary_interleaved or rotary_dim != spec.head_dim
    ):
        raise NotImplementedError(
            "the two-token dense KV-cache update supports only full-head "
            "interleaved rotary embeddings; rotary_interleaved must be True "
            f"and rotary_dim must equal head_dim={spec.head_dim}, got "
            f"rotary_interleaved={rotary_interleaved} and rotary_dim={rotary_dim}"
        )
    if not rotary_interleaved and rotary_dim != spec.head_dim:
        raise NotImplementedError(
            "non-interleaved rotary embeddings are implemented only for "
            "full-head rotation; "
            f"rotary_dim must equal head_dim={spec.head_dim}, got {rotary_dim}"
        )
    supported_rotary_dims = {spec.head_dim}
    if spec.head_dim == 128:
        supported_rotary_dims.add(64)
    if rotary_dim not in supported_rotary_dims:
        expected = " or ".join(str(dim) for dim in sorted(supported_rotary_dims))
        raise NotImplementedError(
            "partial rotary dimensions are implemented only for rotary_dim=64 "
            "with head_dim=128; "
            f"rotary_dim must be {expected}, got {rotary_dim}"
        )
    if rotary_cos.shape[0] < spec.seqlen_k:
        raise ValueError(
            "rotary_cos and rotary_sin must contain the final cache position; "
            f"seqlen_ro must be at least {spec.seqlen_k}, got "
            f"{rotary_cos.shape[0]}"
        )

    for name, tensor in (("rotary_cos", rotary_cos), ("rotary_sin", rotary_sin)):
        if tensor.dtype != spec.dtype:
            raise ValueError(
                f"{name} has dtype {tensor.dtype} but q has dtype {spec.dtype}"
            )
        if not tensor.is_cuda:
            raise ValueError(
                f"{name} must be a CUDA tensor, got device {tensor.device}"
            )
        if tensor.device != q.device:
            raise ValueError(
                f"{name} must be on the same CUDA device as q, k_cache, and v_cache"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _apply_interleaved_rotary(
    tensor: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """Rotate adjacent prefix pairs and preserve the unrotated head tail."""
    rotary_dim = rotary_cos.shape[1] * 2
    rotary_prefix = tensor[..., :rotary_dim]
    pairs = rotary_prefix.float().reshape(
        *tensor.shape[:-1], rotary_dim // 2, 2
    )
    cos = rotary_cos[position].float()
    sin = rotary_sin[position].float()
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated_prefix = (
        torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        .flatten(-2)
        .to(dtype=tensor.dtype)
    )
    if rotary_dim == tensor.shape[-1]:
        return rotated_prefix
    return torch.cat((rotated_prefix, tensor[..., rotary_dim:]), dim=-1)


def _apply_consecutive_interleaved_rotary(
    tensor: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    start_position: int,
) -> torch.Tensor:
    """Rotate sequence tokens at consecutive cache positions."""
    return torch.cat(
        tuple(
            _apply_interleaved_rotary(
                tensor[:, token_index : token_index + 1],
                rotary_cos,
                rotary_sin,
                start_position + token_index,
            )
            for token_index in range(tensor.shape[1])
        ),
        dim=1,
    )


def _apply_neox_rotary(
    tensor: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """Rotate full-head split-half pairs in the GPT-NeoX layout."""
    half_dim = tensor.shape[-1] // 2
    first_half = tensor[..., :half_dim].float()
    second_half = tensor[..., half_dim:].float()
    cos = rotary_cos[position].float()
    sin = rotary_sin[position].float()
    return torch.cat(
        (
            first_half * cos - second_half * sin,
            first_half * sin + second_half * cos,
        ),
        dim=-1,
    ).to(dtype=tensor.dtype)


def _check_core_paged_varlen_spec(spec: AttnShape, page_size: int) -> None:
    """Require an exact paged profile exposed by the core varlen API."""
    requested = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    supported_page_size = (
        page_size == _CORE_PAGED_GENERATED_PAGE_SIZE
        or (
            requested == _CORE_PAGED_KVCACHE_SHAPE
            and page_size in _CORE_PAGED_DECODE_PAGE_SIZES
        )
    )
    if (
        requested not in _CORE_PAGED_VARLEN_SHAPES
        or (
            requested == _CORE_PAGED_CHUNKED_PREFILL_SHAPE
            and not spec.causal
        )
        or not supported_page_size
    ):
        raise UnsupportedShapeError(
            "flash_attn_varlen_func with block_table currently supports only "
            "these bf16 profiles:\n"
            "    batch=2 seqlen_q=200 seqlen_k=320 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True page_size=16\n"
            "    batch=4 seqlen_q=1 seqlen_k=1024 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True or causal=False page_size=16 or 256\n"
            f"got:\n    {spec.describe()}, page_size={page_size}"
        )


def _check_core_paged_kvcache_spec(spec: AttnShape, page_size: int) -> None:
    """Require an exact read-only specialization exposed by the KV-cache API."""
    requested = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    supported_page_size = (
        page_size == _CORE_PAGED_GENERATED_PAGE_SIZE
        or (
            requested == _CORE_PAGED_KVCACHE_SHAPE
            and page_size in _CORE_PAGED_DECODE_PAGE_SIZES
        )
    )
    if (
        requested not in _CORE_PAGED_KVCACHE_SHAPES
        or (requested == _CORE_PAGED_CHUNKED_PREFILL_SHAPE and not spec.causal)
        or not supported_page_size
    ):
        raise UnsupportedShapeError(
            "flash_attn_with_kvcache with block_table currently supports only "
            "these bf16 profiles:\n"
            "    batch=2 seqlen_q=200 seqlen_k=320 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True page_size=16\n"
            "    batch=4 seqlen_q=1 seqlen_k=1024 nheads=8 (GQA 8:2) "
            "head_dim=128 causal=True or causal=False page_size=16 or 256\n"
            f"got:\n    {spec.describe()}, page_size={page_size}"
        )


def _validate_generic_dense_layout(spec: AttnShape) -> tuple[int, int]:
    """Validate signed-int32 indices and return packed Q/K token totals."""
    if spec.head_dim > _GENERIC_DENSE_MAX_HEAD_DIM:
        raise UnsupportedShapeError(
            "no checked-in dense specialization exists for:\n"
            f"    {spec.describe()}\n"
            "the generic dense fallback supports head_dim <= "
            f"{_GENERIC_DENSE_MAX_HEAD_DIM}. To request a specialization, file "
            "an issue at https://github.com/bobrenjc93/helion-attention/issues"
        )

    total_q = spec.batch * spec.seqlen_q
    total_k = spec.batch * spec.seqlen_k
    if max(total_q, total_k) > _INT32_MAX:
        raise UnsupportedShapeError(
            "no checked-in dense specialization exists for:\n"
            f"    {spec.describe()}\n"
            "the generic dense fallback requires packed token offsets to fit "
            "in int32. To request a specialization, file an issue at "
            "https://github.com/bobrenjc93/helion-attention/issues"
        )

    # The packed kernel forms pointers with signed-int32 element arithmetic.
    # Token offsets alone are insufficient: token * stride can overflow first
    # for a large head count or dimension. Q and output share one layout, as do
    # K and V, so bound the maximum valid relative element offset for each.
    layout_numels = (
        ("Q/output", total_q * spec.nheads_q * spec.head_dim),
        ("K/V", total_k * spec.nheads_kv * spec.head_dim),
    )
    for layout, numel in layout_numels:
        max_element_offset = numel - 1
        if max_element_offset > _INT32_MAX:
            raise UnsupportedShapeError(
                "no checked-in dense specialization exists for:\n"
                f"    {spec.describe()}\n"
                "the generic dense fallback uses signed int32 element offsets, "
                f"but {layout} requires maximum offset {max_element_offset} "
                f"(limit {_INT32_MAX}). To request a specialization, file an "
                "issue at https://github.com/bobrenjc93/helion-attention/issues"
            )
    return total_q, total_k


def _validate_alibi_slopes(
    alibi_slopes: torch.Tensor,
    q: torch.Tensor,
    spec: AttnShape,
) -> None:
    """Validate FlashAttention's dense and varlen ALiBi metadata contract."""
    if not isinstance(alibi_slopes, torch.Tensor):
        raise TypeError("alibi_slopes must be a torch.Tensor or None")
    if alibi_slopes.layout != torch.strided:
        raise ValueError("alibi_slopes must use torch.strided layout")
    if alibi_slopes.dtype != torch.float32:
        raise ValueError("alibi_slopes must have dtype torch.float32")
    expected_shapes = ((spec.nheads_q,), (spec.batch, spec.nheads_q))
    if tuple(alibi_slopes.shape) not in expected_shapes:
        raise ValueError(
            "alibi_slopes must have shape [nheads_q] or [batch, nheads_q]; "
            f"expected {expected_shapes[0]} or {expected_shapes[1]}, got "
            f"{tuple(alibi_slopes.shape)}"
        )
    if not alibi_slopes.is_cuda:
        raise ValueError(
            f"alibi_slopes must be a CUDA tensor, got device {alibi_slopes.device}"
        )
    if alibi_slopes.device != q.device:
        raise ValueError("alibi_slopes must be on the same CUDA device as q, k, and v")
    if alibi_slopes.stride(-1) != 1:
        raise ValueError("alibi_slopes must be contiguous in its last dimension")


def _check_varlen_alibi_spec(spec: AttnShape) -> None:
    """Restrict generic varlen ALiBi dispatch to its validated profiles."""
    requested = f"varlen_{spec.key}"
    if requested not in _VARLEN_ALIBI_KEYS:
        supported = ", ".join(sorted(_VARLEN_ALIBI_KEYS))
        raise NotImplementedError(
            "varlen ALiBi slopes are implemented only for "
            f"{supported}; got {requested}"
        )


def _validate_varlen_window_size(window_size: object) -> tuple[int, int]:
    """Return a runtime-safe FlashAttention window pair for varlen dispatch."""
    if not isinstance(window_size, (tuple, list)):
        raise TypeError(
            "window_size must be a tuple or list containing exactly two "
            "Python integers"
        )
    if len(window_size) != 2:
        raise ValueError("window_size must contain exactly two Python integers")
    left, right = window_size
    if type(left) is not int or type(right) is not int:
        raise TypeError("window_size must contain exactly two Python integers")
    if left < -1 or right < -1:
        raise ValueError("window_size bounds must be -1 or non-negative")
    if left > _INT32_MAX or right > _INT32_MAX:
        raise ValueError(
            f"window_size bounds must not exceed signed int32 max {_INT32_MAX}"
        )
    return left, right


def _validate_varlen_self_attention_offsets(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    """Require query and key metadata to describe the same ragged sequences."""
    if q.shape[0] != k.shape[0]:
        raise NotImplementedError(
            "varlen symmetric-window attention requires identical "
            "cu_seqlens_q and cu_seqlens_k for self-attention"
        )

    # The common self-attention and QKV-packed forms pass one tensor twice. In
    # that case equality is structural and no host synchronization is needed,
    # so dynamic offsets remain CUDA-graph capturable. Accept equal independent
    # metadata too, after one deliberate comparison outside graph capture.
    if cu_seqlens_q.data_ptr() == cu_seqlens_k.data_ptr():
        return
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "varlen symmetric-window attention requires shared query/key "
                "offset storage during CUDA graph capture"
            )
        offsets_q = tuple(cu_seqlens_q.tolist())
        offsets_k = tuple(cu_seqlens_k.tolist())
    if offsets_q != offsets_k:
        raise NotImplementedError(
            "varlen symmetric-window attention requires identical "
            "cu_seqlens_q and cu_seqlens_k for self-attention"
        )


def _supports_varlen_diagnostic_return(spec: AttnShape) -> bool:
    """Whether ``spec`` is the one LSE-capable core varlen profile."""
    return (
        f"varlen_{spec.key}" == _VARLEN_DIAGNOSTIC_KEY
        and has_varlen_kernel(spec)
    )


def _has_full_varlen_token_totals(
    q: torch.Tensor,
    k: torch.Tensor,
    spec: AttnShape,
) -> bool:
    """Whether valid packed inputs contain the full declared dense batch.

    Varlen callers are responsible for valid cumulative offsets and lengths no
    larger than the declared maxima. Under that contract, these full totals
    imply that every sequence has its canonical maximum length. Keeping this
    decision shape-only also makes the SDPA autograd path CUDA-graph capturable.
    """
    return (
        q.shape[0] == spec.batch * spec.seqlen_q
        and k.shape[0] == spec.batch * spec.seqlen_k
    )


def _ragged_varlen_attention_lengths(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    spec: AttnShape,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate and return the bounded ragged query and key lengths.

    Reading cumulative offsets on the host is intentionally confined to this
    training-only fallback. Inference retains the generated packed kernel, and
    the full-length training path remains shape-only and graph-capturable.
    """
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "ragged varlen backward is not supported during CUDA graph "
                "capture"
            )
        offsets_q = tuple(cu_seqlens_q.tolist())
        offsets_k = tuple(cu_seqlens_k.tolist())

    if (
        offsets_q[0] != 0
        or offsets_q[-1] != q.shape[0]
        or offsets_k[0] != 0
        or offsets_k[-1] != k.shape[0]
    ):
        raise ValueError(
            "cu_seqlens_q/cu_seqlens_k must start at zero and end at the "
            "corresponding packed token totals"
        )

    lengths_q = tuple(
        stop - start for start, stop in zip(offsets_q, offsets_q[1:])
    )
    lengths_k = tuple(
        stop - start for start, stop in zip(offsets_k, offsets_k[1:])
    )
    if any(length < 0 for length in (*lengths_q, *lengths_k)):
        raise ValueError(
            "cu_seqlens_q/cu_seqlens_k must be monotonically nondecreasing"
        )
    if not any(lengths_q):
        raise NotImplementedError(
            "ragged varlen backward requires at least one nonempty query "
            "sequence"
        )
    if any(length == 0 for length in lengths_k) and offsets_q != offsets_k:
        raise NotImplementedError(
            "ragged varlen backward with empty key sequence slots is "
            "implemented only for causal self-attention with identical "
            "cu_seqlens_q and cu_seqlens_k"
        )
    if any(length > spec.seqlen_q for length in lengths_q) or any(
        length > spec.seqlen_k for length in lengths_k
    ):
        raise ValueError(
            "cu_seqlens_q/cu_seqlens_k contain a sequence longer than the "
            "corresponding declared maximum "
            f"({spec.seqlen_q}, {spec.seqlen_k})"
        )
    return lengths_q, lengths_k


def _ragged_varlen_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths_q: tuple[int, ...],
    lengths_k: tuple[int, ...],
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run each validated nonempty ragged sequence through SDPA."""
    q_sequences = torch.split(q, lengths_q)
    k_sequences = torch.split(k, lengths_k)
    v_sequences = torch.split(v, lengths_k)
    outputs: list[torch.Tensor] = []
    for q_sequence, k_sequence, v_sequence, seqlen_q, seqlen_k in zip(
        q_sequences, k_sequences, v_sequences, lengths_q, lengths_k
    ):
        # Empty query slots contain no packed output rows, so there is no SDPA
        # subproblem to launch. Their K/V slices remain unused and naturally
        # receive zero gradients. At least one nonempty query is guaranteed.
        if seqlen_q == 0:
            continue
        # Preserve the established self-attention call exactly. Unequal
        # sequences need their actual lengths in the SDPA adapter so causal
        # masking uses FlashAttention's bottom-right alignment.
        sequence_spec = spec
        if seqlen_q != seqlen_k:
            sequence_spec = AttnShape(
                batch=1,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                nheads_q=spec.nheads_q,
                nheads_kv=spec.nheads_kv,
                head_dim=spec.head_dim,
                dtype=spec.dtype,
                causal=spec.causal,
            )
        outputs.append(
            dense_attention_sdpa(
                q_sequence.unsqueeze(0),
                k_sequence.unsqueeze(0),
                v_sequence.unsqueeze(0),
                softmax_scale,
                sequence_spec,
            ).squeeze(0)
        )
    return torch.cat(outputs)


def _generic_dense_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
    alibi_slopes: torch.Tensor | None,
    *,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Adapt a validated dense batch to the generic packed Triton runtime."""
    total_q, total_k = _validate_generic_dense_layout(spec)

    # The dense inputs were validated as contiguous, so these views retain the
    # packed runtime's [total, heads, head_dim] layout without copying data.
    packed_q = q.view(total_q, spec.nheads_q, spec.head_dim)
    packed_k = k.view(total_k, spec.nheads_kv, spec.head_dim)
    packed_v = v.view(total_k, spec.nheads_kv, spec.head_dim)
    request_ids = torch.arange(
        spec.batch + 1, device=q.device, dtype=torch.int32
    )
    cu_seqlens_q = request_ids * spec.seqlen_q
    cu_seqlens_k = (
        cu_seqlens_q
        if spec.seqlen_q == spec.seqlen_k
        else request_ids * spec.seqlen_k
    )

    # Keep the Triton dependency lazy for callers that only inspect the
    # specialization manifest.
    from ._paged_attention import packed_attention

    packed_out = packed_attention(
        packed_q,
        packed_k,
        packed_v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        window_size=(-1, -1),
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=False,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_out, torch.Tensor):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention unexpectedly returned softmax LSE")
    return packed_out.view(
        spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim
    )


def _validate_dense_kvcache_index_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
    spec: AttnShape,
) -> None:
    """Validate metadata shared by dense-cache end and left-pad tensors."""
    expected_shape = (spec.batch,)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got "
            f"{tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if not tensor.is_cuda:
        raise ValueError(
            f"{name} must be a CUDA tensor, got device {tensor.device}"
        )
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on the same CUDA device as q, k_cache, and "
            "v_cache"
        )
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use torch.strided layout")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_dense_kvcache_tensor_span(
    cache_seqlens: torch.Tensor,
    cache_leftpad: torch.Tensor | None,
    *,
    device: torch.device,
    spec: AttnShape,
) -> None:
    """Validate the one supported device-resident dense-cache span."""
    _validate_dense_kvcache_index_tensor(
        cache_seqlens,
        "cache_seqlens",
        device=device,
        spec=spec,
    )
    if cache_leftpad is not None:
        _validate_dense_kvcache_index_tensor(
            cache_leftpad,
            "cache_leftpad",
            device=device,
            spec=spec,
        )

    # Recoverable bounds validation requires one deliberate device-to-host
    # synchronization. Reject capture before reading either tensor so graph
    # construction ends cleanly instead of failing inside an implicit sync.
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "tensor-valued cache_seqlens are not supported during CUDA "
                "graph capture"
            )
        if cache_leftpad is None:
            length = int(cache_seqlens.detach().item())
        else:
            leftpad, length = (
                int(value)
                for value in torch.cat(
                    (cache_leftpad.detach(), cache_seqlens.detach())
                ).tolist()
            )

    if cache_leftpad is not None:
        if not 0 <= leftpad < length <= spec.seqlen_k:
            raise ValueError(
                "cache_leftpad and cache_seqlens values must satisfy "
                f"0 <= cache_leftpad < cache_seqlens <= {spec.seqlen_k}, "
                f"got cache_leftpad={leftpad} and cache_seqlens={length}"
            )
        return

    if length < 1 or length > spec.seqlen_k:
        raise ValueError(
            "cache_seqlens values must be in the inclusive range "
            f"[1, {spec.seqlen_k}], got {length}"
        )


def _tensor_length_dense_kvcache_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_leftpad: torch.Tensor | None,
    softmax_scale: float,
    spec: AttnShape,
    *,
    return_softmax_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Read a device-selected cache span with one generic attention launch."""
    # Batch one makes the dense cache row identical to packed token storage.
    # Express its selected span as cumulative offsets so the generic packed
    # runtime can stay in one attention kernel without invoking torch.einsum
    # (and its per-stream cuBLAS workspace).
    from ._paged_attention import packed_attention

    packed_q = q.view(spec.batch, spec.nheads_q, spec.head_dim)
    packed_k = k_cache.view(spec.seqlen_k, spec.nheads_kv, spec.head_dim)
    packed_v = v_cache.view(spec.seqlen_k, spec.nheads_kv, spec.head_dim)
    cu_seqlens_q = torch.arange(
        spec.batch + 1, device=q.device, dtype=torch.int32
    )
    cu_seqlens_k = torch.cat(
        (
            cache_seqlens.new_zeros(1)
            if cache_leftpad is None
            else cache_leftpad,
            cache_seqlens,
        )
    )
    result = packed_attention(
        packed_q,
        packed_k,
        packed_v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=return_softmax_lse,
        shift_fa2_lse=False,
        fa_version=2,
    )

    output_shape = (
        spec.batch,
        spec.seqlen_q,
        spec.nheads_q,
        spec.head_dim,
    )
    if not return_softmax_lse:
        if not isinstance(result, torch.Tensor):  # pragma: no cover - contract guard
            raise RuntimeError(
                "tensor-length KV-cache attention unexpectedly returned LSE"
            )
        return result.reshape(output_shape)

    if not isinstance(result, tuple):  # pragma: no cover - contract guard
        raise RuntimeError(
            "tensor-length KV-cache attention did not return requested LSE"
        )
    output, packed_lse = result
    softmax_lse = packed_lse.transpose(0, 1).contiguous().unsqueeze(-1)
    return output.reshape(output_shape), softmax_lse


def _generic_dense_diagnostic_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
    *,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FA2-compatible dense diagnostics from the packed runtime."""
    total_q, total_k = _validate_generic_dense_layout(spec)

    # The packed runtime stores LSE as [heads, total_q]. Dense FA2 callers
    # receive the same values rearranged as [batch, heads, seqlen_q].
    packed_q = q.view(total_q, spec.nheads_q, spec.head_dim)
    packed_k = k.view(total_k, spec.nheads_kv, spec.head_dim)
    packed_v = v.view(total_k, spec.nheads_kv, spec.head_dim)
    request_ids = torch.arange(
        spec.batch + 1, device=q.device, dtype=torch.int32
    )
    cu_seqlens_q = request_ids * spec.seqlen_q
    cu_seqlens_k = (
        cu_seqlens_q
        if spec.seqlen_q == spec.seqlen_k
        else request_ids * spec.seqlen_k
    )

    from ._paged_attention import packed_attention

    packed_result = packed_attention(
        packed_q,
        packed_k,
        packed_v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        window_size=(-1, -1),
        softcap=softcap,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=True,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_result, tuple):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention did not return softmax LSE")
    packed_out, packed_lse = packed_result
    out = packed_out.view(
        spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim
    )
    softmax_lse = (
        packed_lse.view(spec.nheads_q, spec.batch, spec.seqlen_q)
        .permute(1, 0, 2)
        .contiguous()
    )
    return out, softmax_lse, q.new_empty((0,))


def _generic_varlen_alibi_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
    alibi_slopes: torch.Tensor,
) -> torch.Tensor:
    """Run validated packed ALiBi inputs through the generic Triton runtime."""
    # Keep the Triton dependency lazy for callers that only inspect the
    # specialization manifest.
    from ._paged_attention import packed_attention

    packed_out = packed_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=False,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_out, torch.Tensor):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention unexpectedly returned softmax LSE")
    return packed_out


def _generic_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run the exact validated Llama-3 varlen inference profile generically."""
    # This intentionally is not a general missing-specialization fallback.
    # Both cumulative-length tensors remain device-resident, preserving
    # graph-capturable ragged self- and cross-attention lengths.
    from ._paged_attention import packed_attention

    packed_out = packed_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=False,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_out, torch.Tensor):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention unexpectedly returned softmax LSE")
    return packed_out


def _generic_paged_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run the validated page-256 decode profile through generic Triton."""
    # Page 16 has checked-in generated specializations. This helper is kept
    # deliberately narrow so other page sizes and paged profiles still fail in
    # the core varlen validator before reaching the generic vLLM runtime.
    from ._paged_attention import paged_attention

    packed_out = paged_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=spec.causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=False,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_out, torch.Tensor):  # pragma: no cover - contract guard
        raise RuntimeError("generic paged attention unexpectedly returned softmax LSE")
    return packed_out


def _generic_varlen_symmetric_window_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
    window_size: tuple[int, int],
) -> torch.Tensor:
    """Run a validated noncausal symmetric window through generic Triton."""
    from ._paged_attention import packed_attention

    packed_out = packed_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=window_size,
        softcap=0.0,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=False,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_out, torch.Tensor):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention unexpectedly returned softmax LSE")
    return packed_out


def _generic_varlen_softcap_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> torch.Tensor:
    """Run the validated varlen softcap profile through generic Triton."""
    # Keep the Triton dependency lazy for callers that only inspect the
    # specialization manifest.
    from ._paged_attention import packed_attention

    packed_out = packed_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=_VARLEN_SOFTCAP,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=False,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_out, torch.Tensor):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention unexpectedly returned softmax LSE")
    return packed_out


def _generic_varlen_diagnostic_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    spec: AttnShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FA2-compatible diagnostics from the generic packed runtime."""
    # The generated specialization remains the default fast path, but it does
    # not materialize LSE. The generic packed kernel stores LSE directly in
    # FlashAttention's [heads, total_q] layout without unpacking ragged rows.
    from ._paged_attention import packed_attention

    packed_result = packed_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=spec.seqlen_q,
        max_seqlen_k=spec.seqlen_k,
        dynamic_max_seqlen_q=None,
        dynamic_max_seqlen_k=None,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        q_v=None,
        cp_world_size=1,
        cp_rank=0,
        cp_tot_seqused_k=None,
        out=None,
        return_softmax_lse=True,
        shift_fa2_lse=False,
        fa_version=2,
    )
    if not isinstance(packed_result, tuple):  # pragma: no cover - contract guard
        raise RuntimeError("generic packed attention did not return softmax LSE")
    out, softmax_lse = packed_result
    return out, softmax_lse, q.new_empty((0,))


def is_shape_supported(
    shape: ShapeLike, dtype: torch.dtype = torch.bfloat16, causal: bool = False
) -> bool:
    """True when an accelerated kernel for this exact shape is checked in.

    This intentionally reports specialization availability, not whether the
    generic dense forward fallback can execute the shape.
    """
    return has_kernel(normalize_shape(shape, dtype, causal))


def is_varlen_shape_supported(
    shape: ShapeLike, dtype: torch.dtype = torch.bfloat16, causal: bool = False
) -> bool:
    """True when a packed-sequence kernel for this maximum shape is checked in."""
    return has_varlen_kernel(normalize_shape(shape, dtype, causal))


def is_paged_shape_supported(
    shape: ShapeLike,
    dtype: torch.dtype = torch.bfloat16,
    causal: bool = False,
    page_size: int = 16,
) -> bool:
    """True when vLLM paged attention has this maximum shape and page size."""
    return has_paged_kernel(normalize_shape(shape, dtype, causal), page_size)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``flash_attn.flash_attn_func``.

    Args:
        q: ``[batch, seqlen_q, nheads_q, head_dim]``, contiguous, fp16 or bf16.
        k: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        v: ``[batch, seqlen_k, nheads_kv, head_dim]``.
        dropout_p: values in ``(0, 1)`` are supported only for the shipped
            noncausal bf16 ``(8, 512, 512, 16, 16, 64)`` encoder-training and
            ``(16, 512, 512, 12, 12, 64)`` BERT-base profiles, plus causal bf16
            ``(2, 1024, 1024, 32, 32, 64)`` attention and asymmetric causal
            bf16 ``(1, 64, 320, 8, 2, 128)`` GQA.
        softmax_scale: defaults to ``1 / sqrt(head_dim)``.
        causal: bottom-right causal masking, including unequal sequence lengths.
        softcap: exactly ``50.0`` is supported for forward-only causal bf16
            ``(1, 4096, 4096, 16, 8, 256)`` Gemma-2 attention. Zero retains
            ordinary dispatch; every other positive value remains unsupported.
        alibi_slopes: fp32 CUDA tensor shaped ``[nheads_q]`` or
            ``[batch, nheads_q]``. ALiBi calls are currently forward-only.
        return_attn_probs: for the shipped bf16 BERT-base encoder and causal
            GPT-2 profile, three Llama GQA decode profiles, and the
            ``softcap=50.0`` Gemma-2 profile, return FlashAttention's diagnostic
            tuple. This is available only when no backward is needed and all
            options other than ``softmax_scale`` (plus the documented
            causal/softcap settings) retain their defaults.
        shape: required. Either an :class:`AttnShape`, a 4-tuple
            ``(batch, seqlen, nheads, head_dim)``, or a 6-tuple
            ``(batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim)``.

    Returns:
        ``[batch, seqlen_q, nheads_q, head_dim]``. With the supported
        ``return_attn_probs=True`` subset, returns ``(out, softmax_lse,
        S_dmask)``. The LSE is fp32 with shape
        ``[batch, nheads_q, seqlen_q]`` and ``S_dmask`` is an empty
        input-dtype tensor, matching FlashAttention when dropout is zero.

    Unregistered fp16/bf16 shapes with ``head_dim <= 256``, all ALiBi calls, and
    the supported Gemma-2 softcap call use a generic packed Triton forward
    kernel. The exact default-option,
    default-scale, no-backward noncausal bf16
    ``(2, 1024, 1024, 16, 16, 256)`` call uses direct PyTorch Flash SDPA on
    SM90 when eligible. The corresponding causal bf16
    ``(1, 4096, 4096, 32, 8, 128)``, ``(1, 8192, 8192, 28, 4, 128)``,
    ``(2, 8192, 8192, 16, 16, 128)``, ``(4, 2048, 2048, 28, 4, 128)``, and
    ``(4, 4096, 4096, 32, 32, 128)`` calls use direct cuDNN SDPA. All six
    paths fall back to their generated kernels when ineligible. Grad-enabled
    calls without ALiBi or a generated backward, plus supported positive-dropout
    calls, use PyTorch SDPA autograd.
    Deterministic zero-dropout training on the exact BERT-base and causal GPT-2
    profiles uses PyTorch's direct math SDPA operator for repeatable output and
    gradients.
    Zero-dropout training on the shipped encoder profile retains generated
    forward values. Its omitted/default-scale SM90 path uses raw BSHD PyTorch
    Flash gradients, falling back to the generated backward when Flash is
    unavailable. Explicit-scale, non-SM90, and deterministic calls retain the
    generated backward.
    :func:`is_shape_supported` remains ``False`` for unregistered calls because
    it reports checked-in acceleration only.
    """
    dropout = _reject_unsupported(
        dropout_p,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
        allow_dropout=True,
        allow_alibi=True,
        allow_return_attn_probs=True,
        allow_softcap_return_attn_probs=True,
        allowed_softcap=_GEMMA2_SOFTCAP,
    )
    spec = normalize_shape(shape, q.dtype, causal)
    has_softcap = softcap != 0.0
    if has_softcap and spec.key != _GEMMA2_SOFTCAP_KEY:
        raise NotImplementedError(
            "softcap=50.0 is implemented only for the no-backward bf16 causal "
            "Gemma-2 profile (1, 4096, 4096, 16, 8, 256); "
            f"got {spec.describe()}"
        )
    if dropout != 0.0:
        if spec.key not in _DROPOUT_SDPA_KEYS:
            raise NotImplementedError(
                "dropout is implemented only for the shipped dense profiles "
                f"{', '.join(sorted(_DROPOUT_SDPA_KEYS))}; got {spec.key}"
            )
        if deterministic:
            raise NotImplementedError(
                "deterministic=True is not supported with dropout"
            )
    check_tensors(q, k, v, spec)
    if alibi_slopes is not None:
        _validate_alibi_slopes(alibi_slopes, q, spec)
    default_softmax_scale = softmax_scale is None
    if default_softmax_scale:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    scale = float(softmax_scale)
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v)
    )
    if has_softcap:
        if needs_backward:
            raise NotImplementedError(
                "softcap backward is not implemented; the Gemma-2 softcap "
                "profile is forward-only"
            )
        if return_attn_probs:
            if deterministic:
                raise NotImplementedError(
                    "return_attn_probs=True requires deterministic=False"
                )
            return _generic_dense_diagnostic_forward(
                q,
                k,
                v,
                scale,
                spec,
                softcap=_GEMMA2_SOFTCAP,
            )
        return _generic_dense_forward(
            q,
            k,
            v,
            scale,
            spec,
            None,
            softcap=_GEMMA2_SOFTCAP,
        )
    if return_attn_probs:
        if needs_backward:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented for grad-enabled "
                "calls"
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented with ALiBi slopes"
            )
        if deterministic:
            raise NotImplementedError(
                "return_attn_probs=True requires deterministic=False"
            )
        if not _supports_diagnostic_return(spec):
            raise NotImplementedError(
                "return_attn_probs=True is implemented only for the shipped "
                f"{_BERT_DIAGNOSTIC_KEY} BERT-base encoder, the shipped "
                f"{_CAUSAL_DROPOUT_KEY} causal GPT-2 profile, and the three "
                "shipped batch-1, single-token bf16 Llama GQA decode profiles"
            )
        if spec.key in _GENERIC_DENSE_DIAGNOSTIC_KEYS:
            return _generic_dense_diagnostic_forward(q, k, v, scale, spec)
        out, softmax_lse = lookup(spec)(
            q, k, v, scale, return_softmax_lse=True
        )
        return out, softmax_lse, q.new_empty((0,))
    if alibi_slopes is not None and torch.is_grad_enabled() and (
        needs_backward or alibi_slopes.requires_grad
    ):
        raise NotImplementedError(
            "ALiBi backward is not implemented; ALiBi calls are forward-only"
        )
    if dropout != 0.0:
        return dense_attention_sdpa(q, k, v, scale, spec, dropout)
    if needs_backward:
        if has_backward(spec):
            if (
                default_softmax_scale
                and not deterministic
                and spec.key == _ENCODER_TRAINING_KEY
                and _is_sm90(q.device)
            ):
                return attention_with_flash_gradients(q, k, v, scale, spec)
            return attention_autograd(q, k, v, scale, spec)
        if deterministic:
            if spec.key in _DENSE_DETERMINISTIC_BACKWARD_KEYS:
                return dense_attention_math_sdpa(q, k, v, scale, spec)
            raise NotImplementedError(
                "deterministic=True is not supported by the PyTorch SDPA "
                "autograd fallback"
            )
        # Preserve the generic dense forward's documented shape envelope for
        # unregistered calls even though SDPA itself may accept more shapes.
        if not has_kernel(spec):
            _validate_generic_dense_layout(spec)
        return dense_attention_sdpa(q, k, v, scale, spec)
    if (
        default_softmax_scale
        and not deterministic
        and alibi_slopes is None
        and spec.key == _FLASH_SDPA_FAST_PATH_KEY
        and _is_sm90(q.device)
    ):
        flash_out = dense_attention_flash_default_scale(q, k, v)
        if flash_out is not None:
            return flash_out
    if (
        default_softmax_scale
        and not deterministic
        and alibi_slopes is None
        and spec.key in _CUDNN_SDPA_FAST_PATH_KEYS
        and _is_sm90(q.device)
    ):
        cudnn_out = dense_attention_cudnn_default_scale(q, k, v)
        if cudnn_out is not None:
            return cudnn_out
    if alibi_slopes is None and has_kernel(spec):
        return lookup(spec)(q, k, v, scale)
    return _generic_dense_forward(q, k, v, scale, spec, alibi_slopes)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``flash_attn.flash_attn_varlen_func``.

    ``q`` is ``[total_q, nheads_q, head_dim]``. Without ``block_table``,
    ``k``/``v`` are ``[total_k, nheads_kv, head_dim]``. With ``block_table``,
    the checked-in bf16 page-size-16 profiles are chunked prefill
    ``(2, 200, 320, 8, 2, 128)`` with ``causal=True`` and decode
    ``(4, 1, 1024, 8, 2, 128)`` with either causal flag. They accept caches
    shaped ``[num_blocks, 16, 2, 128]``. The decode profile additionally
    accepts page-size-256 caches through the existing generic paged runtime;
    page-size-16 calls retain generated dispatch. Both paths derive each
    request's used cache length from adjacent ``cu_seqlens_k`` offsets without
    copying them to the host. Page-size-256 decode supports only forward calls
    with the default options, apart from either causal flag and a default or
    custom ``softmax_scale``.
    The int32 CUDA cumulative-length tensors contain ``batch + 1`` offsets.
    ``shape`` uses the same forms as :func:`flash_attn_func`, but its sequence
    dimensions are the maximum query and key lengths rather than dense tensor
    dimensions.

    The packed token totals and individual sequence lengths may change between
    calls to the same specialization.  The batch size, maxima, head geometry,
    dtype, and causal mode must continue to match ``shape``. Forward-only
    ALiBi is supported for both causal modes of the
    ``(8, 512, 512, 16, 16, 64)`` bf16 profile, with fp32 slopes shaped
    ``[16]`` or ``[8, 16]``. The causal Llama-3 profile below accepts fp32
    slopes shaped ``[32]`` or ``[4, 32]`` under the same forward-only contract.

    Causal bf16 Llama-3 GQA attention with maximum shape
    ``(4, 256, 256, 32, 8, 128)`` is also supported when no backward or
    incompatible optional feature is requested. Its device-resident query and
    key offsets may independently describe arbitrary ragged lengths.
    Shared-offset self-attention continues to support a mix of empty and
    nonempty slots. This unregistered profile uses the generic packed Triton
    runtime and accepts the default or a custom ``softmax_scale``, with or
    without the ALiBi slopes described above. Registered profiles retain
    generated dispatch.

    The noncausal version also supports local self-attention with
    ``window_size=(radius, radius)`` for a finite non-negative ``radius``.
    Query and key cumulative offsets must be identical. Forward-only calls use
    the generic packed Triton runtime. Zero-dropout backward is additionally
    supported when all eight sequences have length 512, by reshaping them to a
    dense windowed PyTorch SDPA call. Both paths accept the default or a custom
    ``softmax_scale``. The global ``(-1, -1)`` default retains its existing
    dispatch.

    The causal version of that profile accepts exactly ``softcap=50.0`` for
    calls that do not need backward, with either the default or a custom
    ``softmax_scale``. Softcapped calls use the generic packed Triton runtime;
    ``softcap=0`` retains the generated specialization. Other caps and
    profiles, paged caches, gradients, dropout, ALiBi, local windows, and
    diagnostic returns remain unsupported with softcap.

    The causal version of that profile also supports
    ``return_attn_probs=True`` when no backward is needed, ``causal=True``, and
    all options other than ``softmax_scale`` retain their defaults. It returns
    ``(out, softmax_lse, S_dmask)`` with fp32 LSE shaped ``[nheads_q, total_q]``
    and an empty bf16 ``S_dmask``, matching FlashAttention 2 when dropout is
    zero.

    Both causal modes support zero-dropout backward when all eight query and
    key sequences have length 512. The causal profile additionally supports
    independent query and key cumulative offsets describing eight nonempty key
    sequences and a mix of empty and nonempty query sequences, all of length
    at most 512. Self-attention with identical query/key offsets continues to
    accept mixed empty and nonempty slots. Full-length inputs are reshaped to
    one dense BSHD call; ragged inputs use one bounded PyTorch SDPA call per
    nonempty query sequence. All-empty query batches, empty key slots in
    cross-attention, ragged noncausal (including windowed calls), graph-captured
    ragged, paged, ALiBi, and diagnostic varlen backward remain unsupported.
    Deterministic zero-dropout backward is supported only for the full-length
    causal profile, using direct math SDPA; the noncausal and all ragged forms
    still reject it. Calls that do not need backward retain the generated packed
    kernel, including ragged and full-length calls with the global window.
    """
    window = _validate_varlen_window_size(window_size)
    # Non-default varlen windows have their narrow support checks below. Keep
    # the shared validator responsible for all other feature flags without
    # widening dense or KV-cache window support.
    _reject_unsupported(
        dropout_p,
        (-1, -1),
        softcap,
        alibi_slopes,
        return_attn_probs,
        allow_alibi=block_table is None,
        allow_return_attn_probs=True,
        allowed_softcap=_VARLEN_SOFTCAP,
    )
    if type(max_seqlen_q) is not int or type(max_seqlen_k) is not int:
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")

    spec = normalize_shape(shape, q.dtype, causal)
    has_softcap = softcap != 0.0
    if has_softcap and block_table is not None:
        raise NotImplementedError(
            "varlen softcap is not implemented with paged block_table caches"
        )
    if has_softcap and f"varlen_{spec.key}" != _VARLEN_SOFTCAP_KEY:
        raise NotImplementedError(
            "softcap=50.0 is implemented only for the no-backward causal "
            "varlen bf16 profile (8, 512, 512, 16, 16, 64); "
            f"got {spec.describe()}"
        )
    if max_seqlen_q != spec.seqlen_q or max_seqlen_k != spec.seqlen_k:
        raise ValueError(
            "max_seqlen_q/max_seqlen_k must match the maximum sequence lengths "
            f"declared by shape ({spec.seqlen_q}, {spec.seqlen_k}); got "
            f"({max_seqlen_q}, {max_seqlen_k})"
        )

    is_llama3_varlen_inference = (
        f"varlen_{spec.key}" == _LLAMA3_VARLEN_INFERENCE_KEY
    )
    if is_llama3_varlen_inference and deterministic:
        raise NotImplementedError(
            "deterministic=True is not implemented for the Llama-3 varlen "
            "inference profile"
        )

    has_symmetric_window = window != (-1, -1)
    if has_symmetric_window:
        if window[0] < 0 or window[0] != window[1]:
            raise NotImplementedError(
                "varlen sliding-window attention is implemented only for "
                "finite window_size=(radius, radius) with radius >= 0"
            )
        requested = f"varlen_{spec.key}"
        if requested != _VARLEN_SYMMETRIC_WINDOW_KEY:
            raise NotImplementedError(
                "varlen sliding-window attention with symmetric bounds is "
                "implemented only for "
                f"{_VARLEN_SYMMETRIC_WINDOW_KEY}; got {requested}"
            )
        if block_table is not None:
            raise NotImplementedError(
                "varlen sliding-window attention is not implemented with "
                "block_table"
            )
        if softcap != 0.0:
            raise NotImplementedError(
                "varlen sliding-window attention is not implemented with "
                "softcap"
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "varlen sliding-window attention is not implemented with "
                "ALiBi slopes, including symmetric windows"
            )
        if return_attn_probs:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented with varlen "
                "symmetric-window attention"
            )
        if deterministic:
            raise NotImplementedError(
                "deterministic=True is not supported with varlen "
                "symmetric-window attention"
            )

    if return_attn_probs:
        if block_table is not None:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented with block_table"
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented with ALiBi slopes"
            )
        if deterministic:
            raise NotImplementedError(
                "return_attn_probs=True requires deterministic=False"
            )
        if not _supports_varlen_diagnostic_return(spec):
            raise NotImplementedError(
                "return_attn_probs=True is implemented only for "
                f"{_VARLEN_DIAGNOSTIC_KEY}"
            )

    if alibi_slopes is not None:
        _check_varlen_alibi_spec(spec)

    if block_table is not None:
        page_size = check_paged_varlen_tensors(
            q, k, v, cu_seqlens_q, cu_seqlens_k, block_table, spec
        )
        _check_core_paged_varlen_spec(spec, page_size)
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q, k, v)
        ):
            raise NotImplementedError(
                "flash_attn_varlen_func with block_table is forward-only; "
                "no paged-cache backward kernel is checked in"
            )
        if page_size == 256 and deterministic:
            raise NotImplementedError(
                "deterministic=True is not implemented for core page-size-256 "
                "paged varlen decode"
            )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(spec.head_dim)
        # Keep sequence metadata on-device. This allocates the four per-request
        # used lengths directly on CUDA and remains safe for graph capture.
        seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if page_size != _CORE_PAGED_GENERATED_PAGE_SIZE:
            return _generic_paged_varlen_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                seqused_k,
                block_table,
                float(softmax_scale),
                spec,
            )
        # For the decode profile, a bottom-right-aligned single-token query has
        # equivalent causal and non-causal results. The registry handles that
        # equivalence; chunked prefill still requires its generated causal mode.
        kernel = lookup_paged(spec, page_size)
        return kernel(
            q,
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            float(softmax_scale),
            causal,
        )

    check_varlen_tensors(q, k, v, cu_seqlens_q, cu_seqlens_k, spec)
    if alibi_slopes is not None:
        _validate_alibi_slopes(alibi_slopes, q, spec)
    grad_tensors = (
        (q, k, v, alibi_slopes)
        if alibi_slopes is not None
        else (q, k, v)
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in grad_tensors
    )
    if has_symmetric_window:
        _validate_varlen_self_attention_offsets(
            q, k, cu_seqlens_q, cu_seqlens_k
        )
    if needs_backward:
        if has_softcap:
            raise NotImplementedError(
                "softcap backward is not implemented; the causal varlen "
                "softcap profile is forward-only"
            )
        if return_attn_probs:
            raise NotImplementedError(
                "return_attn_probs=True is not implemented for grad-enabled "
                "calls"
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "ALiBi backward is not implemented; varlen ALiBi calls are "
                "forward-only"
            )
        if f"varlen_{spec.key}" not in _VARLEN_SDPA_BACKWARD_KEYS:
            supported = ", ".join(sorted(_VARLEN_SDPA_BACKWARD_KEYS))
            raise NotImplementedError(
                "varlen backward is implemented only for "
                f"{supported} with eight full-length "
                "sequences"
            )
        full_length = _has_full_varlen_token_totals(q, k, spec)
        if deterministic and (
            f"varlen_{spec.key}" != _VARLEN_DETERMINISTIC_BACKWARD_KEY
            or not full_length
        ):
            raise NotImplementedError(
                "deterministic=True is implemented only for full-length "
                f"{_VARLEN_DETERMINISTIC_BACKWARD_KEY} zero-dropout training"
            )
        if has_symmetric_window and not full_length:
            raise NotImplementedError(
                "varlen sliding-window backward is implemented only when all "
                "eight self-attention sequences have length 512; ragged "
                "windowed calls remain forward-only"
            )
        if (
            not full_length
            and f"varlen_{spec.key}" != _RAGGED_VARLEN_SDPA_BACKWARD_KEY
        ):
            raise NotImplementedError(
                "ragged varlen backward is implemented only for the causal "
                "bf16 (8, 512, 512, 16, 16, 64) profile; "
                "ragged noncausal calls remain forward-only"
            )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(spec.head_dim)
        scale = float(softmax_scale)
        if not full_length:
            lengths_q, lengths_k = _ragged_varlen_attention_lengths(
                q, k, cu_seqlens_q, cu_seqlens_k, spec
            )
            return _ragged_varlen_attention_sdpa(
                q, k, v, lengths_q, lengths_k, scale, spec
            )
        dense_q = q.reshape(
            spec.batch, spec.seqlen_q, spec.nheads_q, spec.head_dim
        )
        dense_k = k.reshape(
            spec.batch, spec.seqlen_k, spec.nheads_kv, spec.head_dim
        )
        dense_v = v.reshape(
            spec.batch, spec.seqlen_k, spec.nheads_kv, spec.head_dim
        )
        if deterministic:
            dense_out = dense_attention_math_sdpa(
                dense_q, dense_k, dense_v, scale, spec
            )
        elif has_symmetric_window:
            dense_out = dense_attention_sdpa(
                dense_q,
                dense_k,
                dense_v,
                scale,
                spec,
                symmetric_window_radius=window[0],
            )
        else:
            # Preserve the existing global backward call and its dispatch
            # signature exactly; the optional radius belongs only to the new
            # finite-window bridge.
            dense_out = dense_attention_sdpa(
                dense_q, dense_k, dense_v, scale, spec
            )
        return dense_out.reshape(q.shape)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    scale = float(softmax_scale)
    if has_symmetric_window:
        return _generic_varlen_symmetric_window_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            spec,
            window,
        )
    if has_softcap:
        return _generic_varlen_softcap_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            spec,
        )
    if return_attn_probs:
        return _generic_varlen_diagnostic_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            spec,
        )
    if alibi_slopes is not None:
        return _generic_varlen_alibi_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            spec,
            alibi_slopes,
        )
    if is_llama3_varlen_inference:
        return _generic_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            scale,
            spec,
        )
    # Keep ordinary slope-free calls on the generated specialization.
    kernel = lookup_varlen(spec)
    return kernel(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        causal,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run varlen self-attention on ``[total, 3, nheads, head_dim]`` QKV."""
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError(
            "qkv must have shape [total, 3, nheads, head_dim], "
            f"got {tuple(qkv.shape)}"
        )
    q, k, v = (qkv[:, index].contiguous() for index in range(3))
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run varlen attention with ``[total_k, 2, nheads_kv, head_dim]`` KV."""
    if kv.ndim != 4 or kv.shape[1] != 2:
        raise ValueError(
            "kv must have shape [total_k, 2, nheads_kv, head_dim], "
            f"got {tuple(kv.shape)}"
        )
    k, v = (kv[:, index].contiguous() for index in range(2))
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )


def _paged_kvcache_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    append_kv: bool,
    cache_seqlens: int | torch.Tensor | None,
    block_table: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_softmax_lse: bool,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Adapt exact read-only dense-query paged profiles to core varlen."""
    if append_kv:
        raise NotImplementedError(
            "paged KV-cache updates are not implemented; paged caches are read-only"
        )
    spec = normalize_shape(shape, q.dtype, causal)
    for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if cache.ndim != 4:
            raise ValueError(
                f"{name} must be a paged KV cache with rank 4, got "
                f"shape {tuple(cache.shape)}"
            )
    if k_cache.shape[1] != v_cache.shape[1]:
        raise ValueError(
            "k_cache and v_cache must have the same page size, got "
            f"{k_cache.shape[1]} and {v_cache.shape[1]}"
        )
    page_size = k_cache.shape[1]
    _check_core_paged_kvcache_spec(spec, page_size)
    requested = (
        spec.batch,
        spec.seqlen_q,
        spec.seqlen_k,
        spec.nheads_q,
        spec.nheads_kv,
        spec.head_dim,
        spec.dtype,
    )
    has_softcap = softcap != 0.0
    if has_softcap and (
        softcap != _PAGE256_PAGED_KVCACHE_SOFTCAP
        or requested != _CORE_PAGED_KVCACHE_SHAPE
        or page_size != 256
    ):
        raise NotImplementedError(
            "softcap=50.0 for paged KV caches is implemented only for the "
            "read-only bf16 page-size-256 batch=4 seqlen_q=1 seqlen_k=1024 "
            "nheads=8 (GQA 8:2) head_dim=128 decode profile"
        )
    if return_softmax_lse and requested != _CORE_PAGED_KVCACHE_SHAPE:
        raise NotImplementedError(
            "return_softmax_lse=True for paged KV caches is implemented only "
            "for the bf16 page-size-16 or page-size-256 batch=4 seqlen_q=1 "
            "seqlen_k=1024 "
            "nheads=8 (GQA 8:2) head_dim=128 decode profile"
        )
    if alibi_slopes is not None:
        supports_alibi_page_size = (
            page_size == _CORE_PAGED_GENERATED_PAGE_SIZE
            or (requested == _CORE_PAGED_KVCACHE_SHAPE and page_size == 256)
        )
        if not supports_alibi_page_size:
            raise NotImplementedError(
                "paged KV-cache ALiBi is implemented only with page-size-16 "
                "caches or page-size-256 caches for the decode profile"
            )
        if requested not in _CORE_PAGED_KVCACHE_SHAPES:
            raise NotImplementedError(
                "paged KV-cache ALiBi is implemented only for the bf16 "
                "page-size-16 batch=2 seqlen_q=200 seqlen_k=320 "
                "chunked-prefill and page-size-16 or page-size-256 batch=4 "
                "seqlen_q=1 seqlen_k=1024 decode profiles with nheads=8 "
                "(GQA 8:2) head_dim=128"
            )
        supports_alibi_lse = (
            requested == _CORE_PAGED_KVCACHE_SHAPE and page_size == 256
        )
        if return_softmax_lse and not supports_alibi_lse:
            raise NotImplementedError(
                "return_softmax_lse=True with paged KV-cache ALiBi is "
                "implemented only for the read-only bf16 page-size-256 "
                "batch=4 seqlen_q=1 seqlen_k=1024 nheads=8 (GQA 8:2) "
                "head_dim=128 decode profile"
            )

    expected_q = (
        spec.batch,
        spec.seqlen_q,
        spec.nheads_q,
        spec.head_dim,
    )
    if tuple(q.shape) != expected_q:
        raise ValueError(
            f"q has shape {tuple(q.shape)} but the declared paged KV-cache "
            f"shape requires {expected_q}"
        )
    if not isinstance(cache_seqlens, torch.Tensor):
        raise NotImplementedError(
            "paged KV caches require cache_seqlens as a CUDA int32 tensor "
            "with shape [batch]"
        )
    if cache_seqlens.layout != torch.strided:
        raise ValueError("cache_seqlens must use torch.strided layout")
    if tuple(cache_seqlens.shape) != (spec.batch,):
        raise ValueError(
            f"cache_seqlens must have shape ({spec.batch},), got "
            f"{tuple(cache_seqlens.shape)}"
        )
    if cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must have dtype torch.int32")
    if not cache_seqlens.is_cuda:
        raise ValueError(
            "cache_seqlens must be a CUDA tensor, got device "
            f"{cache_seqlens.device}"
        )
    if cache_seqlens.device != q.device:
        raise ValueError(
            "cache_seqlens must be on the same CUDA device as q, k_cache, "
            "and v_cache"
        )
    if alibi_slopes is not None:
        _validate_alibi_slopes(alibi_slopes, q, spec)

    grad_tensors = (
        (q, k_cache, v_cache, alibi_slopes)
        if alibi_slopes is not None
        else (q, k_cache, v_cache)
    )
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in grad_tensors
    ):
        if alibi_slopes is not None:
            raise NotImplementedError(
                "ALiBi backward is not implemented; paged KV-cache ALiBi "
                "calls are forward-only"
            )
        raise NotImplementedError(
            "paged KV-cache calls do not support autograd; use inference mode "
            "or detach the inputs"
        )

    scale = (
        1.0 / math.sqrt(spec.head_dim)
        if softmax_scale is None
        else float(softmax_scale)
    )
    packed_q = q.flatten(0, 1)
    cu_seqlens_q = torch.arange(
        0,
        (spec.batch + 1) * spec.seqlen_q,
        spec.seqlen_q,
        device=q.device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.cat(
        (
            cache_seqlens.new_zeros(1),
            cache_seqlens.cumsum(dim=0, dtype=torch.int32),
        )
    )
    if (
        page_size != _CORE_PAGED_GENERATED_PAGE_SIZE
        or return_softmax_lse
        or alibi_slopes is not None
        or has_softcap
    ):
        # The generated paged specialization intentionally stays on the lean
        # page-16 slope-free output-only contract. Reuse the vLLM-compatible
        # single-launch runtime for page-256 decode and its softcap support,
        # ALiBi, or the diagnostic return it already computes.
        check_paged_varlen_tensors(
            packed_q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            spec,
        )
        from ._paged_attention import paged_attention

        # The generic kernel indexes per-request lengths as a flat array and
        # therefore requires unit-stride storage. Preserve the public adapter's
        # support for valid strided metadata by normalizing only this argument.
        seqused_k = cache_seqlens.contiguous()
        # FA2's causal KV-cache ALiBi kernel omits the row-constant
        # ``-slope * aligned_query_position`` from its accumulated logits and
        # reports the correspondingly shifted LSE. Its noncausal cache path
        # retains the mathematical LSE.
        shift_fa2_lse = (
            return_softmax_lse and alibi_slopes is not None and spec.causal
        )
        result = paged_attention(
            packed_q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q=spec.seqlen_q,
            max_seqlen_k=spec.seqlen_k,
            dynamic_max_seqlen_q=None,
            dynamic_max_seqlen_k=None,
            softmax_scale=scale,
            causal=spec.causal,
            window_size=(-1, -1),
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            s_aux=None,
            q_v=None,
            cp_world_size=1,
            cp_rank=0,
            cp_tot_seqused_k=None,
            out=None,
            return_softmax_lse=return_softmax_lse,
            shift_fa2_lse=shift_fa2_lse,
            fa_version=2,
        )
        if return_softmax_lse:
            if not isinstance(result, tuple):  # pragma: no cover - contract guard
                raise RuntimeError("generic paged attention did not return softmax LSE")
            packed_out, packed_lse = result
            softmax_lse = (
                packed_lse.reshape(spec.nheads_q, spec.batch, spec.seqlen_q)
                .permute(1, 0, 2)
                .contiguous()
            )
            return packed_out.reshape(expected_q), softmax_lse
        if not isinstance(result, torch.Tensor):  # pragma: no cover - contract guard
            raise RuntimeError(
                "generic paged attention unexpectedly returned softmax LSE"
            )
        return result.reshape(expected_q)

    packed_out = flash_attn_varlen_func(
        packed_q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        spec.seqlen_q,
        spec.seqlen_k,
        softmax_scale=scale,
        causal=causal,
        block_table=block_table,
        shape=spec,
    )
    return packed_out.reshape(expected_q)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: torch.Tensor | None = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Read from, or append a bounded update to, a FlashAttention-style KV cache.

    The general supported path is a dense, contiguous cache and exactly one
    query token. As with every entry point in this package, ``shape`` is
    required and describes ``q`` plus the cache:
    ``(batch, query_len, cache_len, nheads_q, nheads_kv, head_dim)``. One
    additional speculative-decoding slice accepts exactly two query tokens:
    causal bf16 ``(1, 2, 1024, 32, 8, 128)`` with a full dense cache. It uses
    the generic packed runtime and supports the default or a custom softmax
    scale. A paired two-token K/V update is accepted only at
    ``cache_seqlens=1022`` and fills the final two slots before attention.
    That update accepts full-head interleaved rotary tables and rotates the two
    query/key tokens at positions 1022 and 1023. This profile does not support
    LSE, partial read-only lengths, partial or non-interleaved rotary,
    remapping, autograd, or noncausal attention.

    A separate read-only speculative-decoding slice accepts exactly four query
    tokens: causal bf16 ``(1, 4, 1024, 32, 8, 128)`` with a full dense cache.
    It also uses the generic packed runtime and supports the default or a custom
    softmax scale. This profile does not support LSE, cache updates, partial or
    tensor-valued lengths, rotary, remapping, autograd, or noncausal attention.

    Two read-only paged profiles are also exposed with an int32 CUDA
    ``cache_seqlens`` tensor shaped ``[batch]`` and ``block_table``. Bf16
    ``(2, 200, 320, 8, 2, 128)`` accepts page-size-16 caches and supports only
    ``causal=True`` and uses bottom-right causal alignment for chunked prefill.
    Bf16 ``(4, 1, 1024, 8, 2, 128)`` accepts page-size-16 or page-size-256
    caches and supports both causal modes, which are equivalent for
    single-token bottom-right decode. Page-16 slope-free output-only calls
    route through :func:`flash_attn_varlen_func`; page-256 calls use the generic
    paged runtime and accept exactly ``softcap=50.0`` for forward-only calls.
    Both profiles support ragged logical caches. The page-16 profiles and the
    page-256 decode profile additionally accept forward-only fp32 ALiBi slopes
    shaped ``[8]`` or ``[batch, 8]`` through the generic paged runtime
    (``[2, 8]`` for chunked prefill and ``[4, 8]`` for decode).

    For dense caches, ``cache_seqlens`` may be omitted or supplied as a Python
    integer equal to the declared cache length for a read-only call. The causal
    and noncausal bf16 ``(1, 1, 4096, 32, 8, 128)`` profiles additionally accept
    forward-only fp32 ALiBi slopes shaped ``[32]`` or ``[1, 32]``. These calls
    use the generic packed runtime; omitting slopes retains the generated
    specialization. ALiBi cannot be combined with LSE, updates, partial lengths,
    remapping, left padding, rotary, windows, softcap, or autograd. The causal
    bf16 ``(1, 1, 16384, 32, 8, 128)`` profile additionally accepts a contiguous
    CUDA int32 ``cache_seqlens`` tensor shaped ``[1]`` selecting the end of a
    prefix. A matching ``cache_leftpad`` tensor selects the half-open cache span
    ``[cache_leftpad, cache_seqlens)`` when
    ``0 <= cache_leftpad < cache_seqlens <= 16384``. This tensor-span path
    synchronizes once for recoverable bounds validation and rejects CUDA graph
    capture and autograd. A full, read-only cache for that exact 16K profile may
    pass ``num_splits=16`` to match the generated kernel's fixed split count;
    all other explicit split selectors are unsupported. The explicit selector
    cannot be combined with tensor-valued cache spans, updates, rotary metadata,
    paged caches, or autograd. For the single-token dense paths, a paired ``k``
    and ``v`` update is supported when a Python integer ``cache_seqlens`` is
    exactly one less than the declared length; the update is copied into the
    final cache slot before attention runs. On this dense path,
    ``return_softmax_lse=True`` returns ``(out, softmax_lse)`` with LSE shape
    ``[batch, nheads_q, 1]`` and fp32 dtype, matching FlashAttention. The paged
    decode profile supports the same return for page sizes 16 and 256 through
    the generic single-launch paged runtime; page-16 slope-free output-only
    calls retain the generated specialization. The read-only page-256 decode
    path also supports that LSE return with ALiBi. Its causal result follows
    FlashAttention 2's position-shifted ALiBi LSE convention, while its
    noncausal result retains the mathematical LSE. Cache tensors created in
    inference mode must also be updated in inference mode, and an
    append requires disjoint query, K-cache, and V-cache memory. Tensor-valued
    lengths and left padding on updates and other dense profiles, scalar partial
    lengths, and multi-token updates outside the exact two-token profile fail
    explicitly. A paired ``rotary_cos``/``rotary_sin`` table may be supplied
    for the one-token final-slot append: the default interleaved layout may
    cover the full head or the first 64 dimensions of a 128-dimensional head,
    while the non-interleaved GPT-NeoX layout requires full-head rotation. Both
    layouts rotate ``q`` and the appended ``k`` at ``cache_seqlens``. The exact
    two-token append accepts only full-head interleaved rotary and rotates its
    tokens at consecutive positions. Read-only rotary calls and other paged
    profiles fail explicitly. Paged updates, rotary, and autograd are
    unsupported for both paged profiles. Paged ALiBi LSE returns are limited to
    read-only page-256 decode; updates, other profiles, and all other page sizes
    remain unsupported. Paged softcap is unsupported for updates,
    page-size-16 caches, other profiles, ALiBi, windows, and autograd. Paged
    softmax LSE is unsupported for chunked prefill, other profiles, and page
    sizes other than 16 or 256.
    """
    if (k is None) != (v is None):
        raise ValueError("k and v must be provided together when updating the KV cache")
    append_kv = k is not None
    has_rotary_metadata = rotary_cos is not None or rotary_sin is not None
    if has_rotary_metadata and not append_kv:
        raise NotImplementedError(
            "rotary embeddings are implemented only for a one-token KV-cache "
            "append or the exact two-token append"
        )
    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos and rotary_sin must be provided together")
    apply_rotary = rotary_cos is not None
    if cache_batch_idx is not None:
        raise NotImplementedError("cache_batch_idx is not implemented")
    tensor_cache_leftpad = (
        cache_leftpad if isinstance(cache_leftpad, torch.Tensor) else None
    )
    if cache_leftpad is not None and tensor_cache_leftpad is None:
        raise TypeError("cache_leftpad must be a torch.Tensor or None")
    if num_splits != 0:
        if (
            type(num_splits) is not int
            or num_splits != _EXPLICIT_SPLIT_DENSE_KVCACHE_NUM_SPLITS
        ):
            raise NotImplementedError(
                "explicit num_splits is implemented only as num_splits=16 "
                "for the full, read-only bf16 "
                "(1, 1, 16384, 32, 8, 128) dense KV-cache decode profile; "
                "pass num_splits=0 for all other calls"
            )
        if block_table is not None:
            raise NotImplementedError(
                "num_splits=16 is implemented only for a dense KV cache; "
                "paged caches are not supported"
            )
        if has_rotary_metadata:
            raise NotImplementedError(
                "num_splits=16 does not support rotary embeddings"
            )
        if append_kv:
            raise NotImplementedError(
                "num_splits=16 is implemented only for read-only KV-cache "
                "calls; updates are not supported"
            )
        if (
            isinstance(cache_seqlens, torch.Tensor)
            or tensor_cache_leftpad is not None
        ):
            raise NotImplementedError(
                "num_splits=16 does not support tensor-valued cache spans; "
                "omit cache_seqlens or pass the full cache length as a Python int"
            )
        explicit_split_spec = normalize_shape(shape, q.dtype, causal)
        requested = (
            explicit_split_spec.batch,
            explicit_split_spec.seqlen_q,
            explicit_split_spec.seqlen_k,
            explicit_split_spec.nheads_q,
            explicit_split_spec.nheads_kv,
            explicit_split_spec.head_dim,
            explicit_split_spec.dtype,
        )
        if requested != _EXPLICIT_SPLIT_DENSE_KVCACHE_PROFILE:
            raise NotImplementedError(
                "num_splits=16 is implemented only for the full, read-only "
                "bf16 (1, 1, 16384, 32, 8, 128) dense KV-cache decode profile; "
                f"got {explicit_split_spec.describe()}"
            )
        if torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (q, k_cache, v_cache)
            if isinstance(tensor, torch.Tensor)
        ):
            raise NotImplementedError(
                "num_splits=16 does not support autograd; use inference mode "
                "or detach the inputs"
            )
    # This flag changes only how rotary pairs are laid out, so it remains
    # irrelevant when rotary_cos/rotary_sin are absent.

    _reject_unsupported(
        0.0,
        window_size,
        softcap,
        alibi_slopes,
        False,
        # Paged and the exact dense profile below perform their narrower ALiBi
        # validation after shape normalization.
        allow_alibi=True,
        allowed_softcap=(
            _PAGE256_PAGED_KVCACHE_SOFTCAP
            if block_table is not None
            else None
        ),
    )
    if tensor_cache_leftpad is not None and block_table is not None:
        raise NotImplementedError(
            "cache_leftpad is implemented only for the read-only dense "
            f"{_TENSOR_LENGTH_DENSE_KVCACHE_KEY} tensor-length profile"
        )
    if block_table is not None:
        return _paged_kvcache_forward(
            q,
            k_cache,
            v_cache,
            append_kv=append_kv,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax_lse=return_softmax_lse,
            shape=shape,
        )

    spec = normalize_shape(shape, q.dtype, causal)
    has_dense_alibi = alibi_slopes is not None
    if has_dense_alibi:
        requested = (
            spec.batch,
            spec.seqlen_q,
            spec.seqlen_k,
            spec.nheads_q,
            spec.nheads_kv,
            spec.head_dim,
            spec.dtype,
        )
        if requested != _DENSE_KVCACHE_ALIBI_PROFILE:
            raise NotImplementedError(
                "dense KV-cache ALiBi is implemented only for read-only bf16 "
                "(1, 1, 4096, 32, 8, 128) decode with either causal flag; "
                f"got {spec.describe()}"
            )
        if append_kv:
            raise NotImplementedError(
                "dense KV-cache ALiBi is implemented only for read-only calls; "
                "updates are not supported"
            )
        if return_softmax_lse:
            raise NotImplementedError(
                "return_softmax_lse=True is not implemented with dense "
                "KV-cache ALiBi"
            )
    is_two_token_dense_profile = spec.key == _TWO_TOKEN_DENSE_KVCACHE_KEY
    is_four_token_dense_profile = spec.key == _FOUR_TOKEN_DENSE_KVCACHE_KEY
    if (
        not spec.is_decode
        and not is_two_token_dense_profile
        and not is_four_token_dense_profile
    ):
        raise NotImplementedError(
            "flash_attn_with_kvcache supports multi-token dense queries only "
            "for causal bf16 (1, 2, 1024, 32, 8, 128) and "
            "(1, 4, 1024, 32, 8, 128); all other dense profiles require "
            "seqlen_q=1 with a non-empty cache"
        )
    if is_four_token_dense_profile:
        if append_kv:
            raise NotImplementedError(
                "the four-token dense KV-cache profile is read-only; cache "
                "updates are not implemented"
            )
        if return_softmax_lse:
            raise NotImplementedError(
                "return_softmax_lse is not implemented for the four-token "
                "dense KV-cache profile"
            )
    if is_two_token_dense_profile:
        if return_softmax_lse:
            raise NotImplementedError(
                "return_softmax_lse is not implemented for the two-token "
                "dense KV-cache profile"
            )
    tensor_cache_seqlens = (
        cache_seqlens if isinstance(cache_seqlens, torch.Tensor) else None
    )
    if tensor_cache_leftpad is not None:
        if append_kv:
            raise NotImplementedError(
                "cache_leftpad is supported only for read-only KV-cache calls"
            )
        if tensor_cache_seqlens is None:
            raise NotImplementedError(
                "cache_leftpad requires tensor-valued cache_seqlens"
            )
        if spec.key != _TENSOR_LENGTH_DENSE_KVCACHE_KEY:
            raise NotImplementedError(
                "cache_leftpad is implemented only for the read-only dense "
                f"{_TENSOR_LENGTH_DENSE_KVCACHE_KEY} tensor-length profile"
            )
    if tensor_cache_seqlens is not None and append_kv:
        raise NotImplementedError(
            "tensor-valued cache_seqlens are supported only for read-only KV "
            "cache calls; updates require a Python int"
        )
    if cache_seqlens is not None and type(cache_seqlens) is not int:
        if tensor_cache_seqlens is None:
            raise TypeError(
                "cache_seqlens must be a Python int, a torch.Tensor, or None"
            )

    if (
        tensor_cache_seqlens is not None
        and spec.key != _TENSOR_LENGTH_DENSE_KVCACHE_KEY
    ):
        raise NotImplementedError(
            "tensor-valued cache_seqlens are implemented only for read-only "
            f"{_TENSOR_LENGTH_DENSE_KVCACHE_KEY}"
        )

    if append_kv:
        if cache_seqlens is None:
            raise ValueError(
                "cache_seqlens must be a Python int when updating the KV cache"
            )
        if cache_seqlens < 0 or cache_seqlens >= spec.seqlen_k:
            raise ValueError(
                f"cache_seqlens={cache_seqlens} is out of range for a cache "
                f"of length {spec.seqlen_k}"
            )
        if is_two_token_dense_profile:
            if cache_seqlens + 2 != spec.seqlen_k:
                raise NotImplementedError(
                    "the two-token dense KV-cache update must fill the final "
                    "two cache slots; cache_seqlens + 2 must equal the cache "
                    "length declared by shape"
                )
        elif cache_seqlens + 1 != spec.seqlen_k:
            raise NotImplementedError(
                "only a one-token update that fills the final cache slot is "
                "implemented; cache_seqlens + 1 must equal the cache length "
                "declared by shape"
            )
    elif tensor_cache_seqlens is None and (
        cache_seqlens is not None and cache_seqlens != spec.seqlen_k
    ):
        raise NotImplementedError(
            "partial or ragged scalar KV caches are not implemented for "
            "read-only calls; cache_seqlens must equal the cache length "
            "declared by shape"
        )

    # Dispatch directly after the KV-cache-specific checks. Routing back
    # through flash_attn_func would repeat normalization and validation on
    # every latency-sensitive decode step.
    check_tensors(q, k_cache, v_cache, spec)
    if k_cache.device != q.device or v_cache.device != q.device:
        raise ValueError("q, k_cache, and v_cache must be on the same CUDA device")
    if has_dense_alibi:
        assert alibi_slopes is not None
        _validate_alibi_slopes(alibi_slopes, q, spec)
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q, k_cache, v_cache, alibi_slopes)
        ):
            raise NotImplementedError(
                "ALiBi backward is not implemented; dense KV-cache ALiBi "
                "calls are forward-only"
            )
    if append_kv:
        assert k is not None and v is not None
        update_tokens = 2 if is_two_token_dense_profile else 1
        expected_update = (
            spec.batch,
            update_tokens,
            spec.nheads_kv,
            spec.head_dim,
        )
        for name, tensor in (("k", k), ("v", v)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if (
                not is_two_token_dense_profile
                and tensor.ndim == 4
                and tensor.shape[1] != 1
            ):
                raise NotImplementedError(
                    "multi-token KV-cache updates are not implemented; "
                    f"{name} must contain exactly one token"
                )
            if tuple(tensor.shape) != expected_update:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)} but a "
                    f"{update_tokens}-token update for the declared shape "
                    f"requires {expected_update}"
                )
            if tensor.dtype != spec.dtype:
                raise ValueError(
                    f"{name} has dtype {tensor.dtype} but shape declares {spec.dtype}"
                )
            if not tensor.is_cuda:
                raise ValueError(
                    f"{name} must be a CUDA tensor, got device {tensor.device}"
                )
            if tensor.device != q.device:
                raise ValueError(
                    "q, k_cache, v_cache, and update k/v must be on the same "
                    "CUDA device"
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous in "
                    f"[batch, {update_tokens}, nheads_kv, head_dim] layout"
                )
        if _contiguous_tensors_overlap(k_cache, v_cache):
            raise ValueError("k_cache and v_cache must not overlap when updating")
        if _contiguous_tensors_overlap(q, k_cache) or _contiguous_tensors_overlap(
            q, v_cache
        ):
            raise ValueError("q must not overlap k_cache or v_cache when updating")
        if not torch.is_inference_mode_enabled() and (
            k_cache.is_inference() or v_cache.is_inference()
        ):
            # PyTorch mutates an inference tensor's data before raising when an
            # in-place operation is attempted outside inference mode. Reject
            # before either copy so K and V cannot become desynchronized.
            raise RuntimeError(
                "KV caches created in torch.inference_mode() must be updated "
                "while torch.inference_mode() is enabled"
            )
        if apply_rotary:
            assert rotary_cos is not None and rotary_sin is not None
            _validate_kvcache_rotary(
                rotary_cos,
                rotary_sin,
                q,
                spec,
                rotary_interleaved=rotary_interleaved,
                full_head_interleaved_only=is_two_token_dense_profile,
            )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(spec.head_dim)
    scale = float(softmax_scale)
    if tensor_cache_seqlens is not None:
        _validate_dense_kvcache_tensor_span(
            tensor_cache_seqlens,
            tensor_cache_leftpad,
            device=q.device,
            spec=spec,
        )
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q, k_cache, v_cache)
        ):
            raise NotImplementedError(
                "tensor-valued cache_seqlens do not support autograd"
            )
        return _tensor_length_dense_kvcache_forward(
            q,
            k_cache,
            v_cache,
            tensor_cache_seqlens,
            tensor_cache_leftpad,
            scale,
            spec,
            return_softmax_lse=return_softmax_lse,
        )

    if has_dense_alibi:
        assert alibi_slopes is not None
        return _generic_dense_forward(
            q,
            k_cache,
            v_cache,
            scale,
            spec,
            alibi_slopes,
        )

    if is_four_token_dense_profile:
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q, k_cache, v_cache)
        ):
            raise NotImplementedError(
                "the four-token dense KV-cache profile does not support autograd"
            )
        return _generic_dense_forward(
            q,
            k_cache,
            v_cache,
            scale,
            spec,
            None,
        )

    if is_two_token_dense_profile:
        if torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (q, k_cache, v_cache, k, v, rotary_cos, rotary_sin)
            if tensor is not None
        ):
            raise NotImplementedError(
                "the two-token dense KV-cache profile does not support autograd"
            )
        if not append_kv:
            return _generic_dense_forward(
                q,
                k_cache,
                v_cache,
                scale,
                spec,
                None,
            )
        # This normally cannot reject the exact bounded profile, but perform
        # the generic dispatch's shape validation before either cache write.
        _validate_generic_dense_layout(spec)

    # Preserve the scalar full-cache/update dispatch: those calls still resolve
    # and launch the same checked-in specialization as before.
    kernel = None
    if not is_two_token_dense_profile:
        kernel = lookup(spec)
        needs_backward = torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (q, k_cache, v_cache, k, v, rotary_cos, rotary_sin)
            if tensor is not None
        )
        if needs_backward:
            if append_kv:
                raise NotImplementedError("KV-cache updates do not support autograd")
            lookup_backward(spec)
            return attention_autograd(q, k_cache, v_cache, scale, spec)

    # All input, feature, dispatch, and autograd validation must precede both
    # writes so a rejected call cannot leave a half-updated cache.
    q_for_attention = q
    if append_kv:
        assert k is not None and v is not None and cache_seqlens is not None
        update_tokens = 2 if is_two_token_dense_profile else 1
        cache_tensors = (k_cache, v_cache)
        if apply_rotary:
            assert rotary_cos is not None and rotary_sin is not None
            if is_two_token_dense_profile:
                update_k = _apply_consecutive_interleaved_rotary(
                    k, rotary_cos, rotary_sin, cache_seqlens
                )
                q_for_attention = _apply_consecutive_interleaved_rotary(
                    q, rotary_cos, rotary_sin, cache_seqlens
                )
            else:
                apply_rotary_fn = (
                    _apply_interleaved_rotary
                    if rotary_interleaved
                    else _apply_neox_rotary
                )
                update_k = apply_rotary_fn(
                    k, rotary_cos, rotary_sin, cache_seqlens
                )
                q_for_attention = apply_rotary_fn(
                    q, rotary_cos, rotary_sin, cache_seqlens
                )
        else:
            update_k = (
                k.clone()
                if any(_contiguous_tensors_overlap(k, cache) for cache in cache_tensors)
                else k
            )
        update_v = (
            v.clone()
            if any(_contiguous_tensors_overlap(v, cache) for cache in cache_tensors)
            else v
        )
        update_end = cache_seqlens + update_tokens
        k_cache[:, cache_seqlens:update_end].copy_(update_k)
        v_cache[:, cache_seqlens:update_end].copy_(update_v)
    if is_two_token_dense_profile:
        return _generic_dense_forward(
            q_for_attention,
            k_cache,
            v_cache,
            scale,
            spec,
            None,
        )
    assert kernel is not None
    if return_softmax_lse:
        return kernel(
            q_for_attention,
            k_cache,
            v_cache,
            scale,
            return_softmax_lse=True,
        )
    return kernel(q_for_attention, k_cache, v_cache, scale)


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qkv`` is ``[batch, seqlen, 3, nheads, head_dim]``; see :func:`flash_attn_func`."""
    if qkv.ndim != 5:
        raise ValueError(
            "qkv must be rank 5 with shape "
            "[batch, seqlen, 3, nheads, head_dim]; "
            f"got rank {qkv.ndim} and shape {tuple(qkv.shape)}"
        )
    if qkv.shape[2] != 3:
        raise ValueError(
            "qkv packed axis (dimension 2) must contain exactly 3 entries "
            f"(Q, K, V); got {qkv.shape[2]} in shape {tuple(qkv.shape)}"
        )
    q, k, v = (qkv[:, :, i].contiguous() for i in range(3))
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    shape: ShapeLike,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``kv`` is ``[batch, seqlen_k, 2, nheads_kv, head_dim]``; see :func:`flash_attn_func`."""
    if kv.ndim != 5:
        raise ValueError(
            "kv must be rank 5 with shape "
            "[batch, seqlen_k, 2, nheads_kv, head_dim]; "
            f"got rank {kv.ndim} and shape {tuple(kv.shape)}"
        )
    if kv.shape[2] != 2:
        raise ValueError(
            "kv packed axis (dimension 2) must contain exactly 2 entries "
            f"(K, V); got {kv.shape[2]} in shape {tuple(kv.shape)}"
        )
    k, v = (kv[:, :, i].contiguous() for i in range(2))
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        shape=shape,
    )
