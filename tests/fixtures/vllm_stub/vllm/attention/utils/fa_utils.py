"""The two import forms used by vLLM 0.10.1.1."""

from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.vllm_flash_attn import get_scheduler_metadata
from vllm.vllm_flash_attn.flash_attn_interface import (
    fa_version_unsupported_reason,
)
from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported
