"""Stub of vLLM's package-shaped FlashAttention extension."""

from .flash_attn_interface import fa_version_unsupported_reason
from .flash_attn_interface import is_fa_version_supported

PACKAGE_MARKER = "vllm package preserved"


def flash_attn_varlen_func() -> str:
    return "original vLLM attention"


def get_scheduler_metadata() -> str:
    return "original vLLM scheduler"
