"""Model a fresh Ray worker receiving only vLLM-propagated configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

STUB_ROOT = Path(__file__).resolve().parent / "vllm_stub"
sys.path.insert(0, str(STUB_ROOT))

from helion_attention import vllm_flash_attn as compatibility  # noqa: E402
from helion_attention.vllm_plugin import load_vllm_plugin  # noqa: E402
from vllm.attention.utils import fa_utils  # noqa: E402
import vllm.vllm_flash_attn as package  # noqa: E402

if "HELION_ATTENTION_VLLM" in os.environ:
    raise SystemExit("worker unexpectedly received the obsolete custom flag")

expect_patched = sys.argv[1] == "patched"
load_vllm_plugin()

patched = (
    fa_utils.flash_attn_varlen_func is compatibility.flash_attn_varlen_func
    and fa_utils.get_scheduler_metadata is compatibility.get_scheduler_metadata
    and fa_utils.is_fa_version_supported is compatibility.is_fa_version_supported
)
if patched != expect_patched:
    raise SystemExit(
        f"expected patched={expect_patched}, got patched={patched}; "
        f"VLLM_PLUGINS={os.environ.get('VLLM_PLUGINS')!r}"
    )

from vllm.vllm_flash_attn.layers import rotary  # noqa: E402

if not hasattr(package, "__path__") or not rotary.ROTARY_IMPORT_SUCCEEDED:
    raise SystemExit("vLLM's package tree was not preserved")

print(f"Ray-style worker plugin check passed: patched={patched}")
