"""Exercise the documented package-preserving swap under multiprocessing spawn."""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

STUB_ROOT = Path(__file__).resolve().parent / "vllm_stub"
sys.path.insert(0, str(STUB_ROOT))

from helion_attention import vllm_flash_attn as compatibility  # noqa: E402
from helion_attention.vllm_plugin import load_vllm_plugin  # noqa: E402
from vllm.attention.utils import fa_utils as _preloaded_fa_utils  # noqa: E402,F401

# Spawn re-executes this module as ``__mp_main__``. Installation remains at
# module scope so every child receives the swap, while the CLI/main body below
# is guarded against recursive execution.
os.environ["HELION_ATTENTION_VLLM"] = "1"
load_vllm_plugin()


def _worker(connection: Any) -> None:
    import vllm.vllm_flash_attn as package
    from vllm.attention.utils import fa_utils
    from vllm.vllm_flash_attn.layers import rotary

    connection.send(
        {
            "package_preserved": package.PACKAGE_MARKER == "vllm package preserved",
            "package_path_preserved": hasattr(package, "__path__"),
            "rotary_imported": rotary.ROTARY_IMPORT_SUCCEEDED,
            "attention_patched": (
                fa_utils.flash_attn_varlen_func
                is compatibility.flash_attn_varlen_func
            ),
            "scheduler_patched": (
                fa_utils.get_scheduler_metadata
                is compatibility.get_scheduler_metadata
            ),
            "feature_probe_patched": (
                fa_utils.is_fa_version_supported
                is compatibility.is_fa_version_supported
                and fa_utils.fa_version_unsupported_reason
                is compatibility.fa_version_unsupported_reason
            ),
        }
    )
    connection.close()


def main() -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child,))
    process.start()
    child.close()
    result = parent.recv()
    parent.close()
    process.join(timeout=30)

    if process.exitcode != 0:
        raise SystemExit(f"spawned worker exited with {process.exitcode}")
    failed = [name for name, passed in result.items() if not passed]
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))
    print("vLLM package-preserving spawn smoke passed")


if __name__ == "__main__":
    main()
