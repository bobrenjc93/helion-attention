"""Install the vLLM adapter without replacing vLLM's package tree.

vLLM imports attention calls from :mod:`vllm.vllm_flash_attn`, while model
implementations also import helpers such as
``vllm.vllm_flash_attn.layers.rotary``.  Replacing the package in
``sys.modules`` therefore breaks model loading.  This module patches only the
four Python exports helion-attention implements and leaves the package, its
``__path__``, compiled extensions, and child modules intact.

An opt-in wrapper is registered as a ``vllm.general_plugins`` entry point.
With ``HELION_ATTENTION_VLLM=1``, vLLM installs the swap in each engine and
worker process, including spawned and Ray workers.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from types import ModuleType
from typing import Any

_PACKAGE_EXPORTS = (
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
    "fa_version_unsupported_reason",
)
_INTERFACE_EXPORTS = (
    "is_fa_version_supported",
    "fa_version_unsupported_reason",
)
_MISSING = object()
_ENABLE_ENV = "HELION_ATTENTION_VLLM"


def install_vllm_flash_attn() -> None:
    """Patch vLLM's FlashAttention exports with helion-attention's adapter.

    The operation is idempotent.  Cached ``from ... import ...`` bindings in
    already-loaded vLLM modules are updated only when they still refer to one
    of the original vLLM exports.
    """
    compatibility = import_module("helion_attention.vllm_flash_attn")
    package = import_module("vllm.vllm_flash_attn")
    interface = import_module("vllm.vllm_flash_attn.flash_attn_interface")

    replacements = {
        name: getattr(compatibility, name) for name in _PACKAGE_EXPORTS
    }
    originals: dict[str, list[Any]] = {name: [] for name in replacements}

    for target, names in (
        (package, _PACKAGE_EXPORTS),
        (interface, _INTERFACE_EXPORTS),
    ):
        for name in names:
            original = getattr(target, name, _MISSING)
            if original is not _MISSING:
                originals[name].append(original)
            setattr(target, name, replacements[name])

    # The plugin normally runs before attention backends import these names.
    # Updating identity-matching cached bindings also makes explicit launchers
    # safe when vLLM was partially imported before installation.
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("vllm.") or not isinstance(module, ModuleType):
            continue
        for name, replacement in replacements.items():
            current = getattr(module, name, _MISSING)
            if any(current is original for original in originals[name]):
                setattr(module, name, replacement)


def load_vllm_plugin() -> None:
    """Install from vLLM's plugin hook when the swap was explicitly enabled."""
    if os.environ.get(_ENABLE_ENV) == "1":
        install_vllm_flash_attn()


__all__ = ["install_vllm_flash_attn", "load_vllm_plugin"]
