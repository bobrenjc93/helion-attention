"""Generation-time regressions that do not require Helion or a GPU."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from helion_attention._shape import AttnShape

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATE_PATH = REPO_ROOT / "tools" / "generate.py"
SPEC = importlib.util.spec_from_file_location("helion_attention_generate", GENERATE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate)


def test_persistent_16k_kernel_is_selected_only_for_its_complete_shape() -> None:
    exact = AttnShape(
        batch=1,
        seqlen_q=16384,
        seqlen_k=16384,
        nheads_q=16,
        nheads_kv=16,
        head_dim=128,
        dtype=torch.bfloat16,
        causal=True,
    )
    assert generate.select_dense_kernel_name(exact) == "causal_attention_bshd_16k"

    incompatible = [
        replace(exact, batch=2),
        replace(exact, seqlen_q=8192, seqlen_k=8192),
        replace(exact, nheads_q=32, nheads_kv=32),
        replace(exact, nheads_kv=8),
        replace(exact, head_dim=256),
        replace(exact, nheads_q=1, nheads_kv=1, head_dim=256),
        replace(exact, dtype=torch.float16),
    ]
    for candidate in incompatible:
        assert generate.select_dense_kernel_name(candidate) == "causal_attention_bshd"

    assert (
        generate.select_dense_kernel_name(replace(exact, causal=False))
        == "attention_bshd"
    )
    assert (
        generate.select_dense_kernel_name(replace(exact, seqlen_q=1))
        == "decode_attention_bshd"
    )


def test_failed_upgrade_restores_existing_kernel_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    target = kernel_dir / "shape.py"
    backward_target = kernel_dir / "shape_backward.py"
    manifest = kernel_dir / "manifest.json"
    target.write_bytes(b"existing forward\n")
    backward_target.write_bytes(b"existing backward\n")
    old_manifest = {
        "kernels": [{"key": "shape", "backward": True, "note": "existing"}]
    }
    manifest.write_text(json.dumps(old_manifest, indent=1) + "\n")
    previous = {
        path: path.read_bytes() for path in (target, backward_target, manifest)
    }
    monkeypatch.setattr(generate, "MANIFEST", manifest)

    def fail_verification() -> int:
        assert target.read_text() == "candidate forward\n"
        assert backward_target.read_text() == "candidate backward\n"
        assert json.loads(manifest.read_text())["kernels"] == [
            {"key": "shape", "backward": True, "note": "candidate"}
        ]
        return 1

    with pytest.raises(generate.GenerationVerificationError):
        generate.install_generated_artifacts(
            target=target,
            module="candidate forward\n",
            backward_target=backward_target,
            backward_module="candidate backward\n",
            manifest_entry={"key": "shape", "backward": True, "note": "candidate"},
            verify=fail_verification,
        )

    assert {path: path.read_bytes() for path in previous} == previous


def test_forward_only_upgrade_removes_stale_backward_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    target = kernel_dir / "shape.py"
    backward_target = kernel_dir / "shape_backward.py"
    manifest = kernel_dir / "manifest.json"
    target.write_text("existing forward\n")
    backward_target.write_text("existing backward\n")
    manifest.write_text(
        json.dumps({"kernels": [{"key": "shape", "backward": True}]}) + "\n"
    )
    monkeypatch.setattr(generate, "MANIFEST", manifest)

    def verify_forward_only_candidate() -> int:
        assert target.read_text() == "candidate forward\n"
        assert not backward_target.exists()
        assert json.loads(manifest.read_text())["kernels"] == [
            {"key": "shape", "backward": False}
        ]
        return 0

    generate.install_generated_artifacts(
        target=target,
        module="candidate forward\n",
        backward_target=backward_target,
        backward_module=None,
        manifest_entry={"key": "shape", "backward": False},
        verify=verify_forward_only_candidate,
    )

    assert target.read_text() == "candidate forward\n"
    assert not backward_target.exists()
    assert json.loads(manifest.read_text())["kernels"] == [
        {"key": "shape", "backward": False}
    ]


def test_generator_rejects_multi_cta_cooperative_grid() -> None:
    unsafe = (
        "_launcher(kernel, (_NUM_SM * 4,), launch_cooperative_grid=True)"
    )
    with pytest.raises(RuntimeError, match="4 CTAs per SM"):
        generate.require_portable_cooperative_grid(unsafe)

    generate.require_portable_cooperative_grid(
        "_launcher(kernel, (_NUM_SM,), launch_cooperative_grid=True)"
    )


def test_strip_helion_removes_varlen_constexpr_annotations() -> None:
    source = (
        "import helion.language as hl\n"
        "def varlen_attention(max_seqlen: hl.constexpr, causal: hl.constexpr):\n"
        "    return max_seqlen\n"
    )
    rewritten = generate.strip_helion(source)
    assert "helion" not in rewritten
    assert "hl.constexpr" not in rewritten
    assert "def varlen_attention(max_seqlen, causal):" in rewritten


def test_varlen_artifact_uses_separate_manifest_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    target = kernel_dir / "varlen_shape.py"
    backward_target = kernel_dir / "varlen_shape_backward.py"
    manifest = kernel_dir / "manifest.json"
    dense_entry = {"key": "dense_shape"}
    manifest.write_text(json.dumps({"kernels": [dense_entry]}) + "\n")
    monkeypatch.setattr(generate, "MANIFEST", manifest)

    generate.install_generated_artifacts(
        target=target,
        module="generated varlen forward\n",
        backward_target=backward_target,
        backward_module=None,
        manifest_entry={"key": "varlen_shape", "varlen": True},
        manifest_section="varlen_kernels",
        verify=lambda: 0,
    )

    payload = json.loads(manifest.read_text())
    assert payload["kernels"] == [dense_entry]
    assert payload["varlen_kernels"] == [
        {"key": "varlen_shape", "varlen": True}
    ]


def test_paged_artifact_uses_separate_manifest_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    target = kernel_dir / "paged_shape.py"
    backward_target = kernel_dir / "paged_shape_backward.py"
    manifest = kernel_dir / "manifest.json"
    dense_entry = {"key": "dense_shape"}
    manifest.write_text(json.dumps({"kernels": [dense_entry]}) + "\n")
    monkeypatch.setattr(generate, "MANIFEST", manifest)

    generate.install_generated_artifacts(
        target=target,
        module="generated paged forward\n",
        backward_target=backward_target,
        backward_module=None,
        manifest_entry={"key": "paged_shape", "paged": True, "page_size": 16},
        manifest_section="paged_kernels",
        verify=lambda: 0,
    )

    payload = json.loads(manifest.read_text())
    assert payload["kernels"] == [dense_entry]
    assert payload["paged_kernels"] == [
        {"key": "paged_shape", "paged": True, "page_size": 16}
    ]
