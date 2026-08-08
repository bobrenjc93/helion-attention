"""Generation-time regressions that do not require Helion or a GPU."""

from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from helion_attention._shape import AttnShape

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATE_PATH = REPO_ROOT / "tools" / "generate.py"
SPEC = importlib.util.spec_from_file_location("helion_attention_generate", GENERATE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate)
GENERATE_ALL_PATH = REPO_ROOT / "tools" / "generate_all.py"
GENERATE_ALL_SPEC = importlib.util.spec_from_file_location(
    "helion_attention_generate_all", GENERATE_ALL_PATH
)
assert GENERATE_ALL_SPEC is not None and GENERATE_ALL_SPEC.loader is not None
generate_all = importlib.util.module_from_spec(GENERATE_ALL_SPEC)
GENERATE_ALL_SPEC.loader.exec_module(generate_all)


def test_provenance_is_structured_and_rendered_in_module_docstring() -> None:
    config = {
        "block_sizes": [128, 64],
        "num_stages": 3,
        "num_warps": 8,
        "pid_type": "flat",
    }
    provenance = generate.make_provenance(
        helion_version="1.4.0",
        selection="autotuned",
        configs=[("attention_bshd", config)],
        wall_time_seconds=125.6784,
        measured_time_ms=0.1067024,
    )

    assert provenance == {
        "helion_version": "1.4.0",
        "selection": "autotuned",
        "configs": [{"kernel": "attention_bshd", "config": config}],
        "autotuning_wall_time_seconds": 125.678,
        "measured_time_ms": 0.106702,
    }
    module = generate.build_module(
        "from __future__ import annotations\n\nVALUE = 1\n",
        command="python tools/generate.py --batch 1",
        description="test shape",
        spec_fields={"key": "test"},
        provenance=provenance,
    )
    docstring = ast.get_docstring(ast.parse(module))
    assert docstring is not None
    assert "Helion version: 1.4.0" in docstring
    assert "Autotuning wall time: 125.678 s" in docstring
    assert "Measured time: 0.106702 ms" in docstring
    assert (
        "helion.Config(block_sizes=[128, 64], num_stages=3, num_warps=8, "
        "pid_type='flat')" in docstring
    )


def test_checked_in_modules_publish_manifest_provenance() -> None:
    manifest = json.loads(generate.MANIFEST.read_text())
    for section in ("kernels", "varlen_kernels", "paged_kernels"):
        for entry in manifest[section]:
            artifacts = [
                (
                    generate.KERNELS_DIR / f"{entry['key']}.py",
                    entry["autotuning_provenance"],
                )
            ]
            if entry.get("backward"):
                artifacts.append(
                    (
                        generate.KERNELS_DIR / f"{entry['key']}_backward.py",
                        entry["backward_autotuning_provenance"],
                    )
                )
            for path, provenance in artifacts:
                assert provenance["helion_version"]
                assert provenance["selection"] in {"autotuned", "fixed"}
                assert provenance["configs"]
                assert provenance["autotuning_wall_time_seconds"] >= 0
                assert provenance["measured_time_ms"] > 0
                docstring = ast.get_docstring(ast.parse(path.read_text()))
                assert docstring is not None
                assert generate.format_provenance(provenance) in docstring


def test_catalogue_backfills_missing_forward_and_backward_provenance() -> None:
    request = generate_all.ShapeRequest(1, 64, 8, 128)
    forward_provenance = {"helion_version": "1.4.0"}

    assert not generate_all.entry_is_complete(request, None)
    assert not generate_all.entry_is_complete(request, {"key": "shape"})
    assert generate_all.entry_is_complete(
        request, {"key": "shape", "autotuning_provenance": forward_provenance}
    )

    backward_request = request._replace(backward=True)
    assert not generate_all.entry_is_complete(
        backward_request,
        {
            "key": "shape",
            "backward": True,
            "autotuning_provenance": forward_provenance,
        },
    )
    assert generate_all.entry_is_complete(
        backward_request,
        {
            "key": "shape",
            "backward": True,
            "autotuning_provenance": forward_provenance,
            "backward_autotuning_provenance": forward_provenance,
        },
    )


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
        == "causal_attention_bshd"
    )
    assert (
        generate.select_dense_kernel_name(
            replace(exact, seqlen_q=64, seqlen_k=320)
        )
        == "causal_attention_bshd"
    )


def test_split_kv_decode_is_selected_only_for_validated_shape() -> None:
    long_decode = AttnShape(
        batch=1,
        seqlen_q=1,
        seqlen_k=16384,
        nheads_q=32,
        nheads_kv=8,
        head_dim=128,
        dtype=torch.bfloat16,
        causal=True,
    )
    assert (
        generate.select_dense_kernel_name(long_decode)
        == "decode_attention_bshd_split_kv"
    )

    unsupported = [
        replace(long_decode, batch=2),
        replace(long_decode, seqlen_k=32768),
        replace(long_decode, nheads_q=28, nheads_kv=4),
        replace(long_decode, head_dim=256),
        replace(long_decode, dtype=torch.float16),
    ]
    for candidate in unsupported:
        assert generate.select_dense_kernel_name(candidate) == "causal_attention_bshd"

    for cache_length in (1024, 4096):
        assert (
            generate.select_dense_kernel_name(
                replace(long_decode, seqlen_k=cache_length)
            )
            == "causal_attention_bshd"
        )

    assert (
        generate.select_dense_kernel_name(replace(long_decode, causal=False))
        == "attention_bshd"
    )


def test_decode_autotuning_rejects_persistent_launches() -> None:
    flat = SimpleNamespace(pid_type="flat")
    xyz = SimpleNamespace(pid_type="xyz")
    persistent = SimpleNamespace(pid_type="persistent_interleaved")

    assert generate.nonpersistent_decode_config(flat) is flat
    assert generate.nonpersistent_decode_config(xyz) is xyz
    assert generate.nonpersistent_decode_config(persistent) is None


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


def test_split_kv_modules_are_composed_without_constexpr_collisions() -> None:
    partial = (
        "from __future__ import annotations\n"
        "_BLOCK_SIZE = 128\n"
        "def partial(): return _BLOCK_SIZE\n"
    )
    combine = (
        "from __future__ import annotations\n"
        "_BLOCK_SIZE = 8\n"
        "def combine(): return _BLOCK_SIZE\n"
    )
    partial = generate.namespace_generated_constants(partial, "PARTIAL")
    combine = generate.namespace_generated_constants(combine, "COMBINE")
    merged = generate.merge_generated_modules(partial, combine)

    assert merged.count("from __future__ import annotations") == 1
    assert "_PARTIAL_BLOCK_SIZE = 128" in merged
    assert "_COMBINE_BLOCK_SIZE = 8" in merged
    assert f"num_splits = {generate.SPLIT_KV_DECODE_SPLITS}" in (
        generate._SPLIT_KV_DECODE_ENTRY_POINT
    )


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
