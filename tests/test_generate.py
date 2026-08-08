"""Generation-time regressions that do not require Helion or a GPU."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType
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


def test_retained_incumbent_preserves_origin_and_records_rejected_search() -> None:
    incumbent = {"block_sizes": [256, 64], "num_warps": 8}
    candidate = {"block_sizes": [256, 32], "num_warps": 8}
    origin = generate.make_provenance(
        helion_version="1.3.0",
        selection="autotuned",
        configs=[("attention_bshd", incumbent)],
        wall_time_seconds=412.25,
        measured_time_ms=0.293216,
    )
    provenance = generate.retain_incumbent_provenance(
        origin,
        helion_version="1.4.0",
        candidate_configs=[("attention_bshd", candidate)],
        wall_time_seconds=779.924,
        incumbent_time_ms=0.29598,
        candidate_time_ms=0.34879,
    )

    assert provenance["helion_version"] == "1.3.0"
    assert provenance["artifact_origin_selection"] == "autotuned"
    assert provenance["autotuning_wall_time_seconds"] == 412.25
    assert provenance["measured_time_ms"] == 0.293216
    assert provenance["rejected_search"] == {
        "helion_version": "1.4.0",
        "configs": [{"kernel": "attention_bshd", "config": candidate}],
        "autotuning_wall_time_seconds": 779.924,
        "candidate_measured_time_ms": 0.34879,
        "incumbent_measured_time_ms": 0.29598,
    }
    rendered = generate.format_provenance(provenance)
    assert "Config selection: incumbent" in rendered
    assert "Helion version: 1.3.0" in rendered
    assert "Rejected search:\n        Helion version: 1.4.0" in rendered
    assert "Candidate time: 0.348790 ms" in rendered
    assert generate.provenance_config(provenance, "attention_bshd") == incumbent


def test_candidate_requires_a_material_speedup() -> None:
    assert generate.candidate_is_faster(1.0, 0.97)
    assert not generate.candidate_is_faster(1.0, 0.99)
    assert not generate.candidate_is_faster(1.0, 1.01)


def test_incumbent_config_is_prepended_to_existing_search_seeds() -> None:
    existing = {"block_sizes": [64], "num_warps": 4}
    incumbent = {"block_sizes": [128], "num_warps": 8}
    kernel = SimpleNamespace(settings=SimpleNamespace(autotune_seed_configs=existing))

    selected = generate.seed_incumbent_config(
        kernel, incumbent, config_factory=lambda **values: values
    )

    assert generate.config_dict(selected) == incumbent
    assert [
        generate.config_dict(config) for config in kernel.settings.autotune_seed_configs
    ] == [incumbent, existing]


def install_fake_benchmarking(
    monkeypatch: pytest.MonkeyPatch,
    *,
    do_bench,  # noqa: ANN001
    interleaved_bench,  # noqa: ANN001
) -> None:
    helion = ModuleType("helion")
    autotuner = ModuleType("helion.autotuner")
    benchmarking = ModuleType("helion.autotuner.benchmarking")
    benchmarking.do_bench = do_bench  # type: ignore[attr-defined]
    benchmarking.interleaved_bench = interleaved_bench  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "helion", helion)
    monkeypatch.setitem(sys.modules, "helion.autotuner", autotuner)
    monkeypatch.setitem(sys.modules, "helion.autotuner.benchmarking", benchmarking)


def test_legacy_fixed_artifact_is_compared_without_helion_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def interleaved(runners, *, repeat, desc):  # noqa: ANN001, ANN202
        assert repeat == generate.AUTOTUNE_ACCEPTANCE_REPEAT
        assert "incumbent" in desc
        observed.extend(runner() for runner in runners)
        return 1.0, 1.2

    install_fake_benchmarking(
        monkeypatch,
        do_bench=lambda runner, **kwargs: 1.2,
        interleaved_bench=interleaved,
    )

    class Bound:
        def __call__(self):  # noqa: ANN204
            return "candidate"

    kernel = SimpleNamespace(
        name="fixed_attention",
        configs=[{"block_sizes": [64]}],
        bind=lambda args: Bound(),
    )
    result, provenance = generate.run_with_provenance(
        kernel,
        (),
        helion_version="1.4.0",
        incumbent=lambda: "incumbent",
    )

    assert observed == ["incumbent", "candidate"]
    assert result == "incumbent"
    assert provenance["artifact_origin_unrecorded"] is True
    assert provenance["artifact_origin_selection"] is None
    assert provenance["helion_version"] is None
    assert provenance["configs"] == []
    assert provenance["rejected_search"]["helion_version"] == "1.4.0"


def test_explicit_replacement_override_skips_incumbent_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_interleaved(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("replacement override must skip incumbent benchmarking")

    def benchmark(runner, **kwargs):  # noqa: ANN001, ANN003, ANN202
        assert runner() == "candidate"
        return 0.2

    install_fake_benchmarking(
        monkeypatch,
        do_bench=benchmark,
        interleaved_bench=fail_interleaved,
    )
    kernel = SimpleNamespace(
        name="fixed_attention",
        configs=[{"block_sizes": [64]}],
        bind=lambda args: lambda: "candidate",
    )
    origin = generate.make_provenance(
        helion_version="1.3.0",
        selection="fixed",
        configs=[("fixed_attention", {"block_sizes": [128]})],
        wall_time_seconds=0.0,
        measured_time_ms=0.1,
    )

    result, provenance = generate.run_with_provenance(
        kernel,
        (),
        helion_version="1.4.0",
        incumbent_provenance=origin,
        incumbent=lambda: "incumbent",
        replace_incumbent=True,
    )

    assert result == "candidate"
    assert provenance["helion_version"] == "1.4.0"
    assert provenance["selection"] == "fixed"


def test_incumbent_loader_keeps_legacy_artifact_without_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "shape.py"
    artifact.write_text("generated module\n")
    runner = lambda: "incumbent"
    monkeypatch.setattr(generate, "load_incumbent_runner", lambda *args, **kwargs: runner)

    inputs = generate.incumbent_artifact_inputs(
        artifact=artifact,
        provenance=None,
        kernel_key="shape",
        entry_point="attention",
        args=(),
    )

    assert inputs == {"incumbent_provenance": None, "incumbent": runner}
    assert generate.incumbent_artifact_inputs(
        artifact=artifact,
        provenance=None,
        kernel_key="shape",
        entry_point="attention",
        args=(),
        replace_incumbent=True,
    ) == {}


def test_generate_all_force_forwards_replacement_override() -> None:
    request = generate_all.ShapeRequest(1, 64, 8, 128)

    ordinary = generate_all.generation_command(request)
    forced = generate_all.generation_command(request, replace_incumbent=True)

    assert "--replace-incumbent" not in ordinary
    assert forced[:-1] == ordinary
    assert forced[-1] == "--replace-incumbent"


def test_refresh_module_provenance_preserves_executable_body() -> None:
    original = generate.make_provenance(
        helion_version="1.4.0",
        selection="autotuned",
        configs=[("attention_bshd", {"block_sizes": [64]})],
        wall_time_seconds=10.0,
        measured_time_ms=0.2,
    )
    retained = generate.retain_incumbent_provenance(
        original,
        helion_version="1.4.0",
        candidate_configs=[("attention_bshd", {"block_sizes": [128]})],
        wall_time_seconds=20.0,
        incumbent_time_ms=0.1,
        candidate_time_ms=0.2,
    )
    module = generate.build_module(
        "from __future__ import annotations\n\nVALUE = 1\n",
        command="python tools/generate.py --batch 1",
        description="test shape",
        spec_fields={"key": "test"},
        provenance=original,
    )
    executable = module[module.index("from __future__ import annotations") :]

    refreshed = generate.refresh_module_provenance(module, retained)

    assert (
        refreshed[refreshed.index("from __future__ import annotations") :] == executable
    )
    assert generate.format_provenance(retained) in ast.get_docstring(
        ast.parse(refreshed)
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
                assert provenance["selection"] in {
                    "autotuned",
                    "fixed",
                    "incumbent",
                }
                assert provenance["configs"]
                if provenance.get("artifact_origin_unrecorded"):
                    assert provenance["helion_version"] is None
                    wall_time = provenance["autotuning_wall_time_seconds"]
                    assert wall_time is None or wall_time == 0
                else:
                    assert provenance["helion_version"]
                    assert provenance["autotuning_wall_time_seconds"] >= 0
                assert provenance["measured_time_ms"] > 0
                if provenance["selection"] == "incumbent":
                    assert provenance["artifact_origin_selection"] in {
                        "autotuned",
                        "fixed",
                    }
                    rejected = provenance["rejected_search"]
                    assert rejected["helion_version"]
                    assert rejected["configs"]
                    assert rejected["autotuning_wall_time_seconds"] >= 0
                    assert rejected["candidate_measured_time_ms"] > 0
                    assert rejected["incumbent_measured_time_ms"] > 0
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


def test_persistent_kernel_is_selected_only_for_validated_shapes() -> None:
    persistent_16k = AttnShape(
        batch=1,
        seqlen_q=16384,
        seqlen_k=16384,
        nheads_q=16,
        nheads_kv=16,
        head_dim=128,
        dtype=torch.bfloat16,
        causal=True,
    )
    persistent_qwen_2k = AttnShape(
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        nheads_q=28,
        nheads_kv=4,
        head_dim=128,
        dtype=torch.bfloat16,
        causal=True,
    )
    persistent_qwen_4k = replace(
        persistent_qwen_2k,
        seqlen_q=4096,
        seqlen_k=4096,
    )
    persistent_qwen_b4_4k = replace(
        persistent_qwen_4k,
        batch=4,
    )
    persistent_qwen_b4_8k = replace(
        persistent_qwen_b4_4k,
        seqlen_q=8192,
        seqlen_k=8192,
    )
    persistent_mha_2k = AttnShape(
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        nheads_q=32,
        nheads_kv=32,
        head_dim=128,
        dtype=torch.bfloat16,
        causal=True,
    )
    persistent_llama_2k = replace(
        persistent_mha_2k,
        nheads_kv=8,
    )
    for exact in (
        persistent_16k,
        persistent_qwen_2k,
        persistent_qwen_4k,
        persistent_qwen_b4_4k,
        persistent_qwen_b4_8k,
        persistent_llama_2k,
        persistent_mha_2k,
    ):
        assert (
            generate.select_dense_kernel_name(exact)
            == "causal_attention_bshd_16k"
        )

    incompatible = [
        replace(persistent_16k, batch=2),
        replace(persistent_16k, seqlen_q=8192, seqlen_k=8192),
        replace(persistent_16k, nheads_q=32, nheads_kv=32),
        replace(persistent_16k, nheads_kv=8),
        replace(persistent_16k, head_dim=256),
        replace(persistent_16k, nheads_q=1, nheads_kv=1, head_dim=256),
        replace(persistent_16k, dtype=torch.float16),
        replace(persistent_qwen_2k, batch=2),
        replace(persistent_qwen_2k, head_dim=64),
        replace(persistent_qwen_2k, dtype=torch.float16),
        replace(persistent_qwen_4k, batch=2),
        replace(persistent_qwen_4k, seqlen_q=8192, seqlen_k=8192),
        replace(persistent_qwen_4k, nheads_q=32, nheads_kv=8),
        replace(persistent_qwen_4k, head_dim=64),
        replace(persistent_qwen_4k, dtype=torch.float16),
        replace(persistent_qwen_b4_4k, batch=2),
        replace(persistent_qwen_b4_4k, seqlen_q=2048, seqlen_k=2048),
        replace(persistent_qwen_b4_4k, nheads_q=32, nheads_kv=8),
        replace(persistent_qwen_b4_4k, head_dim=64),
        replace(persistent_qwen_b4_4k, dtype=torch.float16),
        replace(persistent_qwen_b4_8k, batch=1),
        replace(persistent_qwen_b4_8k, nheads_q=32, nheads_kv=8),
        replace(persistent_qwen_b4_8k, head_dim=64),
        replace(persistent_qwen_b4_8k, dtype=torch.float16),
        replace(persistent_llama_2k, batch=2),
        replace(persistent_llama_2k, seqlen_q=4096, seqlen_k=4096),
        replace(persistent_llama_2k, nheads_kv=4),
        replace(persistent_llama_2k, head_dim=64),
        replace(persistent_llama_2k, dtype=torch.float16),
        replace(persistent_mha_2k, batch=2),
        replace(persistent_mha_2k, seqlen_q=4096, seqlen_k=4096),
        replace(persistent_mha_2k, head_dim=64),
        replace(persistent_mha_2k, dtype=torch.float16),
    ]
    for candidate in incompatible:
        assert generate.select_dense_kernel_name(candidate) == "causal_attention_bshd"

    assert (
        generate.select_dense_kernel_name(replace(persistent_16k, causal=False))
        == "attention_bshd"
    )
    assert (
        generate.select_dense_kernel_name(replace(persistent_16k, seqlen_q=1))
        == "causal_attention_bshd"
    )
    assert (
        generate.select_dense_kernel_name(
            replace(persistent_16k, seqlen_q=64, seqlen_k=320)
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
        path: path.read_bytes() for path in (target, backward_target)
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
    assert json.loads(manifest.read_text()) == old_manifest


def test_failed_parallel_verification_preserves_successful_manifest_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    manifest = kernel_dir / "manifest.json"
    old_a = {"key": "shape_a", "note": "old a"}
    manifest.write_text(json.dumps({"kernels": [old_a]}) + "\n")
    monkeypatch.setattr(generate, "MANIFEST", manifest)
    a_target = kernel_dir / "shape_a.py"
    a_backward = kernel_dir / "shape_a_backward.py"
    b_target = kernel_dir / "shape_b.py"
    b_backward = kernel_dir / "shape_b_backward.py"
    a_target.write_text("old a\n")
    a_verifying = threading.Event()
    allow_a_failure = threading.Event()

    def fail_a_after_b_succeeds() -> int:
        a_verifying.set()
        assert allow_a_failure.wait(timeout=5)
        return 1

    def install_a() -> None:
        generate.install_generated_artifacts(
            target=a_target,
            module="candidate a\n",
            backward_target=a_backward,
            backward_module=None,
            manifest_entry={"key": "shape_a", "note": "candidate a"},
            verify=fail_a_after_b_succeeds,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        failed_a = executor.submit(install_a)
        assert a_verifying.wait(timeout=5)
        generate.install_generated_artifacts(
            target=b_target,
            module="candidate b\n",
            backward_target=b_backward,
            backward_module=None,
            manifest_entry={"key": "shape_b", "note": "candidate b"},
            verify=lambda: 0,
        )
        allow_a_failure.set()
        with pytest.raises(generate.GenerationVerificationError):
            failed_a.result(timeout=5)

    assert a_target.read_text() == "old a\n"
    assert b_target.read_text() == "candidate b\n"
    assert json.loads(manifest.read_text())["kernels"] == [
        old_a,
        {"key": "shape_b", "note": "candidate b"},
    ]


def test_manifest_publication_is_atomic_for_unlocked_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    manifest = kernel_dir / "manifest.json"
    old_payload = {"kernels": [{"key": "shape_a", "note": "old"}]}
    old_text = json.dumps(old_payload) + "\n"
    manifest.write_text(old_text)
    monkeypatch.setattr(generate, "MANIFEST", manifest)
    first_half_written = threading.Event()
    allow_publication = threading.Event()
    real_dumps = json.dumps

    def split_dump(payload, handle, **kwargs):  # noqa: ANN001, ANN202
        serialized = real_dumps(payload, **kwargs)
        midpoint = len(serialized) // 2
        handle.write(serialized[:midpoint])
        handle.flush()
        first_half_written.set()
        assert allow_publication.wait(timeout=5)
        handle.write(serialized[midpoint:])

    monkeypatch.setattr(generate.json, "dump", split_dump)
    replacement = {"key": "shape_a", "note": "new"}
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(generate.upsert_manifest, replacement)
        assert first_half_written.wait(timeout=5)
        # Runtime registry readers do not take the writer lock. They must still
        # see a complete old snapshot while the new temporary file is partial.
        assert manifest.read_text() == old_text
        assert json.loads(manifest.read_text()) == old_payload
        allow_publication.set()
        assert writer.result(timeout=5) == old_payload["kernels"][0]

    assert json.loads(manifest.read_text()) == {"kernels": [replacement]}


def test_atomic_manifest_publication_serializes_concurrent_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    manifest = kernel_dir / "manifest.json"
    manifest.write_text('{"kernels": []}\n')
    monkeypatch.setattr(generate, "MANIFEST", manifest)
    start = threading.Barrier(2)

    def publish(entry: dict[str, object]) -> None:
        start.wait(timeout=5)
        generate.upsert_manifest(entry)

    entries = [{"key": "shape_a"}, {"key": "shape_b"}]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, entry) for entry in entries]
        for future in futures:
            future.result(timeout=5)

    assert json.loads(manifest.read_text()) == {"kernels": entries}


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


def test_direct_decode_generation_exposes_online_softmax_lse() -> None:
    source = '''"""Config: helion.Config(block_sizes=[8, 128], num_warps=4)"""

def _helion_causal_attention_bshd(q, k, v, out, qk_scale, _RDIM_SIZE_3: tl.constexpr):
        # src[helion_kernels.py:270]: out[b, tile_m, h, :] = result.to(out.dtype)
        tl.store(out + tl.broadcast_to(offset_1 * 128 + (0 + indices_3)[None, :] * 1, [_BLOCK_SIZE_2, _RDIM_SIZE_3]), v_20, None)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, *, _launcher=_default_launcher):
    out = torch.empty_like(q)
    _launcher(_helion_causal_attention_bshd, q, k, v, out, qk_scale, _RDIM_SIZE_3, num_warps=4)
    return out'''
    spec = AttnShape(1, 1, 2048, 32, 8, 128, torch.bfloat16, True)

    rewritten = generate.add_direct_decode_lse_support(source, spec)

    assert "STORE_LSE: tl.constexpr" in rewritten
    assert "libdevice.log2(l_i)" in rewritten
    assert "STORE_LSE=return_softmax_lse" in rewritten
    assert "offset_1 + tl.arange(0, 1), lse, None" in rewritten
    assert "return (out, softmax_lse)" in rewritten
    assert "helion.Config(block_sizes=[8, 128], num_warps=4)" in rewritten


def test_noncausal_decode_generation_supports_arbitrary_dense_shape() -> None:
    source = """def _helion_attention_bshd(q, k, v, out, qk_scale, _NUM_SM: tl.constexpr, _RDIM_SIZE_3: tl.constexpr):
            # src[helion_kernels.py:203]: out[b, tile_m, h, :] = acc.to(out.dtype)
            out_desc = tl.make_tensor_descriptor(out, shape, strides, block_shape)
            # src[helion_kernels.py:203]: out[b, tile_m, h, :] = acc.to(out.dtype)
            out_desc.store([offset_0, offset_2, offset_1, 0], value)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, *, _launcher=_default_launcher):
    out = torch.empty_like(q)
    _launcher(_helion_attention_bshd, q, k, v, out, qk_scale, _NUM_SM, _RDIM_SIZE_3, num_warps=8)
    return out"""
    spec = AttnShape(3, 1, 3072, 12, 3, 80, torch.float16, False)

    rewritten = generate.add_direct_decode_lse_support(source, spec)

    assert "_helion_attention_bshd" in rewritten
    assert (
        "softmax_lse + offset_0 * 12 + offset_1 + tl.arange(0, 1)"
        in rewritten
    )
    assert "return_softmax_lse: bool = False" in rewritten


def test_rewritten_noncausal_decode_executes_without_query_index() -> None:
    source = """def _helion_attention_bshd(q, k, v, out, qk_scale, _RDIM_SIZE_3: tl.constexpr):
    offset_1 = 0
    m_i = torch.tensor([2.0])
    l_i = torch.tensor([4.0])
    tl.store(out + tl.arange(0, 1), torch.zeros(1), None)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, *, _launcher=_default_launcher):
    out = torch.empty_like(q)
    qk_scale = sm_scale
    _RDIM_SIZE_3 = q.size(3)
    _launcher(_helion_attention_bshd, q, k, v, out, qk_scale, _RDIM_SIZE_3, num_warps=4)
    return out"""
    spec = AttnShape(1, 1, 128, 2, 1, 8, torch.float16, False)
    rewritten = generate.add_direct_decode_lse_support(source, spec)
    # Execute the kernel body with a tiny TL shim so name resolution is tested
    # hermetically; the former `indices_2` rewrite raises NameError here.
    stores = []
    fake_tl = SimpleNamespace(
        constexpr=object(),
        arange=lambda start, stop: torch.arange(start, stop),
        store=lambda pointer, value, mask: stores.append((pointer, value, mask)),
    )

    def fake_launcher(kernel, *args, **kwargs):  # noqa: ANN001, ANN202
        kwargs.pop("num_warps")
        kernel(*args, **kwargs)

    namespace = {
        "_default_launcher": fake_launcher,
        "libdevice": SimpleNamespace(log2=torch.log2),
        "tl": fake_tl,
        "torch": torch,
    }
    exec(compile(rewritten, "<rewritten-noncausal-decode>", "exec"), namespace)
    q = torch.ones((1, 1, 2, 8), dtype=torch.float16)
    k = v = torch.ones((1, 128, 1, 8), dtype=torch.float16)

    out, softmax_lse = namespace["attention"](
        q, k, v, 0.5, return_softmax_lse=True
    )

    assert out.shape == q.shape
    assert softmax_lse.shape == (1, 2, 1)
    assert softmax_lse.dtype == torch.float32
    assert len(stores) == 2


def test_single_head_decode_generation_omits_folded_head_offset() -> None:
    source = """def _helion_causal_attention_bshd(q, k, v, out, qk_scale, _RDIM_SIZE_3: tl.constexpr):
        # src[helion_kernels.py:270]: out[b, tile_m, h, :] = result.to(out.dtype)
        tl.store(out + indices_3, value, None)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, *, _launcher=_default_launcher):
    out = torch.empty_like(q)
    _launcher(_helion_causal_attention_bshd, q, k, v, out, qk_scale, _RDIM_SIZE_3, num_warps=4)
    return out"""
    spec = AttnShape(1, 1, 128, 1, 1, 64, torch.bfloat16, True)

    rewritten = generate.add_direct_decode_lse_support(source, spec)

    assert "tl.store(softmax_lse + tl.arange(0, 1), lse, None)" in rewritten
    assert "offset_1" not in rewritten


def test_decode_lse_generation_is_gated_by_layout_not_shape_key() -> None:
    decode = AttnShape(3, 1, 3072, 12, 3, 80, torch.float16, False)

    assert generate.should_add_direct_decode_lse(
        decode, paged=False, varlen=False, split_kv=False
    )
    assert not generate.should_add_direct_decode_lse(
        decode, paged=True, varlen=False, split_kv=False
    )
    assert not generate.should_add_direct_decode_lse(
        decode, paged=False, varlen=True, split_kv=False
    )
    assert not generate.should_add_direct_decode_lse(
        decode, paged=False, varlen=False, split_kv=True
    )
    assert not generate.should_add_direct_decode_lse(
        replace(decode, seqlen_q=64),
        paged=False,
        varlen=False,
        split_kv=False,
    )


def test_split_decode_generation_exposes_combined_softmax_lse() -> None:
    source = """def _helion_decode_attention_bshd_split_kv_combine(partial_stats, partial_acc, out, _RDIM_SIZE_2: tl.constexpr, _RDIM_SIZE_3: tl.constexpr):
    tl.store(out + (offset_1 * 128 + (0 + indices_3)[None, :] * 1), v_6, None)

def _attention_split_kv_combine(partial_acc: torch.Tensor, partial_stats: torch.Tensor, out: torch.Tensor, *, _launcher=_default_launcher):
    _launcher(kernel, partial_stats, partial_acc, out, _RDIM_SIZE_2, _RDIM_SIZE_3, num_warps=1)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, *, _launcher=_default_launcher):
    out = torch.empty_like(q)
    with context():
        return _attention_split_kv_combine(
            partial_acc, partial_stats, out, _launcher=launch
        )"""

    rewritten = generate.add_split_decode_lse_support(source)

    assert "STORE_LSE: tl.constexpr" in rewritten
    assert "libdevice.log2(denominator)" in rewritten
    assert "STORE_LSE=return_softmax_lse" in rewritten
    assert "return (out, softmax_lse)" in rewritten


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
