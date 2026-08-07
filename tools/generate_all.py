"""Autotune every catalogued shape, one job per visible GPU.

Usage: python tools/generate_all.py [--gpus 8] [--only substring] [--force]

Each job is an independent ``tools/generate.py`` process pinned to one GPU, so
a failure in one shape never takes down the campaign. Logs land in
``.autotune-logs/<shape>.log``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / ".autotune-logs"
MANIFEST = REPO_ROOT / "helion_attention" / "kernels" / "manifest.json"

sys.path.insert(0, str(REPO_ROOT / "tools"))
from shapes import CATALOGUE  # noqa: E402
from shapes import ShapeRequest  # noqa: E402


def existing_keys() -> set[str]:
    if not MANIFEST.exists():
        return set()
    with MANIFEST.open() as handle:
        return {entry["key"] for entry in json.load(handle)["kernels"]}


def expected_key(request: ShapeRequest) -> str:
    heads_kv = request.nheads_kv if request.nheads_kv is not None else request.nheads
    return (
        f"b{request.batch}_sq{request.seqlen}_sk{request.seqlen}"
        f"_hq{request.nheads}_hkv{heads_kv}_d{request.head_dim}"
        f"_{request.dtype}_{'causal' if request.causal else 'noncausal'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--only", default="", help="substring filter on the shape name")
    parser.add_argument("--force", action="store_true", help="regenerate existing kernels")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    have = set() if args.force else existing_keys()
    pending = [
        request
        for request in CATALOGUE
        if args.only in request.name and expected_key(request) not in have
    ]
    if not pending:
        print("nothing to do")
        return 0
    print(f"{len(pending)} shapes to autotune on {args.gpus} GPUs")

    running: dict[int, tuple[ShapeRequest, subprocess.Popen[bytes], float]] = {}
    queue = list(pending)
    results: list[tuple[str, bool, float]] = []
    while queue or running:
        while queue and len(running) < args.gpus:
            gpu = next(i for i in range(args.gpus) if i not in running)
            request = queue.pop(0)
            log_path = LOG_DIR / f"{request.name}.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                [sys.executable, str(REPO_ROOT / "tools" / "generate.py"), *request.args],
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
            running[gpu] = (request, process, time.time())
            print(f"[gpu {gpu}] start {request.name}", flush=True)
        time.sleep(5)
        for gpu, (request, process, started) in list(running.items()):
            if process.poll() is None:
                continue
            del running[gpu]
            elapsed = time.time() - started
            ok = process.returncode == 0
            results.append((request.name, ok, elapsed))
            print(
                f"[gpu {gpu}] {'done ' if ok else 'FAILED'} {request.name} "
                f"in {elapsed/60:.1f} min ({len(results)}/{len(pending)})",
                flush=True,
            )

    failures = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failures)}/{len(results)} shapes generated")
    for name in failures:
        print(f"  failed: {name}  (see {LOG_DIR / (name + '.log')})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
