"""The shape catalogue this repository ships kernels for.

Each entry is one autotuned kernel. Adding a shape here and running
``tools/generate_all.py`` is the whole process for supporting a new one.
"""

from __future__ import annotations

from typing import NamedTuple


class ShapeRequest(NamedTuple):
    batch: int
    seqlen: int
    nheads: int
    head_dim: int
    dtype: str = "bf16"
    causal: bool = False
    nheads_kv: int | None = None
    label: str = ""
    seqlen_k: int | None = None

    @property
    def args(self) -> list[str]:
        argv = [
            "--batch", str(self.batch),
            "--seqlen", str(self.seqlen),
            "--nheads", str(self.nheads),
            "--head-dim", str(self.head_dim),
            "--dtype", self.dtype,
        ]
        if self.nheads_kv is not None:
            argv += ["--nheads-kv", str(self.nheads_kv)]
        if self.seqlen_k is not None:
            argv += ["--seqlen-k", str(self.seqlen_k)]
        if self.causal:
            argv.append("--causal")
        if self.label:
            argv += ["--label", self.label]
        return argv

    @property
    def name(self) -> str:
        kv = "" if self.nheads_kv is None else f"_hkv{self.nheads_kv}"
        sk = "" if self.seqlen_k is None else f"_sk{self.seqlen_k}"
        return (
            f"b{self.batch}_s{self.seqlen}{sk}_h{self.nheads}{kv}_d{self.head_dim}"
            f"_{self.dtype}_{'causal' if self.causal else 'noncausal'}"
        )


CATALOGUE: list[ShapeRequest] = [
    ShapeRequest(2, 1024, 32, 64, label="GPT-2 medium style prefill"),
    ShapeRequest(2, 1024, 32, 64, causal=True, label="GPT-2 medium style, causal"),
    ShapeRequest(8, 512, 16, 64, label="short-sequence encoder batch"),
    ShapeRequest(8, 2048, 16, 64, causal=True, label="small decoder, long batch"),
    ShapeRequest(4, 4096, 32, 128, label="7B-class prefill, bidirectional"),
    ShapeRequest(4, 4096, 32, 128, causal=True, label="7B-class prefill, causal"),
    ShapeRequest(1, 2048, 32, 128, causal=True, label="single-sequence 7B prefill"),
    ShapeRequest(2, 8192, 16, 128, causal=True, label="long-context causal"),
    ShapeRequest(1, 16384, 16, 128, causal=True, label="16k context, single sequence"),
    ShapeRequest(4, 4096, 32, 128, causal=True, nheads_kv=8, label="Llama-3-8B GQA 4:1"),
    ShapeRequest(2, 1024, 32, 64, dtype="fp16", label="fp16 coverage"),
    ShapeRequest(4, 4096, 32, 128, dtype="fp16", causal=True, label="fp16 causal coverage"),
    ShapeRequest(1, 2048, 32, 128, causal=True, nheads_kv=8, label="Llama-3-8B GQA 4:1"),
    ShapeRequest(1, 4096, 32, 128, causal=True, nheads_kv=8, label="Llama-3-8B GQA 4:1"),
    ShapeRequest(1, 8192, 32, 128, causal=True, nheads_kv=8, label="Llama-3-8B GQA 4:1"),
    ShapeRequest(4, 2048, 32, 128, causal=True, nheads_kv=8, label="Llama-3-8B GQA 4:1"),
    ShapeRequest(4, 8192, 32, 128, causal=True, nheads_kv=8, label="Llama-3-8B GQA 4:1"),
    ShapeRequest(1, 2048, 28, 128, causal=True, nheads_kv=4, label="Qwen2-7B GQA 7:1"),
    ShapeRequest(1, 4096, 28, 128, causal=True, nheads_kv=4, label="Qwen2-7B GQA 7:1"),
    ShapeRequest(1, 8192, 28, 128, causal=True, nheads_kv=4, label="Qwen2-7B GQA 7:1"),
    ShapeRequest(4, 2048, 28, 128, causal=True, nheads_kv=4, label="Qwen2-7B GQA 7:1"),
    ShapeRequest(4, 4096, 28, 128, causal=True, nheads_kv=4, label="Qwen2-7B GQA 7:1"),
    ShapeRequest(4, 8192, 28, 128, causal=True, nheads_kv=4, label="Qwen2-7B GQA 7:1"),
    ShapeRequest(
        1,
        1,
        32,
        128,
        causal=True,
        nheads_kv=8,
        label="Llama-3-8B single-token decode, 1k KV cache",
        seqlen_k=1024,
    ),
    ShapeRequest(
        1,
        1,
        32,
        128,
        causal=True,
        nheads_kv=8,
        label="Llama-3-8B single-token decode, 4k KV cache",
        seqlen_k=4096,
    ),
    ShapeRequest(
        1,
        1,
        32,
        128,
        causal=True,
        nheads_kv=8,
        label="Llama-3-8B single-token decode, 16k KV cache",
        seqlen_k=16384,
    ),
]
