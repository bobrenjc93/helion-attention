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
    backward: bool = False
    varlen: bool = False
    paged: bool = False
    page_size: int = 16

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
        if self.backward:
            argv.append("--backward")
        if self.varlen:
            argv.append("--varlen")
        if self.paged:
            argv += ["--paged", "--page-size", str(self.page_size)]
        if self.label:
            argv += ["--label", self.label]
        return argv

    @property
    def name(self) -> str:
        kv = "" if self.nheads_kv is None else f"_hkv{self.nheads_kv}"
        sk = "" if self.seqlen_k is None else f"_sk{self.seqlen_k}"
        prefix = "paged_" if self.paged else "varlen_" if self.varlen else ""
        suffix = f"_ps{self.page_size}" if self.paged else ""
        return prefix + (
            f"b{self.batch}_s{self.seqlen}{sk}_h{self.nheads}{kv}_d{self.head_dim}"
            f"_{self.dtype}_{'causal' if self.causal else 'noncausal'}"
        ) + suffix


CATALOGUE: list[ShapeRequest] = [
    ShapeRequest(16, 512, 12, 64, label="BERT-base encoder"),
    ShapeRequest(8, 16, 8, 32, causal=True, label="small-model decoder"),
    ShapeRequest(8, 32, 8, 32, causal=True, label="small-model decoder"),
    ShapeRequest(8, 64, 8, 32, causal=True, label="small-model decoder"),
    ShapeRequest(8, 128, 8, 32, causal=True, label="small-model decoder"),
    ShapeRequest(8, 256, 8, 32, causal=True, label="small-model decoder"),
    ShapeRequest(8, 512, 8, 32, causal=True, label="small-model decoder"),
    ShapeRequest(8, 1024, 8, 32, causal=True, label="small-model decoder"),
    ShapeRequest(2, 1024, 32, 64, label="GPT-2 medium style prefill"),
    ShapeRequest(2, 1024, 32, 64, causal=True, label="GPT-2 medium style, causal"),
    ShapeRequest(
        8,
        512,
        16,
        64,
        label="short-sequence encoder batch",
        backward=True,
    ),
    ShapeRequest(8, 2048, 16, 64, causal=True, label="small decoder, long batch"),
    ShapeRequest(2, 1024, 16, 256, label="vision/multimodal encoder"),
    ShapeRequest(4, 4096, 32, 128, label="7B-class prefill, bidirectional"),
    ShapeRequest(4, 4096, 32, 128, causal=True, label="7B-class prefill, causal"),
    ShapeRequest(1, 2048, 32, 128, causal=True, label="single-sequence 7B prefill"),
    ShapeRequest(2, 8192, 16, 128, causal=True, label="long-context causal"),
    ShapeRequest(1, 16384, 16, 128, causal=True, label="16k context, single sequence"),
    ShapeRequest(4, 4096, 32, 128, causal=True, nheads_kv=8, label="Llama-3-8B GQA 4:1"),
    ShapeRequest(2, 1024, 32, 64, dtype="fp16", label="fp16 coverage"),
    ShapeRequest(4, 4096, 32, 128, dtype="fp16", causal=True, label="fp16 causal coverage"),
    ShapeRequest(
        1,
        64,
        32,
        128,
        causal=True,
        nheads_kv=8,
        label="Llama-3-8B 64-token GQA prefill",
    ),
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
    ShapeRequest(
        1,
        64,
        8,
        128,
        causal=True,
        nheads_kv=2,
        label="vLLM chunked-prefill causal alignment",
        seqlen_k=320,
    ),
    ShapeRequest(
        8,
        512,
        16,
        64,
        label="ragged encoder batch",
        varlen=True,
    ),
    ShapeRequest(
        8,
        512,
        16,
        64,
        causal=True,
        label="ragged decoder batch",
        varlen=True,
    ),
    ShapeRequest(
        4,
        1,
        8,
        128,
        causal=True,
        nheads_kv=2,
        label="vLLM paged ragged decode",
        seqlen_k=1024,
        paged=True,
    ),
    ShapeRequest(
        2,
        200,
        8,
        128,
        causal=True,
        nheads_kv=2,
        label="vLLM paged chunked prefill",
        seqlen_k=320,
        paged=True,
    ),
]
