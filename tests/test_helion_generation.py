"""GPU regressions for compiling fresh kernels from the Helion source."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
requires_generation = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("helion") is None,
    reason="needs a GPU and the optional Helion generation dependency",
)


@requires_generation
def test_equal_length_causal_source_regenerates_first_tile() -> None:
    """A fresh 64x64 specialization must execute the first causal tile."""
    script = textwrap.dedent(
        """
        import math
        import sys

        import helion
        import torch

        sys.path.insert(0, "tools")
        import helion_kernels

        config = helion.Config(
            block_sizes=[64, 64],
            num_stages=3,
            num_warps=4,
            pid_type="flat",
        )
        kernel = helion.kernel(
            helion_kernels.causal_attention_bshd.fn,
            config=config,
            settings=helion_kernels.causal_attention_bshd.settings,
        )
        generator = torch.Generator(device="cuda").manual_seed(2026)
        q = torch.randn(
            (1, 64, 8, 128),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        k = torch.randn(
            (1, 64, 2, 128),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        v = torch.randn(
            (1, 64, 2, 128),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        scale = 1.0 / math.sqrt(128)
        actual = kernel(q, k, v, scale)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            is_causal=True,
            enable_gqa=True,
            scale=scale,
        ).transpose(1, 2)
        torch.testing.assert_close(
            actual.float(), expected, atol=5e-2, rtol=2e-2
        )
        assert actual[:, :1].float().abs().max().item() > 0
        """
    )
    with tempfile.TemporaryDirectory(
        prefix=".helion-generation-", dir=REPO_ROOT
    ) as cache_dir:
        env = os.environ.copy()
        env["HELION_CACHE_DIR"] = cache_dir
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        env["TRITON_CACHE_DIR"] = cache_dir
        env["XDG_CACHE_HOME"] = cache_dir
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
