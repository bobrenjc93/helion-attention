"""Timing defaults shared by the benchmark harness and its renderers."""

from __future__ import annotations

DEFAULT_ROUNDS = 3
WARMUP_MS = 50
MEASUREMENT_MS = 200


def timing_methodology(rounds: int = DEFAULT_ROUNDS) -> str:
    """Describe the round-robin timing protocol in report-ready prose."""
    number_words = {1: "one", 2: "two", 3: "three"}
    round_count = number_words.get(rounds, str(rounds))
    round_label = "round" if rounds == 1 else "rounds"
    return (
        f"Times are the median of {round_count} interleaved {round_label}, "
        f"each with {WARMUP_MS} ms warmup and {MEASUREMENT_MS} ms measurement; "
        "lower is better."
    )
