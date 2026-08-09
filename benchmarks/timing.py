"""Timing defaults shared by the benchmark harness and its renderers."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_ROUNDS = 3
WARMUP_MS = 50
MEASUREMENT_MS = 200


def timing_metadata(rounds: int = DEFAULT_ROUNDS) -> dict[str, int]:
    """Return the timing parameters persisted in benchmark JSON reports."""
    return {
        "rounds": rounds,
        "warmup_ms": WARMUP_MS,
        "measurement_ms": MEASUREMENT_MS,
    }


def timing_from_report(report: Mapping[str, object]) -> tuple[int, int, int]:
    """Read report timing, falling back for artifacts created before it was recorded."""
    timing = report.get("timing")
    if timing is None:
        return DEFAULT_ROUNDS, WARMUP_MS, MEASUREMENT_MS
    if not isinstance(timing, Mapping):
        raise TypeError("benchmark report timing must be an object")
    return (
        int(timing["rounds"]),
        int(timing["warmup_ms"]),
        int(timing["measurement_ms"]),
    )


def timing_methodology(
    rounds: int = DEFAULT_ROUNDS,
    warmup_ms: int = WARMUP_MS,
    measurement_ms: int = MEASUREMENT_MS,
) -> str:
    """Describe the round-robin timing protocol in report-ready prose."""
    number_words = {1: "one", 2: "two", 3: "three"}
    round_count = number_words.get(rounds, str(rounds))
    round_label = "round" if rounds == 1 else "rounds"
    return (
        f"Times are the median of {round_count} interleaved {round_label}, "
        f"each with {warmup_ms} ms warmup and {measurement_ms} ms measurement; "
        "lower is better."
    )


def report_timing_methodology(report: Mapping[str, object]) -> str:
    """Describe the timing parameters recorded in a benchmark report."""
    return timing_methodology(*timing_from_report(report))
