"""Warm-up and repeated timing / memory measurement (Phase 9)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator


@dataclass
class TimingResult:
    mean_sec: float
    std_sec: float
    samples_sec: list[float] = field(default_factory=list)


@contextmanager
def measure_block() -> Generator[list[float], None, None]:
    """Collect repeated timing samples (caller appends durations)."""
    samples: list[float] = []
    yield samples


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 2,
    iterations: int = 5,
) -> TimingResult:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / max(len(samples) - 1, 1)
    return TimingResult(mean_sec=mean, std_sec=var**0.5, samples_sec=samples)
