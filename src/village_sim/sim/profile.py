"""Low-overhead runtime profiling support for simulation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TimingCategory = Literal[
    "environment",
    "perception",
    "policy_pathing",
    "goap",
    "snapshots",
    "result_aggregation",
]

TIMING_CATEGORIES: tuple[TimingCategory, ...] = (
    "environment",
    "perception",
    "policy_pathing",
    "goap",
    "snapshots",
    "result_aggregation",
)


@dataclass(slots=True)
class TimingBucket:
    """Accumulated wall-clock timing for one benchmark category."""

    seconds: float = 0.0
    calls: int = 0

    def add(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError("timing seconds must be non-negative")
        self.seconds += seconds
        self.calls += 1


@dataclass(slots=True)
class SimulationTimings:
    """Mutable timing totals collected by instrumented simulation runs."""

    buckets: dict[TimingCategory, TimingBucket] = field(default_factory=dict)

    def add(self, category: TimingCategory, seconds: float) -> None:
        bucket = self.buckets.get(category)
        if bucket is None:
            bucket = TimingBucket()
            self.buckets[category] = bucket
        bucket.add(seconds)

    def merge(self, other: SimulationTimings) -> None:
        for category in TIMING_CATEGORIES:
            bucket = other.buckets.get(category)
            if bucket is not None:
                self.add_total(category, bucket.seconds, bucket.calls)

    def add_total(self, category: TimingCategory, seconds: float, calls: int) -> None:
        if seconds < 0.0:
            raise ValueError("timing seconds must be non-negative")
        if calls < 0:
            raise ValueError("timing calls must be non-negative")
        bucket = self.buckets.get(category)
        if bucket is None:
            bucket = TimingBucket()
            self.buckets[category] = bucket
        bucket.seconds += seconds
        bucket.calls += calls

    def seconds_for(self, category: TimingCategory) -> float:
        bucket = self.buckets.get(category)
        if bucket is None:
            return 0.0
        return bucket.seconds

    def calls_for(self, category: TimingCategory) -> int:
        bucket = self.buckets.get(category)
        if bucket is None:
            return 0
        return bucket.calls

    def total_measured_seconds(self) -> float:
        total = 0.0
        for category in TIMING_CATEGORIES:
            total += self.seconds_for(category)
        return total
