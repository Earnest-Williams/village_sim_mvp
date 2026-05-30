"""Performance benchmark runner for deterministic village simulations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

from village_sim.core.config import SimConfig
from village_sim.sim.engine import Simulation
from village_sim.sim.metrics import SimResult
from village_sim.sim.profile import TIMING_CATEGORIES, SimulationTimings, TimingCategory


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Result and timing data from one benchmarked simulation."""

    seed: int
    ticks: int
    result: SimResult
    timings: SimulationTimings


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate benchmark metrics for a group of sequential simulation runs."""

    runs: list[BenchmarkRun]
    timings: SimulationTimings
    wall_seconds: float

    @property
    def ticks(self) -> int:
        return sum(run.ticks for run in self.runs)

    @property
    def sims(self) -> int:
        return len(self.runs)

    @property
    def ticks_per_second(self) -> float:
        if self.wall_seconds <= 0.0:
            return 0.0
        return float(self.ticks) / self.wall_seconds

    @property
    def sims_per_second(self) -> float:
        if self.wall_seconds <= 0.0:
            return 0.0
        return float(self.sims) / self.wall_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Village Sim performance across sequential seeds."
    )
    parser.add_argument(
        "--seeds", type=int, required=True, help="number of seeds to run"
    )
    parser.add_argument(
        "--days", type=int, required=True, help="simulated days per seed"
    )
    parser.add_argument("--width", type=int, required=True, help="world width in cells")
    parser.add_argument(
        "--height", type=int, required=True, help="world height in cells"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="first deterministic RNG seed; later runs increment by one",
    )
    parser.add_argument(
        "--discoverables",
        action="store_true",
        help="seed the live world with canonical discoverables",
    )
    parser.add_argument(
        "--goap",
        action="store_true",
        help="enable bounded GOAP control during benchmark runs",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="store ASCII snapshots every N ticks; 0 disables periodic snapshots",
    )
    return parser


def run_benchmark(
    *,
    seeds: int,
    days: int,
    width: int,
    height: int,
    seed_start: int,
    discoverables: bool,
    goap: bool,
    snapshot_every: int,
) -> BenchmarkReport:
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    if snapshot_every < 0:
        raise ValueError("snapshot_every must be non-negative")

    runs: list[BenchmarkRun] = []
    aggregate_timings = SimulationTimings()
    wall_start: float = perf_counter()
    for offset in range(seeds):
        seed = seed_start + offset
        timings = SimulationTimings()
        config = SimConfig(
            width=width,
            height=height,
            max_days=days,
            seed=seed,
            enable_initial_discoverables=discoverables,
            enable_goap_control=goap,
        )
        sim = Simulation(config, profiler=timings)
        result = sim.run(snapshot_every=snapshot_every)
        runs.append(
            BenchmarkRun(
                seed=seed,
                ticks=sim.tick,
                result=result,
                timings=timings,
            )
        )
        aggregate_timings.merge(timings)

    aggregation_start: float = perf_counter()
    aggregation_seconds = perf_counter() - aggregation_start
    aggregate_timings.add("result_aggregation", aggregation_seconds)
    wall_seconds = perf_counter() - wall_start
    return BenchmarkReport(
        runs=runs,
        timings=aggregate_timings,
        wall_seconds=wall_seconds,
    )


def print_report(report: BenchmarkReport) -> None:
    survived_count: int = sum(1 for run in report.runs if run.result.survived)
    average_days: float = sum(run.result.days_elapsed for run in report.runs) / float(
        report.sims
    )
    average_distance: float = sum(
        run.result.distance_walked for run in report.runs
    ) / float(report.sims)

    print("Benchmark:")
    print(f"  sims: {report.sims}")
    print(f"  ticks: {report.ticks}")
    print(f"  total wall time: {report.wall_seconds:.6f} sec")
    print(f"  ticks/sec: {report.ticks_per_second:.2f}")
    print(f"  sims/sec: {report.sims_per_second:.2f}")
    print(f"  survived full duration: {survived_count}/{report.sims}")
    print(f"  average days elapsed: {average_days:.2f}")
    print(f"  average distance walked: {average_distance:.1f}")
    print("Timing splits:")
    measured_seconds = report.timings.total_measured_seconds()
    for category in TIMING_CATEGORIES:
        print(_format_timing_line(category, report.timings, measured_seconds))


def _format_timing_line(
    category: TimingCategory,
    timings: SimulationTimings,
    measured_seconds: float,
) -> str:
    seconds = timings.seconds_for(category)
    calls = timings.calls_for(category)
    percentage = 0.0
    if measured_seconds > 0.0:
        percentage = seconds / measured_seconds * 100.0
    label = category.replace("_", "/" if category == "policy_pathing" else " ")
    return f"  {label}: {seconds:.6f} sec ({percentage:.1f}%, calls={calls})"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_benchmark(
        seeds=args.seeds,
        days=args.days,
        width=args.width,
        height=args.height,
        seed_start=args.seed_start,
        discoverables=args.discoverables,
        goap=args.goap,
        snapshot_every=args.snapshot_every,
    )
    print_report(report)


if __name__ == "__main__":
    main()
