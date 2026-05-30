"""Benchmark runner tests."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from village_sim.bench import print_report, run_benchmark
from village_sim.sim.profile import TIMING_CATEGORIES

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bench_representative_outputs.json"


class TestBenchmarkRunner(unittest.TestCase):
    def test_representative_seed_outputs_match_fixture(self) -> None:
        report = run_benchmark(
            seeds=3,
            days=1,
            width=16,
            height=16,
            seed_start=1,
            discoverables=False,
            goap=False,
            snapshot_every=0,
        )

        actual: list[dict[str, Any]] = []
        for run in report.runs:
            result = run.result
            actual.append(
                {
                    "seed": result.seed,
                    "ticks": run.ticks,
                    "days_elapsed": result.days_elapsed,
                    "survived": result.survived,
                    "death_reason": result.death_reason,
                    "final_health": result.final_health,
                    "final_thirst": result.final_thirst,
                    "final_hunger": result.final_hunger,
                    "final_fatigue": result.final_fatigue,
                    "final_cold_stress": result.final_cold_stress,
                    "water_discoveries": result.water_discoveries,
                    "food_discoveries": result.food_discoveries,
                    "distance_walked": result.distance_walked,
                    "remembered_water_sites": result.remembered_water_sites,
                    "remembered_food_sites": result.remembered_food_sites,
                    "action_library_size": result.action_library_size,
                    "goap_plan_executions": result.goap_plan_executions,
                }
            )

        expected = json.loads(FIXTURE_PATH.read_text())
        self.assertEqual(actual, expected)

    def test_report_includes_performance_counters_and_timing_splits(self) -> None:
        report = run_benchmark(
            seeds=1,
            days=1,
            width=16,
            height=16,
            seed_start=9,
            discoverables=False,
            goap=False,
            snapshot_every=0,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            print_report(report)

        text = output.getvalue()
        self.assertIn("ticks/sec:", text)
        self.assertIn("sims/sec:", text)
        self.assertIn("total wall time:", text)
        for category in TIMING_CATEGORIES:
            if category == "policy_pathing":
                label = "policy/pathing"
            else:
                label = category.replace("_", " ")
            self.assertIn(f"  {label}:", text)


if __name__ == "__main__":
    unittest.main()
