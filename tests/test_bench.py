"""Benchmark runner tests."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from typing import Any

from village_sim.bench import print_report, run_benchmark
from village_sim.sim.profile import TIMING_CATEGORIES

EXPECTED_REPRESENTATIVE_OUTPUTS: list[dict[str, Any]] = [
    {
        "seed": 1,
        "ticks": 144,
        "days_elapsed": 1.0,
        "survived": True,
        "death_reason": None,
        "final_health": 1.0,
        "final_thirst": 0.6279999999999971,
        "final_hunger": 0.5009999999999963,
        "final_fatigue": 0.0,
        "final_cold_stress": 0.1440000000000001,
        "water_discoveries": 26,
        "food_discoveries": 7,
        "distance_walked": 78,
        "remembered_water_sites": 26,
        "remembered_food_sites": 7,
        "action_library_size": 0,
        "goap_plan_executions": 0,
    },
    {
        "seed": 2,
        "ticks": 144,
        "days_elapsed": 1.0,
        "survived": True,
        "death_reason": None,
        "final_health": 1.0,
        "final_thirst": 0.6279999999999972,
        "final_hunger": 0.5009999999999963,
        "final_fatigue": 0.0,
        "final_cold_stress": 0.12100000000000008,
        "water_discoveries": 197,
        "food_discoveries": 6,
        "distance_walked": 77,
        "remembered_water_sites": 197,
        "remembered_food_sites": 6,
        "action_library_size": 0,
        "goap_plan_executions": 0,
    },
    {
        "seed": 3,
        "ticks": 144,
        "days_elapsed": 1.0,
        "survived": True,
        "death_reason": None,
        "final_health": 1.0,
        "final_thirst": 0.6279999999999969,
        "final_hunger": 0.5009999999999967,
        "final_fatigue": 0.0,
        "final_cold_stress": 0.06650000000000003,
        "water_discoveries": 23,
        "food_discoveries": 8,
        "distance_walked": 72,
        "remembered_water_sites": 23,
        "remembered_food_sites": 8,
        "action_library_size": 0,
        "goap_plan_executions": 0,
    },
]


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

        self.assertEqual(actual, EXPECTED_REPRESENTATIVE_OUTPUTS)

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
