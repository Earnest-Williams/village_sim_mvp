from __future__ import annotations

import unittest

from village_sim.core.config import SimConfig
from village_sim.sim.engine import Simulation


class SimulationTests(unittest.TestCase):
    def test_simulation_is_deterministic_for_same_seed(self) -> None:
        config = SimConfig(width=32, height=32, max_days=5, seed=123)
        result_a = Simulation(config).run()
        result_b = Simulation(config).run()

        self.assertEqual(result_a.to_json_obj(), result_b.to_json_obj())

    def test_simulation_runs_and_emits_events(self) -> None:
        config = SimConfig(width=32, height=32, max_days=3, seed=5)
        sim = Simulation(config)
        result = sim.run()

        self.assertGreater(result.days_elapsed, 0.0)
        self.assertGreater(len(sim.events), 0)
        self.assertGreaterEqual(result.water_discoveries, 0)
        self.assertGreaterEqual(result.food_discoveries, 0)

    def test_snapshot_contains_agent(self) -> None:
        config = SimConfig(width=24, height=24, max_days=1, seed=3)
        sim = Simulation(config)
        sim.step()
        snapshot = sim.snapshot(include_ascii=True)

        self.assertEqual(len(snapshot.agents), 1)
        self.assertIsNotNone(snapshot.ascii_map)


if __name__ == "__main__":
    unittest.main()
