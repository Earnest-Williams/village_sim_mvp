"""Initial agent count tests."""

from __future__ import annotations

import unittest

import numpy as np

from village_sim.agent.state import MAX_AGENTS
from village_sim.core.config import SimConfig
from village_sim.sim.engine import Simulation


class InitialAgentsTests(unittest.TestCase):
    def test_simulation_starts_requested_number_of_agents(self) -> None:
        config = SimConfig(width=16, height=16, max_days=1, seed=4, initial_agents=40)

        sim = Simulation(config)
        result = sim.result()

        self.assertEqual(int(np.count_nonzero(sim.agents.active)), 40)
        self.assertEqual(result.initial_agents, 40)
        self.assertEqual(result.final_active_agents, 40)

    def test_simulation_starts_requested_agents_below_custom_capacity(self) -> None:
        config = SimConfig(width=16, height=16, max_days=1, seed=4, initial_agents=2)

        sim = Simulation(config, max_agents=5)

        self.assertEqual(int(np.count_nonzero(sim.agents.active)), 2)

    def test_config_rejects_zero_initial_agents(self) -> None:
        config = SimConfig(initial_agents=0)

        with self.assertRaisesRegex(ValueError, "initial_agents"):
            config.validate()

    def test_simulation_rejects_agent_count_above_capacity(self) -> None:
        config = SimConfig(initial_agents=MAX_AGENTS + 1)

        with self.assertRaisesRegex(
            ValueError, rf"initial_agents \({MAX_AGENTS + 1}\).*max_agents \({MAX_AGENTS}\)"
        ):
            Simulation(config)


if __name__ == "__main__":
    unittest.main()
