"""Multi-agent survival policy regression tests."""

from __future__ import annotations

import numpy as np

from village_sim.agent.state import ACTION_TO_ID
from village_sim.core.config import SimConfig
from village_sim.core.types import ActionKind
from village_sim.sim.engine import Simulation


def test_multi_agent_agents_move_or_drink_before_death() -> None:
    sim = Simulation(
        SimConfig(width=32, height=32, max_days=2, seed=1, initial_agents=5)
    )
    start_positions: list[tuple[np.int32, np.int32]] = list(
        zip(sim.agents.x.copy(), sim.agents.y.copy(), strict=False)
    )

    sim.run()

    moved: list[bool] = [
        (int(sim.agents.x[i]), int(sim.agents.y[i]))
        != (int(start_positions[i][0]), int(start_positions[i][1]))
        for i in range(5)
    ]
    assert any(moved)
    assert int(np.count_nonzero(sim.agents.active[:5])) > 0


def test_multi_agent_policy_selects_non_idle_action_after_one_tick() -> None:
    sim = Simulation(
        SimConfig(width=32, height=32, max_days=1, seed=1, initial_agents=5)
    )

    sim.step()

    active_count: int = int(np.count_nonzero(sim.agents.active))
    idle_id: int = ACTION_TO_ID[ActionKind.IDLE]
    active_actions = sim.agents.current_action[sim.agents.active]
    if active_count > 1:
        assert bool(np.any(active_actions != idle_id))
