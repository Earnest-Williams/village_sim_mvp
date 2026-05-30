from __future__ import annotations

import unittest

from village_sim.agent.state import (
    AgentState,
    sync_agent_from_arrays,
    sync_agent_to_arrays,
    make_agent_arrays,
)
from village_sim.core.types import ActionKind, GoalKind, Position


class AgentArraySyncTests(unittest.TestCase):
    def test_sync_from_arrays_falls_back_for_invalid_action_and_goal_ids(self) -> None:
        agent = AgentState(agent_id=1, position=Position(0, 0))
        arrays = make_agent_arrays(1)
        sync_agent_to_arrays(arrays, agent, 0)
        arrays.current_goal[0] = 999
        arrays.current_action[0] = 999

        sync_agent_from_arrays(arrays, agent, 0)

        self.assertEqual(agent.current_goal, GoalKind.EXPLORE)
        self.assertEqual(agent.current_action, ActionKind.IDLE)


if __name__ == "__main__":
    unittest.main()
