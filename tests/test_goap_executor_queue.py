"""Vectorized GOAP action queue tests."""

from __future__ import annotations

import numpy as np

from village_sim.agent.state import make_agent_arrays
from village_sim.goap.executor import (
    agents_requiring_goap_plan,
    enqueue_agent_actions,
    execute_action_queue,
)


def test_action_queue_decrements_and_limits_planning_budget() -> None:
    arrays = make_agent_arrays(4)
    arrays.active[:] = np.asarray([True, True, True, False], dtype=np.bool_)
    enqueue_agent_actions(
        arrays,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([3, 4], dtype=np.int32),
        np.asarray([2, 1], dtype=np.int32),
    )

    active = execute_action_queue(arrays)
    planners = agents_requiring_goap_plan(arrays, max_plans_per_tick=1)

    assert active.tolist() == [0, 1]
    assert arrays.action_queue_duration.tolist() == [1, 0, 0, 0]
    assert arrays.action_queue_kind.tolist() == [3, 0, 0, 0]
    assert planners.tolist() == [1]
