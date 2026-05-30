"""Agent handoff protocol tests."""

from __future__ import annotations

import unittest

import numpy as np

from village_sim.agent.handoff import (
    RegionBounds,
    group_departing_agents_by_neighbor,
    receive_agent_handoff,
)
from village_sim.agent.memory import GlobalMemory, ResourceMemory
from village_sim.agent.state import make_agent_arrays
from village_sim.core.types import Position, ResourceKind


class TestAgentHandoff(unittest.TestCase):
    def test_agent_arrays_can_scale_beyond_legacy_default(self) -> None:
        arrays = make_agent_arrays(10_001)

        self.assertEqual(arrays.count, 10_001)
        self.assertEqual(arrays.active.shape[0], 10_001)

    def test_departing_agent_moves_to_neighbor_with_memory(self) -> None:
        source_arrays = make_agent_arrays(4)
        source_arrays.active[:2] = True
        source_arrays.x[:2] = np.asarray([8, 3], dtype=np.int32)
        source_arrays.y[:2] = np.asarray([2, 3], dtype=np.int32)
        source_agent_ids = np.asarray([10, 11, 0, 0], dtype=np.int64)
        source_memory = GlobalMemory()
        source_memory.queue_memory(
            10,
            ResourceMemory(
                position=Position(1, 1),
                kind=ResourceKind.WATER,
                last_seen_tick=0,
                last_amount=1.0,
                confidence=0.7,
            ),
        )
        source_memory.flush_pending()

        buffers = group_departing_agents_by_neighbor(
            source_arrays,
            source_memory,
            RegionBounds(origin_x=0, origin_y=0, width=8, height=8),
            source_agent_ids,
        )

        self.assertEqual(set(buffers.keys()), {(1, 0)})
        self.assertFalse(bool(source_arrays.active[0]))
        self.assertTrue(bool(source_arrays.active[1]))
        self.assertEqual(source_memory.frame.height, 0)

        target_arrays = make_agent_arrays(4)
        target_agent_ids = np.zeros(4, dtype=np.int64)
        target_memory = GlobalMemory()
        received = receive_agent_handoff(
            target_arrays,
            target_memory,
            RegionBounds(origin_x=8, origin_y=0, width=8, height=8),
            target_agent_ids,
            buffers[(1, 0)],
        )

        self.assertEqual(received, 1)
        self.assertTrue(bool(target_arrays.active[0]))
        self.assertEqual(int(target_arrays.x[0]), 0)
        self.assertEqual(int(target_arrays.y[0]), 2)
        self.assertEqual(int(target_agent_ids[0]), 10)
        self.assertEqual(target_memory.frame.height, 1)


if __name__ == "__main__":
    unittest.main()
