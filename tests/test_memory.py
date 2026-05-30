from __future__ import annotations

import unittest

from village_sim.agent.memory import AgentMemory
from village_sim.core.config import SimConfig
from village_sim.core.types import Position, ResourceKind, ResourceSighting


class MemoryTests(unittest.TestCase):
    def test_observe_records_new_resource(self) -> None:
        memory = AgentMemory()
        sighting = ResourceSighting(Position(2, 3), ResourceKind.WATER, 1.0)

        is_new = memory.observe(sighting, tick=10)

        self.assertTrue(is_new)
        self.assertEqual(len(memory.resource_memories), 1)
        self.assertEqual(memory.resource_memories[0].position, Position(2, 3))

    def test_best_memory_prefers_confident_nearby_resource(self) -> None:
        memory = AgentMemory()
        config = SimConfig()
        memory.observe(
            ResourceSighting(Position(2, 3), ResourceKind.WATER, 1.0), tick=1
        )
        memory.observe(
            ResourceSighting(Position(50, 50), ResourceKind.WATER, 1.0), tick=1
        )

        best = memory.best_memory(
            ResourceKind.WATER, Position(0, 0), tick=2, config=config
        )

        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.position, Position(2, 3))

    def test_food_memory_decays_more_than_water(self) -> None:
        config = SimConfig(ticks_per_day=144)
        water = ResourceSighting(Position(1, 1), ResourceKind.WATER, 1.0)
        food = ResourceSighting(Position(2, 2), ResourceKind.FOOD, 1.0)
        memory = AgentMemory()
        memory.observe(water, tick=0)
        memory.observe(food, tick=0)

        water_memory = memory.resource_memories[0]
        food_memory = memory.resource_memories[1]
        later_tick = config.ticks_per_day * 5

        self.assertGreater(
            water_memory.decayed_confidence(later_tick, config),
            food_memory.decayed_confidence(later_tick, config),
        )

    def test_update_after_eviction_uses_consistent_lookup(self) -> None:
        memory = AgentMemory(capacity=2)
        memory.observe(ResourceSighting(Position(0, 0), ResourceKind.WATER, 1.0), tick=1)
        memory.observe(ResourceSighting(Position(1, 1), ResourceKind.FOOD, 1.0), tick=2)
        memory.observe(ResourceSighting(Position(2, 2), ResourceKind.WATER, 1.0), tick=3)

        memory.observe(ResourceSighting(Position(1, 1), ResourceKind.FOOD, 0.25), tick=4)

        self.assertEqual(len(memory.resource_memories), 2)
        self.assertNotIn(Position(0, 0), [m.position for m in memory.resource_memories])
        self.assertEqual(memory.resource_memories[0].position, Position(1, 1))
        self.assertAlmostEqual(memory.resource_memories[0].last_amount, 0.25)


if __name__ == "__main__":
    unittest.main()
