"""Tests for discoverable perception, memory, interaction, and regrowth (§31)."""

from __future__ import annotations

import unittest

from village_sim.world.discoverables import (
    AgentNeeds,
    DiscoverableAgentMemory,
    DiscoverableKind,
    DiscoverableWorld,
    exploit_discoverable,
    make_discoverable_test_world,
    perceive_discoverables,
    update_discoverables_daily,
    update_discoverable_memory,
)


class TestDiscoverablePerception(unittest.TestCase):
    def test_agent_perceives_nearby_spring(self) -> None:
        world = make_discoverable_test_world()
        observations = perceive_discoverables(world, agent_x=10, agent_y=12, vision_radius=4)
        ids = {obs.discoverable_id for obs in observations}
        self.assertIn("spring_001", ids)

    def test_agent_does_not_perceive_distant_spring(self) -> None:
        world = make_discoverable_test_world()
        observations = perceive_discoverables(world, agent_x=0, agent_y=0, vision_radius=3)
        ids = {obs.discoverable_id for obs in observations}
        self.assertNotIn("spring_001", ids)


class TestDiscoverableMemory(unittest.TestCase):
    def test_agent_remembers_seen_discoverable(self) -> None:
        world = make_discoverable_test_world()
        memory = DiscoverableAgentMemory()
        obs = perceive_discoverables(world, agent_x=10, agent_y=12, vision_radius=4)
        update_discoverable_memory(memory, obs, tick=100)

        self.assertIn("spring_001", memory.discoverables)
        self.assertEqual(
            memory.discoverables["spring_001"].kind,
            DiscoverableKind.FRESHWATER_SPRING,
        )
        self.assertEqual(memory.discoverables["spring_001"].confidence, 1.0)

    def test_memory_updates_on_revisit(self) -> None:
        world = make_discoverable_test_world()
        memory = DiscoverableAgentMemory()
        obs = perceive_discoverables(world, agent_x=10, agent_y=12, vision_radius=4)
        update_discoverable_memory(memory, obs, tick=100)
        update_discoverable_memory(memory, obs, tick=200)

        self.assertEqual(memory.discoverables["spring_001"].last_seen_tick, 200)


class TestSpringExploitation(unittest.TestCase):
    def test_agent_drinks_from_spring(self) -> None:
        world = make_discoverable_test_world()
        spring = world.discoverables["spring_001"]
        needs = AgentNeeds(hunger=0.1, thirst=0.9, fatigue=0.1, health=1.0)

        success = exploit_discoverable(needs, spring)

        self.assertTrue(success)
        self.assertLess(needs.thirst, 0.9)

    def test_spring_does_not_deplete(self) -> None:
        world = make_discoverable_test_world()
        spring = world.discoverables["spring_001"]
        needs = AgentNeeds(hunger=0.1, thirst=0.9, fatigue=0.1, health=1.0)

        for _ in range(50):
            exploit_discoverable(needs, spring)

        self.assertEqual(spring.amount, 9999.0)


class TestBerryBushDepletion(unittest.TestCase):
    def test_berry_bush_depletes(self) -> None:
        world = make_discoverable_test_world()
        bush = world.discoverables["berry_bush_001"]
        needs = AgentNeeds(hunger=0.9, thirst=0.1, fatigue=0.1, health=1.0)

        for _ in range(4):
            self.assertTrue(exploit_discoverable(needs, bush))

        self.assertEqual(bush.amount, 0.0)
        self.assertFalse(exploit_discoverable(needs, bush))

    def test_berry_bush_reduces_hunger(self) -> None:
        world = make_discoverable_test_world()
        bush = world.discoverables["berry_bush_001"]
        needs = AgentNeeds(hunger=0.9, thirst=0.1, fatigue=0.1, health=1.0)
        before = needs.hunger

        exploit_discoverable(needs, bush)

        self.assertLess(needs.hunger, before)


class TestBerryBushRegrowth(unittest.TestCase):
    def test_berry_bush_regrows(self) -> None:
        world = make_discoverable_test_world()
        bush = world.discoverables["berry_bush_001"]
        bush.amount = 0.0

        update_discoverables_daily(world)
        update_discoverables_daily(world)

        self.assertAlmostEqual(bush.amount, 1.0)

    def test_berry_bush_does_not_exceed_max(self) -> None:
        world = make_discoverable_test_world()
        bush = world.discoverables["berry_bush_001"]

        for _ in range(100):
            update_discoverables_daily(world)

        self.assertEqual(bush.amount, bush.max_amount)


if __name__ == "__main__":
    unittest.main()
