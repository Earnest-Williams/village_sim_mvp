"""Runtime integration tests for discoverables and synthesized actions."""

from __future__ import annotations

import unittest

from village_sim.core.config import SimConfig
from village_sim.core.time import clock_from_tick
from village_sim.core.types import Position
from village_sim.orchestrator.action_model import update_confidence_after_execution
from village_sim.orchestrator.snapshotting import make_state_snapshot
from village_sim.agent.perception import perceive
from village_sim.sim.engine import Simulation
from village_sim.world.discoverables import (
    DiscoverableAgentMemory,
    DiscoverableKind,
    discoverable_at_or_adjacent,
    exploit_nearby_discoverable,
    make_initial_discoverables,
    update_discoverable_memory,
)


def _make_discoverable_sim() -> Simulation:
    config = SimConfig(
        width=32,
        height=32,
        max_days=1,
        seed=11,
        enable_initial_discoverables=True,
    )
    return Simulation(config)


class TestInitialDiscoverablesRuntime(unittest.TestCase):
    def test_config_flag_seeds_canonical_discoverables(self) -> None:
        sim = _make_discoverable_sim()

        self.assertIn("spring_001", sim.world.discoverables)
        self.assertIn("berry_bush_001", sim.world.discoverables)

    def test_default_config_does_not_seed_canonical_discoverables(self) -> None:
        config = SimConfig(width=32, height=32, max_days=1, seed=11)
        sim = Simulation(config)

        self.assertEqual(sim.world.discoverables, {})

    def test_make_initial_discoverables_returns_fresh_instances(self) -> None:
        first = make_initial_discoverables()
        second = make_initial_discoverables()
        first["berry_bush_001"].amount = 0.0

        self.assertEqual(second["berry_bush_001"].amount, 4.0)


class TestDiscoverableMemoryRuntime(unittest.TestCase):
    def test_update_discoverable_memory_returns_new_ids_once(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        clock = clock_from_tick(sim.tick, sim.config)
        observation = perceive(sim.world, sim.agent.position, clock, sim.config)

        first = update_discoverable_memory(
            sim.discoverable_memory,
            observation.discoverables,
            sim.tick,
        )
        second = update_discoverable_memory(
            sim.discoverable_memory,
            observation.discoverables,
            sim.tick + 1,
        )

        self.assertIn("spring_001", first)
        self.assertEqual(second, [])

    def test_live_step_stores_perceived_discoverables(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        sim.agent.thirst = 0.1
        sim.agent.hunger = 0.1
        sim.step()

        self.assertIn("spring_001", sim.discoverable_memory.discoverables)


class TestDiscoverableActionsRuntime(unittest.TestCase):
    def test_agent_state_exploits_nearby_spring(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        sim.agent.thirst = 0.9

        exploited_id = exploit_nearby_discoverable(
            sim.world,
            sim.agent,
            sim.agent.position,
        )

        self.assertEqual(exploited_id, "spring_001")
        self.assertLess(sim.agent.thirst, 0.9)

    def test_agent_state_exploits_nearby_berry_bush(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(20, 18)
        sim.agent.hunger = 0.9
        bush = sim.world.discoverables["berry_bush_001"]
        before_amount = bush.amount

        exploited_id = exploit_nearby_discoverable(
            sim.world,
            sim.agent,
            sim.agent.position,
        )

        self.assertEqual(exploited_id, "berry_bush_001")
        self.assertLess(sim.agent.hunger, 0.9)
        self.assertLess(bush.amount, before_amount)

    def test_depleted_berry_bush_cannot_be_exploited(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(20, 18)
        sim.agent.hunger = 0.9
        sim.world.discoverables["berry_bush_001"].amount = 0.0

        exploited_id = exploit_nearby_discoverable(
            sim.world,
            sim.agent,
            sim.agent.position,
        )

        self.assertIsNone(exploited_id)

    def test_discoverable_at_or_adjacent_prefers_adjacent(self) -> None:
        sim = _make_discoverable_sim()
        item = discoverable_at_or_adjacent(sim.world, 13, 12)

        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.discoverable_id, "spring_001")


class TestSnapshottingRuntime(unittest.TestCase):
    def test_snapshot_includes_visible_target_and_memory_facts(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        sim.agent.thirst = 0.9
        clock = clock_from_tick(sim.tick, sim.config)
        observation = perceive(sim.world, sim.agent.position, clock, sim.config)
        update_discoverable_memory(
            sim.discoverable_memory,
            observation.discoverables,
            sim.tick,
        )

        snapshot = make_state_snapshot(
            tick=sim.tick,
            agent=sim.agent,
            observation=observation,
            discoverable_memory=sim.discoverable_memory,
            clock=clock,
        )

        self.assertEqual(snapshot.tick, sim.tick)
        self.assertEqual(snapshot.x, 12)
        self.assertEqual(snapshot.y, 12)
        self.assertEqual(snapshot.needs.thirst, 0.9)
        self.assertEqual(snapshot.symbolic["target_type"], "freshwater_spring")
        self.assertEqual(snapshot.symbolic["known_water"], True)

    def test_snapshot_known_food_reflects_discoverable_memory(self) -> None:
        memory = DiscoverableAgentMemory()
        sim = _make_discoverable_sim()
        sim.agent.position = Position(20, 18)
        clock = clock_from_tick(sim.tick, sim.config)
        observation = perceive(sim.world, sim.agent.position, clock, sim.config)
        update_discoverable_memory(memory, observation.discoverables, sim.tick)

        snapshot = make_state_snapshot(
            tick=sim.tick,
            agent=sim.agent,
            observation=observation,
            discoverable_memory=memory,
            clock=clock,
        )

        self.assertEqual(snapshot.symbolic["known_food"], True)


class TestLiveTrajectoryAndGoapRuntime(unittest.TestCase):
    def test_live_spring_exploit_records_trajectory(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        sim.agent.thirst = 0.9
        before_thirst = sim.agent.thirst

        sim.step()

        self.assertLess(sim.agent.thirst, before_thirst)
        self.assertIn("spring_001", sim.discoverable_memory.discoverables)
        self.assertEqual(len(sim.recorded_trajectories), 1)
        self.assertEqual(sim.recorded_trajectories[0].task_name, "thirst")

    def test_live_spring_exploit_advances_tick_by_interaction_ticks(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        sim.agent.thirst = 0.9
        spring = sim.world.discoverables["spring_001"]

        sim.step()

        self.assertEqual(sim.tick, spring.interaction_ticks)

    def test_step_skips_depleted_discoverable_exploitation(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(20, 18)
        sim.agent.hunger = 0.9
        bush = sim.world.discoverables["berry_bush_001"]
        bush.amount = 0.0
        bush.regrowth_per_day = 0.0

        sim.step()

        self.assertEqual(len(sim.recorded_trajectories), 0)
        self.assertEqual(sim.tick, 1)
        self.assertEqual(bush.amount, 0.0)

    def test_repeated_spring_exploits_synthesize_actions_and_goap_plan(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        for _ in range(10):
            sim.agent.position = Position(12, 12)
            sim.agent.thirst = 0.9
            sim.agent.hunger = 0.1
            sim.agent.fatigue = 0.1
            sim.step()

        action_ids = {action.action_id for action in sim.action_library.all_actions()}
        self.assertIn("action_exploit_freshwater_spring_001_v1", action_ids)
        self.assertIn("action_exploit_freshwater_spring_template_v1", action_ids)

        sim.agent.position = Position(12, 12)
        sim.agent.thirst = 0.9
        steps = sim.current_goap_plan({"thirst_bucket": "low"})

        self.assertGreater(len(steps), 0)
        self.assertIn("thirst_delta", steps[0].action.effects)

        action = steps[0].action
        previous_trials = action.confidence.trials
        update_confidence_after_execution(
            action,
            success=True,
            death=False,
            timeout=False,
        )
        self.assertEqual(action.confidence.trials, previous_trials + 1)

    def test_repeated_berry_exploits_synthesize_actions(self) -> None:
        sim = _make_discoverable_sim()
        bush = sim.world.discoverables["berry_bush_001"]
        self.assertEqual(bush.kind, DiscoverableKind.BERRY_BUSH)
        for _ in range(10):
            bush.amount = bush.max_amount
            sim.agent.position = Position(20, 18)
            sim.agent.hunger = 0.9
            sim.agent.thirst = 0.1
            sim.agent.fatigue = 0.1
            sim.step()

        action_ids = {action.action_id for action in sim.action_library.all_actions()}
        self.assertIn("action_exploit_berry_bush_001_v1", action_ids)
        self.assertIn("action_exploit_berry_bush_template_v1", action_ids)

    def test_current_goap_plan_empty_when_not_at_discoverable(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = Position(12, 12)
        for _ in range(10):
            sim.agent.position = Position(12, 12)
            sim.agent.thirst = 0.9
            sim.agent.hunger = 0.1
            sim.step()

        sim.agent.position = Position(0, 0)
        sim.agent.thirst = 0.9
        steps = sim.current_goap_plan({"thirst_bucket": "low"})

        self.assertEqual(steps, [])


if __name__ == "__main__":
    unittest.main()
