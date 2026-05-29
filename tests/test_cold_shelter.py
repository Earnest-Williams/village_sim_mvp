"""Cold-stress shelter learning tests."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from village_sim.agent.needs import update_needs
from village_sim.agent.perception import Observation, perceive
from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.time import clock_from_tick
from village_sim.core.types import Position
from village_sim.orchestrator.snapshotting import make_state_snapshot
from village_sim.orchestrator.symbolic import extract_symbolic_state
from village_sim.run import print_result
from village_sim.sim.engine import Simulation
from village_sim.world.discoverables import DiscoverableAgentMemory, DiscoverableMemory
from village_sim.world.discoverables import (
    Discoverable,
    DiscoverableKind,
    exploit_nearby_discoverable,
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


def _cave_position(sim: Simulation) -> Position:
    cave: Discoverable = sim.world.discoverables["cave_001"]
    return Position(cave.x, cave.y)


class TestColdShelter(unittest.TestCase):
    def test_memory_target_prioritizes_most_critical_need(self) -> None:
        agent = AgentState(agent_id=1, position=Position(10, 10))
        agent.thirst = 0.95
        agent.hunger = 0.20
        agent.cold_stress = 0.61
        memory = DiscoverableAgentMemory(
            discoverables={
                "spring_001": DiscoverableMemory(
                    discoverable_id="spring_001",
                    kind=DiscoverableKind.FRESHWATER_SPRING,
                    x=10,
                    y=11,
                    last_seen_tick=0,
                    last_known_amount=1.0,
                    confidence=1.0,
                ),
                "berry_001": DiscoverableMemory(
                    discoverable_id="berry_001",
                    kind=DiscoverableKind.BERRY_BUSH,
                    x=20,
                    y=20,
                    last_seen_tick=0,
                    last_known_amount=1.0,
                    confidence=1.0,
                ),
                "cave_001": DiscoverableMemory(
                    discoverable_id="cave_001",
                    kind=DiscoverableKind.CAVE,
                    x=9,
                    y=10,
                    last_seen_tick=0,
                    last_known_amount=1.0,
                    confidence=1.0,
                ),
            }
        )
        clock = clock_from_tick(0, SimConfig())
        symbolic = extract_symbolic_state(
            agent=agent,
            observation=Observation(),
            disc_memory=memory,
            clock=clock,
        )

        self.assertEqual(symbolic["target_type"], "freshwater_spring")

        agent.cold_stress = 0.98
        symbolic = extract_symbolic_state(
            agent=agent,
            observation=Observation(),
            disc_memory=memory,
            clock=clock,
        )
        self.assertEqual(symbolic["target_type"], "cave")

    def test_agent_state_and_cold_update(self) -> None:
        agent = AgentState(agent_id=1, position=Position(0, 0))
        config = SimConfig()
        before = agent.cold_stress

        update_needs(agent, config, is_night=True, is_raining=False, is_sheltered=False)

        self.assertGreater(agent.cold_stress, before)
        before_shelter = agent.cold_stress

        update_needs(agent, config, is_sheltered=True)

        self.assertLess(agent.cold_stress, before_shelter)

    def test_cave_is_seeded(self) -> None:
        sim = _make_discoverable_sim()
        cave = sim.world.discoverables["cave_001"]

        self.assertEqual(cave.kind, DiscoverableKind.CAVE)
        self.assertEqual(cave.satisfies_need, "cold_stress")

    def test_cave_can_reduce_cold_stress(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = _cave_position(sim)
        sim.agent.cold_stress = 0.9

        exploited_id = exploit_nearby_discoverable(
            sim.world,
            sim.agent,
            sim.agent.position,
        )

        self.assertEqual(exploited_id, "cave_001")
        self.assertLess(sim.agent.cold_stress, 0.9)

    def test_live_cave_exploit_records_trajectory(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = _cave_position(sim)
        sim.agent.cold_stress = 0.9
        sim.agent.thirst = 0.1
        sim.agent.hunger = 0.1
        sim.agent.fatigue = 0.1
        before_cold = sim.agent.cold_stress

        sim.step()

        self.assertLess(sim.agent.cold_stress, before_cold)
        self.assertEqual(len(sim.recorded_trajectories), 1)
        self.assertEqual(sim.recorded_trajectories[0].task_name, "cold_stress")

    def test_repeated_cave_exploits_synthesize_actions(self) -> None:
        sim = _make_discoverable_sim()
        cave = sim.world.discoverables["cave_001"]
        self.assertEqual(cave.kind, DiscoverableKind.CAVE)
        for _ in range(10):
            sim.agent.position = _cave_position(sim)
            sim.agent.cold_stress = 0.9
            sim.agent.thirst = 0.1
            sim.agent.hunger = 0.1
            sim.agent.fatigue = 0.1
            sim.step()

        actions = sim.action_library.all_actions()
        action_ids = {action.action_id for action in actions}
        self.assertIn("action_exploit_cave_001_v1", action_ids)
        self.assertIn("action_exploit_cave_template_v1", action_ids)
        self.assertTrue(
            any("cold_stress_delta" in action.effects for action in actions)
        )

    def test_symbolic_snapshot_includes_cold_facts(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.position = _cave_position(sim)
        sim.agent.cold_stress = 0.9
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

        self.assertIn("cold_stress_bucket", snapshot.symbolic)
        self.assertEqual(snapshot.symbolic["known_shelter"], True)
        self.assertEqual(snapshot.symbolic["target_type"], "cave")

    def test_result_and_cli_summary_include_cold_stress(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.cold_stress = 0.42

        result = sim.result()
        output = io.StringIO()
        with redirect_stdout(output):
            print_result(result)

        self.assertEqual(result.final_cold_stress, sim.agent.cold_stress)
        self.assertIn("cold_stress=0.42", output.getvalue())

    def test_urgent_goal_chooses_highest_need_above_threshold(self) -> None:
        sim = _make_discoverable_sim()
        sim.agent.thirst = 0.61
        sim.agent.hunger = 0.10
        sim.agent.cold_stress = 0.95

        self.assertEqual(sim._urgent_goap_goal(), {"cold_stress_bucket": "low"})


if __name__ == "__main__":
    unittest.main()
