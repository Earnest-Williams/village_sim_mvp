from __future__ import annotations

import random
import unittest

from village_sim.agent.decision import DecisionSource, DecisionTrace
from village_sim.agent.memory import AgentMemory, ResourceMemory
from village_sim.agent.perception import Observation
from village_sim.agent.policy import choose_and_execute_action
from village_sim.agent.state import AgentState, MemoryMarker
from village_sim.core.config import SimConfig
from village_sim.core.time import clock_from_tick
from village_sim.core.types import Position, ResourceKind, ResourceSighting, TerrainKind
from village_sim.sim.engine import Simulation
from village_sim.view.ascii_view import ROLE_COLORS, render_map_model
from village_sim.view.stc_map import STC_ROLE_STYLE
from village_sim.world.grid import index_of
from village_sim.world.world import World


class LearningVisibilityTests(unittest.TestCase):
    def test_decision_trace_for_visible_resource(self) -> None:
        config = SimConfig(width=8, height=8, seed=1)
        world = _empty_world(config.width, config.height)
        agent = AgentState(agent_id=1, position=Position(x=1, y=1), thirst=0.70)
        memory = _agent_memory()
        observation = Observation(
            visible_water=[
                ResourceSighting(
                    position=Position(x=5, y=1),
                    kind=ResourceKind.WATER,
                    amount=1.0,
                )
            ]
        )

        choose_and_execute_action(
            agent,
            memory,
            observation,
            world,
            clock_from_tick(40, config),
            random.Random(2),
            config,
        )

        self.assertEqual(agent.decision_trace.source, DecisionSource.VISIBLE_RESOURCE)
        self.assertEqual(agent.decision_trace.target_kind, "water")

    def test_decision_trace_for_remembered_resource(self) -> None:
        config = SimConfig(width=12, height=12, seed=2)
        world = _empty_world(config.width, config.height)
        agent = AgentState(agent_id=1, position=Position(x=1, y=1), thirst=0.70)
        memory = _agent_memory()
        target = Position(x=8, y=8)
        memory.resource_memories.append(
            ResourceMemory(
                position=target,
                kind=ResourceKind.WATER,
                last_seen_tick=0,
                last_amount=1.0,
                confidence=0.95,
            )
        )

        choose_and_execute_action(
            agent,
            memory,
            Observation(),
            world,
            clock_from_tick(40, config),
            random.Random(3),
            config,
        )

        self.assertEqual(
            agent.decision_trace.source, DecisionSource.REMEMBERED_RESOURCE
        )
        self.assertEqual(agent.decision_trace.target_x, target.x)
        self.assertEqual(agent.decision_trace.target_y, target.y)

    def test_decision_trace_for_search_near_memory(self) -> None:
        config = SimConfig(width=12, height=12, seed=3)
        world = _empty_world(config.width, config.height)
        agent = AgentState(agent_id=1, position=Position(x=5, y=6), hunger=0.80)
        agent.ensure_visit_buffer(world.width * world.height)
        memory = _agent_memory()
        target = Position(x=5, y=5)
        memory.resource_memories.append(
            ResourceMemory(
                position=target,
                kind=ResourceKind.FOOD,
                last_seen_tick=0,
                last_amount=1.0,
                confidence=0.95,
            )
        )

        choose_and_execute_action(
            agent,
            memory,
            Observation(),
            world,
            clock_from_tick(40, config),
            random.Random(4),
            config,
        )

        self.assertEqual(agent.decision_trace.source, DecisionSource.SEARCH_NEAR_MEMORY)
        self.assertEqual(agent.decision_trace.target_x, target.x)
        self.assertEqual(agent.decision_trace.target_y, target.y)

    def test_memory_reinforcement_stats_and_event(self) -> None:
        sim = _controlled_simulation()
        water_position = Position(x=2, y=1)
        sim.agent.position = Position(x=1, y=1)
        sim.agent.thirst = 0.80
        sim.world.water[index_of(sim.world.width, water_position)] = 1.0
        sim.memory.observe(
            ResourceSighting(
                position=water_position,
                kind=ResourceKind.WATER,
                amount=1.0,
            ),
            0,
        )

        sim.step()

        self.assertEqual(sim.learning.memory_reinforced_water, 1)
        self.assertTrue(
            any(
                event.message.startswith("reinforced water memory at 2,1")
                for event in sim.events
            )
        )

    def test_memory_failure_stats_and_event(self) -> None:
        sim = _controlled_simulation()
        target = Position(x=2, y=1)
        sim.agent.position = Position(x=1, y=1)
        sim.agent.thirst = 0.80
        sim.memory.observe(
            ResourceSighting(
                position=target,
                kind=ResourceKind.WATER,
                amount=1.0,
            ),
            0,
        )

        sim.step()

        self.assertEqual(sim.learning.memory_failed_water, 1)
        self.assertTrue(
            any(
                event.message.startswith("weakened water memory at 2,1")
                for event in sim.events
            )
        )

    def test_sim_result_learning_fields(self) -> None:
        sim = Simulation(SimConfig(width=16, height=16, max_days=1, seed=12))
        result = sim.run()

        self.assertGreaterEqual(result.learning.memory_use_ratio, 0.0)
        self.assertLessEqual(result.learning.memory_use_ratio, 1.0)
        self.assertGreaterEqual(result.learning.learned_water_sites, 0)

    def test_map_hides_resource_memory_glyphs_and_preserves_agent_priority(
        self,
    ) -> None:
        world = _empty_world(8, 8)
        agent = AgentState(agent_id=1, position=Position(x=1, y=1))
        agent.memory_markers = [
            MemoryMarker(
                position=Position(x=2, y=2),
                kind=ResourceKind.WATER,
                confidence=0.80,
            ),
            MemoryMarker(
                position=Position(x=3, y=2),
                kind=ResourceKind.FOOD,
                confidence=0.70,
            ),
            MemoryMarker(
                position=Position(x=1, y=1),
                kind=ResourceKind.WATER,
                confidence=0.95,
            ),
        ]

        rendered = render_map_model(world, agent)

        water_marker = rendered.rows[2][2]
        food_marker = rendered.rows[2][3]

        self.assertIn(water_marker.role, {'grass', 'brush'})
        self.assertIn(food_marker.role, {'grass', 'brush'})
        self.assertIn(water_marker.char, {'.', ',', '\"'})
        self.assertIn(food_marker.char, {'.', ',', '\"'})
        self.assertEqual(rendered.rows[1][1].char, "@")

    def test_replay_snapshot_decision_fields(self) -> None:
        sim = _controlled_simulation()
        sim.agent.decision_trace = DecisionTrace(
            source=DecisionSource.REMEMBERED_RESOURCE,
            target_kind="water",
            target_x=2,
            target_y=1,
            memory_confidence=0.91,
        )

        snapshot = sim.snapshot()
        agent_snapshot = snapshot.agents[0]

        self.assertEqual(agent_snapshot.decision_source, "remembered_resource")
        self.assertEqual(agent_snapshot.decision_target_kind, "water")
        self.assertEqual(agent_snapshot.decision_target_x, 2)
        self.assertEqual(agent_snapshot.decision_target_y, 1)

    def test_gui_stc_omits_resource_memory_styles(self) -> None:
        for role in ("remembered_water", "remembered_food", "stale_memory"):
            self.assertNotIn(role, ROLE_COLORS)
            self.assertNotIn(role, STC_ROLE_STYLE)


def _agent_memory() -> AgentMemory:
    return AgentMemory()


def _controlled_simulation() -> Simulation:
    config = SimConfig(width=8, height=8, max_days=1, seed=5)
    sim = Simulation(config)
    sim.world = _empty_world(config.width, config.height)
    sim.agent.position = Position(x=1, y=1)
    sim.agent.ensure_visit_buffer(sim.world.width * sim.world.height)
    return sim


def _empty_world(width: int, height: int) -> World:
    terrain: list[int] = [int(TerrainKind.GRASS) for _ in range(width * height)]
    water: list[float] = [0.0 for _ in terrain]
    food: list[float] = [0.0 for _ in terrain]
    food_capacity: list[float] = [0.0 for _ in terrain]
    return World(
        width=width,
        height=height,
        height_map=[0.5 for _ in terrain],
        terrain=terrain,
        water=water,
        food=food,
        food_capacity=food_capacity,
        seed=1,
        tile_size_meters=2.0,
    )


if __name__ == "__main__":
    unittest.main()
