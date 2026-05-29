"""End-to-end tests for induced GOAP discoverable travel and exploitation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from village_sim.agent.perception import perceive
from village_sim.core.config import SimConfig
from village_sim.core.time import clock_from_tick
from village_sim.core.types import Position
from village_sim.goap.executor import PlanExecutor
from village_sim.goap.planner import PlanStep
from village_sim.orchestrator.action_model import (
    ActionConfidence,
    ActionLibrary,
    ActionLifecycle,
    ActionScope,
    CostModel,
    ExecutionPayload,
    ExecutorType,
    SynthesizedAction,
    TargetBinding,
)
from village_sim.orchestrator.symbolic import extract_symbolic_state
from village_sim.world.discoverables import update_discoverable_memory
from village_sim.sim.engine import Simulation


def _make_sim() -> Simulation:
    return Simulation(
        SimConfig(
            seed=1,
            width=32,
            height=32,
            max_days=4,
            enable_initial_discoverables=True,
            enable_goap_control=True,
        )
    )


def _remember_discoverable(sim: Simulation, position: Position) -> None:
    sim.agent.position = position
    clock = clock_from_tick(sim.tick, sim.config)
    observation = perceive(sim.world, sim.agent.position, clock, sim.config)
    update_discoverable_memory(
        sim.discoverable_memory, observation.discoverables, sim.tick
    )


def _seed_pathfinder_action(target_id: str, target_type: str) -> SynthesizedAction:
    return SynthesizedAction(
        schema_version=1,
        action_id=f"seed_move_to_{target_id}",
        display_name="seed move",
        scope=ActionScope.INSTANCE,
        lifecycle=ActionLifecycle.CANDIDATE,
        preconditions={
            "has_target_location": True,
            "at_known_target": False,
            "target_id": target_id,
            "target_type": target_type,
        },
        soft_preconditions={},
        effects={},
        side_effects={},
        cost_model=CostModel(base_ticks=12.0, distance_weight=1.0),
        confidence=ActionConfidence(
            trials=10,
            successful_trials=10,
            failed_trials=0,
            success_rate=1.0,
            death_rate=0.0,
            timeout_rate=0.0,
        ),
        execution_payload=ExecutionPayload(
            type=ExecutorType.PATHFINDER,
            policy_id="policy_move_to_known_discoverable_v1",
            policy_version=1,
            target_binding=TargetBinding(mode="resource_id", resource_id=target_id),
        ),
    )


def _synthesize_spring_exploit_actions(sim: Simulation) -> None:
    for _ in range(10):
        sim.agent.position = Position(12, 12)
        sim.agent.thirst = 0.9
        sim.agent.hunger = 0.1
        sim.agent.fatigue = 0.1
        sim.step()


def _synthesize_spring_travel_actions(sim: Simulation) -> None:
    action = _seed_pathfinder_action("spring_001", "freshwater_spring")
    for _ in range(10):
        executor = PlanExecutor(sim)
        sim.agent.position = Position(0, 0)
        sim.agent.thirst = 0.1
        sim.agent.hunger = 0.1
        sim.agent.fatigue = 0.1
        result = executor.execute_step(PlanStep(action=action, expected_cost=1.0))
        self_recorded = result.trajectory is not None
        if self_recorded:
            assert result.trajectory is not None
            sim.recorded_trajectories.append(result.trajectory)
            sim.orchestrator.record(result.trajectory)
            for synthesized in sim.orchestrator.synthesize_all():
                sim.action_library.add(synthesized)


class TestSymbolicKnownTargetFacts(unittest.TestCase):
    def test_memory_target_facts_when_agent_is_away_from_spring(self) -> None:
        sim = _make_sim()
        _remember_discoverable(sim, Position(12, 12))
        sim.agent.position = Position(0, 0)
        sim.agent.thirst = 0.9
        clock = clock_from_tick(sim.tick, sim.config)
        observation = perceive(sim.world, sim.agent.position, clock, sim.config)

        symbolic = extract_symbolic_state(
            sim.agent,
            observation,
            sim.discoverable_memory,
            clock,
        )

        self.assertEqual(symbolic["has_target_location"], True)
        self.assertEqual(symbolic["target_id"], "spring_001")
        self.assertEqual(symbolic["target_type"], "freshwater_spring")
        self.assertNotEqual(symbolic["distance_to_target_bucket"], "at")
        self.assertEqual(symbolic["at_known_target"], False)

    def test_memory_target_facts_when_agent_is_adjacent_to_spring(self) -> None:
        sim = _make_sim()
        _remember_discoverable(sim, Position(12, 12))
        sim.agent.position = Position(11, 12)
        sim.agent.thirst = 0.9
        clock = clock_from_tick(sim.tick, sim.config)
        observation = perceive(sim.world, sim.agent.position, clock, sim.config)

        symbolic = extract_symbolic_state(
            sim.agent,
            observation,
            sim.discoverable_memory,
            clock,
        )

        self.assertEqual(symbolic["at_known_target"], True)
        self.assertEqual(symbolic["distance_to_target_bucket"], "at")


class TestGoapEndToEnd(unittest.TestCase):
    def test_travel_trajectories_synthesize_move_actions(self) -> None:
        sim = _make_sim()
        _remember_discoverable(sim, Position(12, 12))
        _synthesize_spring_travel_actions(sim)

        action_ids = {action.action_id for action in sim.action_library.all_actions()}
        self.assertIn("action_move_to_spring_001_v1", action_ids)
        self.assertIn(
            "action_move_to_known_freshwater_spring_template_v1",
            action_ids,
        )

    def test_goap_can_move_to_known_spring_and_drink(self) -> None:
        sim = _make_sim()
        _remember_discoverable(sim, Position(12, 12))
        _synthesize_spring_exploit_actions(sim)
        _synthesize_spring_travel_actions(sim)

        before_trajectory_count = len(sim.recorded_trajectories)
        travel_action = sim.action_library.get(
            "action_move_to_known_freshwater_spring_template_v1"
        )
        self.assertIsNotNone(travel_action)
        assert travel_action is not None
        previous_trials = travel_action.confidence.trials

        sim.agent.position = Position(0, 0)
        sim.agent.thirst = 0.9
        sim.agent.hunger = 0.1
        sim.agent.fatigue = 0.1
        steps = sim.current_goap_plan({"thirst_bucket": "low"})
        self.assertGreaterEqual(len(steps), 2)
        self.assertTrue(steps[0].action.action_id.startswith("action_move_to"))
        self.assertTrue(steps[1].action.action_id.startswith("action_exploit"))

        results = sim.execute_goap_plan({"thirst_bucket": "low"})

        spring = sim.world.discoverables["spring_001"]
        distance = max(
            abs(sim.agent.position.x - spring.x),
            abs(sim.agent.position.y - spring.y),
        )
        self.assertTrue(all(result.success for result in results))
        self.assertLessEqual(distance, 1)
        self.assertLess(sim.agent.thirst, 0.9)
        self.assertGreater(len(sim.recorded_trajectories), before_trajectory_count)
        self.assertGreater(travel_action.confidence.trials, previous_trials)

    def test_goap_step_keeps_awake_ticks_aligned_with_elapsed_ticks(self) -> None:
        sim = _make_sim()
        _remember_discoverable(sim, Position(12, 12))
        _synthesize_spring_exploit_actions(sim)
        _synthesize_spring_travel_actions(sim)

        sim.agent.position = Position(0, 0)
        sim.agent.thirst = 0.9
        start_tick = sim.tick
        start_awake_ticks = sim.agent.awake_ticks

        sim.step()

        self.assertEqual(
            sim.agent.awake_ticks - start_awake_ticks, sim.tick - start_tick
        )

    def test_action_library_round_trips_travel_payloads(self) -> None:
        sim = _make_sim()
        _remember_discoverable(sim, Position(12, 12))
        _synthesize_spring_travel_actions(sim)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "actions.json"
            sim.action_library.save(path)
            loaded = ActionLibrary.load(path)

        action = loaded.get("action_move_to_spring_001_v1")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.execution_payload.type, ExecutorType.PATHFINDER)
        self.assertEqual(
            action.execution_payload.target_binding.resource_id,
            "spring_001",
        )
        self.assertEqual(action.symbolic_effects["at_known_target"], True)


if __name__ == "__main__":
    unittest.main()
