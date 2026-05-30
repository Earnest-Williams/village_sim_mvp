"""Tests for bounded GOAP planning: chaining, filtering, and cost ordering (§31)."""

from __future__ import annotations

import unittest

import msgpack
from village_sim.goap.planner import (
    _action_advances_goal,
    _action_applicable,
    _expected_cost,
    plan,
)
from village_sim.msgpack_codec import pack_default, unpack_object_hook
from village_sim.orchestrator.action_model import (
    ActionConfidence,
    ActionLifecycle,
    ActionScope,
    CostModel,
    ExecutionPayload,
    ExecutorType,
    SynthesizedAction,
    TargetBinding,
)
from village_sim.orchestrator.induction import EffectEstimate
from village_sim.orchestrator.symbolic import FactValue


def _make_action(
    action_id: str,
    preconditions: dict[str, FactValue],
    effects: dict[str, EffectEstimate],
    base_ticks: float = 10.0,
    success_rate: float = 0.9,
    lifecycle: ActionLifecycle = ActionLifecycle.TRUSTED,
    fatigue_weight: float = 0.0,
    symbolic_effects: dict[str, FactValue] | None = None,
    executor_type: ExecutorType = ExecutorType.SCRIPTED_PRIMITIVE,
) -> SynthesizedAction:
    trials = 100
    successful_trials = round(success_rate * trials)
    return SynthesizedAction(
        schema_version=1,
        action_id=action_id,
        display_name=action_id,
        scope=ActionScope.TEMPLATE,
        lifecycle=lifecycle,
        preconditions=preconditions,
        soft_preconditions={},
        effects=effects,
        side_effects={},
        cost_model=CostModel(
            base_ticks=base_ticks,
            fatigue_weight=fatigue_weight,
        ),
        confidence=ActionConfidence(
            trials=trials,
            successful_trials=successful_trials,
            failed_trials=trials - successful_trials,
            success_rate=success_rate,
            death_rate=0.0,
            timeout_rate=0.0,
        ),
        execution_payload=ExecutionPayload(
            type=executor_type,
            policy_id="primitive_drink",
            policy_version=1,
            target_binding=TargetBinding(mode="resource_id", resource_id="spring_001"),
        ),
        symbolic_effects=symbolic_effects or {},
    )


def _thirst_effect() -> dict[str, EffectEstimate]:
    return {
        "thirst_delta": EffectEstimate(
            mean=-0.65,
            p10=-0.75,
            p90=-0.55,
            confidence=0.9,
        )
    }


class TestActionApplicable(unittest.TestCase):
    def test_applicable_when_preconditions_met(self) -> None:
        action = _make_action(
            "drink",
            preconditions={"at_discoverable": True, "target_type": "freshwater_spring"},
            effects=_thirst_effect(),
        )
        state: dict[str, FactValue] = {
            "at_discoverable": True,
            "target_type": "freshwater_spring",
            "thirst_bucket": "high",
        }
        self.assertTrue(_action_applicable(action, state))

    def test_not_applicable_when_precondition_missing(self) -> None:
        action = _make_action(
            "drink",
            preconditions={"at_discoverable": True},
            effects=_thirst_effect(),
        )
        state: dict[str, FactValue] = {"at_discoverable": False}
        self.assertFalse(_action_applicable(action, state))


class TestExpectedCost(unittest.TestCase):
    def test_lower_cost_for_high_success_rate(self) -> None:
        high_rate = _make_action(
            "a",
            preconditions={},
            effects={},
            base_ticks=10.0,
            success_rate=0.95,
        )
        low_rate = _make_action(
            "b",
            preconditions={},
            effects={},
            base_ticks=10.0,
            success_rate=0.5,
        )
        self.assertLess(_expected_cost(high_rate), _expected_cost(low_rate))

    def test_fatigue_weight_increases_cost(self) -> None:
        no_fatigue = _make_action("a", preconditions={}, effects={}, fatigue_weight=0.0)
        with_fatigue = _make_action(
            "b",
            preconditions={},
            effects={},
            fatigue_weight=1.0,
        )
        self.assertLess(_expected_cost(no_fatigue), _expected_cost(with_fatigue))


class TestActionAdvancesGoal(unittest.TestCase):
    def test_advances_thirst_goal(self) -> None:
        action = _make_action("drink", preconditions={}, effects=_thirst_effect())
        goal = {"thirst_bucket": "low"}
        state: dict[str, FactValue] = {}
        self.assertTrue(_action_advances_goal(action, state, goal))

    def test_symbolic_effect_advances_intermediate_state(self) -> None:
        action = _make_action(
            "move",
            preconditions={"has_target_location": True},
            effects={},
            symbolic_effects={"at_known_target": True},
        )
        state: dict[str, FactValue] = {
            "has_target_location": True,
            "at_known_target": False,
        }
        goal = {"thirst_bucket": "low"}
        self.assertTrue(_action_advances_goal(action, state, goal))

    def test_does_not_advance_irrelevant_goal(self) -> None:
        action = _make_action("drink", preconditions={}, effects=_thirst_effect())
        goal = {"hunger_bucket": "low"}
        state: dict[str, FactValue] = {}
        self.assertFalse(_action_advances_goal(action, state, goal))

    def test_skips_already_satisfied_goal(self) -> None:
        action = _make_action("drink", preconditions={}, effects=_thirst_effect())
        goal = {"thirst_bucket": "low"}
        state: dict[str, FactValue] = {"thirst_bucket": "low"}
        self.assertFalse(_action_advances_goal(action, state, goal))


class TestMsgpackExecutionPayloadRoundTrip(unittest.TestCase):
    def test_preserves_executor_type_enum(self) -> None:
        payload = ExecutionPayload(
            type=ExecutorType.PATHFINDER,
            policy_id="pathfinder_v1",
            policy_version=1,
            target_binding=TargetBinding(mode="resource_id", resource_id="spring_001"),
        )
        raw = msgpack.packb(payload, default=pack_default, use_bin_type=True)
        restored = msgpack.unpackb(raw, raw=False, object_hook=unpack_object_hook)
        self.assertIsInstance(restored, ExecutionPayload)
        self.assertIs(restored.type, ExecutorType.PATHFINDER)
        self.assertEqual(restored.type.value, "PATHFINDER")


class TestPlan(unittest.TestCase):
    def test_plan_returns_lowest_cost_one_step_route(self) -> None:
        cheap = _make_action(
            "drink_cheap",
            preconditions={"at_discoverable": True},
            effects=_thirst_effect(),
            base_ticks=5.0,
            success_rate=0.95,
        )
        expensive = _make_action(
            "drink_expensive",
            preconditions={"at_discoverable": True},
            effects=_thirst_effect(),
            base_ticks=20.0,
            success_rate=0.95,
        )
        state = {"at_discoverable": True}
        goal = {"thirst_bucket": "low"}

        steps = plan(state, goal, [expensive, cheap])
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].action.action_id, "drink_cheap")

    def test_plan_chains_move_then_exploit(self) -> None:
        move = _make_action(
            "move",
            preconditions={
                "has_target_location": True,
                "at_known_target": False,
                "target_type": "freshwater_spring",
            },
            effects={},
            base_ticks=10.0,
            symbolic_effects={
                "at_known_target": True,
                "at_discoverable": True,
                "visible_discoverable": True,
            },
            executor_type=ExecutorType.PATHFINDER,
        )
        drink = _make_action(
            "drink",
            preconditions={
                "at_known_target": True,
                "at_discoverable": True,
                "target_type": "freshwater_spring",
            },
            effects=_thirst_effect(),
            base_ticks=3.0,
        )
        state: dict[str, FactValue] = {
            "has_target_location": True,
            "at_known_target": False,
            "at_discoverable": False,
            "target_type": "freshwater_spring",
            "thirst_bucket": "critical",
        }
        steps = plan(state, {"thirst_bucket": "low"}, [drink, move])
        self.assertEqual([step.action.action_id for step in steps], ["move", "drink"])

    def test_already_at_target_returns_only_exploit(self) -> None:
        move = _make_action(
            "move",
            preconditions={"at_known_target": False},
            effects={},
            symbolic_effects={"at_known_target": True},
        )
        drink = _make_action(
            "drink",
            preconditions={"at_known_target": True, "at_discoverable": True},
            effects=_thirst_effect(),
        )
        state: dict[str, FactValue] = {
            "at_known_target": True,
            "at_discoverable": True,
            "thirst_bucket": "critical",
        }
        steps = plan(state, {"thirst_bucket": "low"}, [move, drink])
        self.assertEqual([step.action.action_id for step in steps], ["drink"])

    def test_no_known_target_location_returns_empty(self) -> None:
        move = _make_action(
            "move",
            preconditions={"has_target_location": True},
            effects={},
            symbolic_effects={"at_known_target": True},
        )
        drink = _make_action(
            "drink",
            preconditions={"at_known_target": True},
            effects=_thirst_effect(),
        )
        state: dict[str, FactValue] = {"has_target_location": False}
        steps = plan(state, {"thirst_bucket": "low"}, [move, drink])
        self.assertEqual(steps, [])

    def test_plan_excludes_deprecated_by_default(self) -> None:
        deprecated = _make_action(
            "old_drink",
            preconditions={"at_discoverable": True},
            effects=_thirst_effect(),
            lifecycle=ActionLifecycle.DEPRECATED,
        )
        state = {"at_discoverable": True}
        goal = {"thirst_bucket": "low"}

        steps = plan(state, goal, [deprecated])
        self.assertEqual(steps, [])

    def test_plan_can_include_deprecated_when_floor_allows_it(self) -> None:
        deprecated = _make_action(
            "old_drink",
            preconditions={"at_discoverable": True},
            effects=_thirst_effect(),
            lifecycle=ActionLifecycle.DEPRECATED,
        )
        state = {"at_discoverable": True}
        goal = {"thirst_bucket": "low"}

        steps = plan(
            state,
            goal,
            [deprecated],
            agent_lifecycle_floor=ActionLifecycle.DEPRECATED,
        )
        self.assertEqual([step.action.action_id for step in steps], ["old_drink"])

    def test_plan_excludes_when_preconditions_unmet(self) -> None:
        action = _make_action(
            "drink",
            preconditions={"at_discoverable": True},
            effects=_thirst_effect(),
        )
        state: dict[str, FactValue] = {}
        goal = {"thirst_bucket": "low"}

        steps = plan(state, goal, [action])
        self.assertEqual(steps, [])


if __name__ == "__main__":
    unittest.main()
