"""Tests for the GOAP planner: action selection, filtering, cost ordering (§31)."""

from __future__ import annotations

import unittest

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
from village_sim.goap.planner import (
    _action_advances_goal,
    _action_applicable,
    _expected_cost,
    plan,
)


def _make_action(
    action_id: str,
    preconditions: dict[str, FactValue],
    effects: dict[str, EffectEstimate],
    base_ticks: float = 10.0,
    success_rate: float = 0.9,
    lifecycle: ActionLifecycle = ActionLifecycle.TRUSTED,
    fatigue_weight: float = 0.0,
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
            type=ExecutorType.SCRIPTED_PRIMITIVE,
            policy_id="primitive_drink",
            policy_version=1,
            target_binding=TargetBinding(mode="resource_id", resource_id="spring_001"),
        ),
    )


class TestActionApplicable(unittest.TestCase):
    def test_applicable_when_preconditions_met(self) -> None:
        action = _make_action(
            "drink",
            preconditions={"at_discoverable": True, "target_type": "freshwater_spring"},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
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
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
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
            "b", preconditions={}, effects={}, fatigue_weight=1.0
        )
        self.assertLess(_expected_cost(no_fatigue), _expected_cost(with_fatigue))


class TestActionAdvancesGoal(unittest.TestCase):
    def test_advances_thirst_goal(self) -> None:
        action = _make_action(
            "drink",
            preconditions={},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
        )
        goal = {"thirst_bucket": "low"}
        state: dict[str, FactValue] = {}
        self.assertTrue(_action_advances_goal(action, state, goal))

    def test_does_not_advance_irrelevant_goal(self) -> None:
        action = _make_action(
            "drink",
            preconditions={},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
        )
        goal = {"hunger_bucket": "low"}
        state: dict[str, FactValue] = {}
        self.assertFalse(_action_advances_goal(action, state, goal))

    def test_skips_already_satisfied_goal(self) -> None:
        """Goals already satisfied in current state must be skipped."""
        action = _make_action(
            "drink",
            preconditions={},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
        )
        goal = {"thirst_bucket": "low"}
        # State already satisfies the goal — action should not be considered.
        state: dict[str, FactValue] = {"thirst_bucket": "low"}
        self.assertFalse(_action_advances_goal(action, state, goal))


class TestPlan(unittest.TestCase):
    def test_plan_returns_applicable_actions_sorted_by_cost(self) -> None:
        cheap = _make_action(
            "drink_cheap",
            preconditions={"at_discoverable": True},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
            base_ticks=5.0,
            success_rate=0.95,
        )
        expensive = _make_action(
            "drink_expensive",
            preconditions={"at_discoverable": True},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
            base_ticks=20.0,
            success_rate=0.95,
        )
        state = {"at_discoverable": True}
        goal = {"thirst_bucket": "low"}

        steps = plan(state, goal, [expensive, cheap])
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].action.action_id, "drink_cheap")

    def test_plan_excludes_deprecated_by_default(self) -> None:
        deprecated = _make_action(
            "old_drink",
            preconditions={"at_discoverable": True},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
            lifecycle=ActionLifecycle.DEPRECATED,
        )
        state = {"at_discoverable": True}
        goal = {"thirst_bucket": "low"}

        steps = plan(state, goal, [deprecated])
        self.assertEqual(steps, [])

    def test_plan_excludes_when_preconditions_unmet(self) -> None:
        action = _make_action(
            "drink",
            preconditions={"at_discoverable": True},
            effects={
                "thirst_delta": EffectEstimate(
                    mean=-0.65, p10=-0.75, p90=-0.55, confidence=0.9
                )
            },
        )
        state: dict[str, FactValue] = {}  # at_discoverable not set
        goal = {"thirst_bucket": "low"}

        steps = plan(state, goal, [action])
        self.assertEqual(steps, [])


if __name__ == "__main__":
    unittest.main()
