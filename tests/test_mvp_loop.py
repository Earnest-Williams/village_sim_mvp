"""End-to-end MVP integration test: Pioneer → Orchestrator → Townsfolk plan (§31)."""

from __future__ import annotations

import unittest

from village_sim.core.types import PrimitiveAction
from village_sim.goap.planner import plan
from village_sim.orchestrator.action_model import ActionLifecycle
from village_sim.orchestrator.orchestrator import Orchestrator
from village_sim.orchestrator.symbolic import FactValue
from village_sim.orchestrator.trajectory import (
    NeedState,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
)


def _make_spring_trajectory(i: int) -> Trajectory:
    """Create one successful thirst-reduction trajectory at the spring."""
    symbolic: dict[str, FactValue] = {
        "at_discoverable": True,
        "target_type": "freshwater_spring",
        "target_has_resource": True,
        "target_id": "spring_001",
    }
    before = StateSnapshot(
        tick=0,
        agent_id=1,
        x=12,
        y=12,
        needs=NeedState(hunger=0.1, thirst=0.8, fatigue=0.1, health=1.0),
        symbolic=symbolic,
    )
    after = StateSnapshot(
        tick=10,
        agent_id=1,
        x=12,
        y=12,
        needs=NeedState(hunger=0.1, thirst=0.15, fatigue=0.1, health=1.0),
        symbolic=symbolic,
    )
    return Trajectory(
        trajectory_id=f"traj_{i:03d}",
        policy_id="policy_spring_v1",
        task_name="thirst",
        steps=[
            TrajectoryStep(
                before=before,
                action=PrimitiveAction.DRINK,
                after=after,
                reward=1.0,
            )
        ],
    )


class TestMVPLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = Orchestrator()
        for i in range(15):
            self.orchestrator.record(_make_spring_trajectory(i))

    def test_orchestrator_synthesizes_action(self) -> None:
        actions = self.orchestrator.synthesize_all()
        self.assertEqual(len(actions), 2)

    def test_orchestrator_synthesizes_instance_and_template_actions(self) -> None:
        actions = self.orchestrator.synthesize_all()
        action_ids = {action.action_id for action in actions}
        self.assertIn("action_exploit_freshwater_spring_001_v1", action_ids)
        self.assertIn("action_exploit_freshwater_spring_template_v1", action_ids)

    def test_synthesized_action_has_thirst_effect(self) -> None:
        actions = self.orchestrator.synthesize_all()
        self.assertEqual(len(actions), 2)
        action = actions[0]
        self.assertIn("thirst_delta", action.effects)
        self.assertLess(action.effects["thirst_delta"].mean, 0.0)

    def test_synthesized_action_has_at_discoverable_precondition(self) -> None:
        actions = self.orchestrator.synthesize_all()
        self.assertEqual(len(actions), 2)
        action = actions[0]
        self.assertIn("at_discoverable", action.preconditions)
        self.assertEqual(action.preconditions["at_discoverable"], True)

    def test_synthesized_action_lifecycle_is_candidate(self) -> None:
        actions = self.orchestrator.synthesize_all()
        self.assertEqual(len(actions), 2)
        # 15 trials < 100 required for promotion; stays CANDIDATE
        self.assertTrue(
            all(action.lifecycle is ActionLifecycle.CANDIDATE for action in actions)
        )

    def test_townsfolk_plan_is_not_empty(self) -> None:
        actions = self.orchestrator.synthesize_all()

        townsfolk_state: dict[str, FactValue] = {
            "at_discoverable": True,
            "target_type": "freshwater_spring",
            "target_has_resource": True,
            "target_id": "spring_001",
            "thirst_bucket": "high",
        }
        townsfolk_goal: dict[str, FactValue] = {"thirst_bucket": "low"}

        steps = plan(
            townsfolk_state,
            townsfolk_goal,
            actions,
            agent_lifecycle_floor=ActionLifecycle.CANDIDATE,
        )
        self.assertGreater(len(steps), 0)

    def test_townsfolk_plan_picks_thirst_reducing_action(self) -> None:
        actions = self.orchestrator.synthesize_all()

        townsfolk_state: dict[str, FactValue] = {
            "at_discoverable": True,
            "target_type": "freshwater_spring",
            "target_has_resource": True,
            "target_id": "spring_001",
            "thirst_bucket": "high",
        }
        townsfolk_goal: dict[str, FactValue] = {"thirst_bucket": "low"}

        steps = plan(
            townsfolk_state,
            townsfolk_goal,
            actions,
            agent_lifecycle_floor=ActionLifecycle.CANDIDATE,
        )
        self.assertGreater(len(steps), 0)
        top_action = steps[0].action
        self.assertIn("thirst_delta", top_action.effects)
        self.assertLess(top_action.effects["thirst_delta"].mean, 0.0)

    def test_plan_empty_when_state_does_not_match_preconditions(self) -> None:
        actions = self.orchestrator.synthesize_all()

        townsfolk_state: dict[str, FactValue] = {
            "at_discoverable": False,  # agent is not at the spring
            "thirst_bucket": "high",
        }
        townsfolk_goal: dict[str, FactValue] = {"thirst_bucket": "low"}

        steps = plan(
            townsfolk_state,
            townsfolk_goal,
            actions,
            agent_lifecycle_floor=ActionLifecycle.CANDIDATE,
        )
        self.assertEqual(steps, [])


if __name__ == "__main__":
    unittest.main()
