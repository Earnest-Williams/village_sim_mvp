"""Tests for orchestrator task evaluators, clustering, and induction (§31)."""

from __future__ import annotations

import unittest

from village_sim.core.types import PrimitiveAction
from village_sim.orchestrator.evaluator import (
    NeedName,
    cluster_key_for_trajectory,
    evaluate_hunger_task,
    evaluate_thirst_task,
)
from village_sim.orchestrator.induction import (
    average_cost,
    fact_frequency,
    infer_hard_preconditions,
    infer_need_effect,
)
from village_sim.orchestrator.action_model import ActionScope
from village_sim.orchestrator.orchestrator import _action_suffix
from village_sim.orchestrator.trajectory import (
    NeedState,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
)


def _make_snapshot(
    tick: int,
    hunger: float,
    thirst: float,
    at_discoverable: bool = True,
    target_type: str = "freshwater_spring",
    health: float = 1.0,
) -> StateSnapshot:
    return StateSnapshot(
        tick=tick,
        agent_id=1,
        x=10,
        y=12,
        needs=NeedState(hunger=hunger, thirst=thirst, fatigue=0.1, health=health),
        symbolic={
            "at_discoverable": at_discoverable,
            "target_type": target_type,
            "target_has_resource": True,
            "target_id": f"{target_type}_001",
        },
    )


def _make_trajectory(
    hunger_start: float,
    thirst_start: float,
    hunger_end: float,
    thirst_end: float,
    ticks: int = 10,
    target_type: str = "freshwater_spring",
    task_name: str = "thirst",
    health_end: float = 1.0,
) -> Trajectory:
    before = _make_snapshot(0, hunger_start, thirst_start, target_type=target_type)
    after = _make_snapshot(
        ticks, hunger_end, thirst_end, target_type=target_type, health=health_end
    )
    return Trajectory(
        trajectory_id="traj_001",
        policy_id="policy_spring_v1",
        task_name=task_name,
        steps=[
            TrajectoryStep(
                before=before,
                action=PrimitiveAction.DRINK,
                after=after,
                reward=1.0,
            )
        ],
    )


class TestTaskEvaluators(unittest.TestCase):
    def test_thirst_success(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.15)
        result = evaluate_thirst_task(traj)
        self.assertTrue(result.success)
        self.assertFalse(result.death)

    def test_thirst_failure_small_delta(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.75)  # only -0.05 delta
        result = evaluate_thirst_task(traj)
        self.assertFalse(result.success)

    def test_thirst_failure_death(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.15, health_end=0.0)
        result = evaluate_thirst_task(traj)
        self.assertFalse(result.success)
        self.assertTrue(result.death)

    def test_hunger_success(self) -> None:
        traj = _make_trajectory(
            0.8, 0.1, 0.4, 0.1, target_type="berry_bush", task_name="hunger"
        )
        result = evaluate_hunger_task(traj)
        self.assertTrue(result.success)


class TestClustering(unittest.TestCase):
    def test_cluster_key(self) -> None:
        traj = _make_trajectory(0.1, 0.8, 0.1, 0.15)
        key = cluster_key_for_trajectory(traj)
        self.assertIn("freshwater_spring", key)
        self.assertIn("thirst", key)


class TestPreconditionInference(unittest.TestCase):
    def test_infer_hard_preconditions(self) -> None:
        successful = [_make_trajectory(0.1, 0.8, 0.1, 0.15) for _ in range(10)]
        failed: list[Trajectory] = []
        prec = infer_hard_preconditions(successful, failed)
        self.assertIn("at_discoverable", prec)
        self.assertEqual(prec["at_discoverable"], True)

    def test_fact_frequency(self) -> None:
        trajs = [_make_trajectory(0.1, 0.8, 0.1, 0.15) for _ in range(4)]
        freq = fact_frequency(trajs, "at_discoverable", True)
        self.assertEqual(freq, 1.0)


class TestEffectInference(unittest.TestCase):
    def test_infer_thirst_effect(self) -> None:
        trajs = [_make_trajectory(0.1, 0.8, 0.1, 0.15) for _ in range(5)]
        est = infer_need_effect(trajs, NeedName.THIRST)
        self.assertIsNotNone(est)
        assert est is not None
        self.assertLess(est.mean, 0.0)
        self.assertGreater(est.confidence, 0.0)

    def test_average_cost(self) -> None:
        trajs = [_make_trajectory(0.1, 0.8, 0.1, 0.15, ticks=10) for _ in range(3)]
        cost = average_cost(trajs)
        self.assertAlmostEqual(cost, 10.0)


class TestActionSuffix(unittest.TestCase):
    def test_preserves_full_target_id_when_suffix_is_not_numeric(self) -> None:
        self.assertEqual(
            _action_suffix(
                "freshwater_spring", "spring_001_active", ActionScope.INSTANCE
            ),
            "spring_001_active",
        )


if __name__ == "__main__":
    unittest.main()
