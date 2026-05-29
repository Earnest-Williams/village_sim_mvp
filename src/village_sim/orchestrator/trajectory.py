"""Trajectory recording data model (§9).

A trajectory records state before and after each primitive action.
The TrajectoryRecorder helper accumulates steps into a Trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from village_sim.core.types import PrimitiveAction
from village_sim.orchestrator.symbolic import FactValue, SymbolicState


@dataclass(slots=True)
class NeedState:
    hunger: float
    thirst: float
    fatigue: float
    health: float
    cold_stress: float = 0.0


@dataclass(slots=True)
class StateSnapshot:
    tick: int
    agent_id: int
    x: int
    y: int
    needs: NeedState
    symbolic: SymbolicState


@dataclass(slots=True)
class TrajectoryStep:
    before: StateSnapshot
    action: PrimitiveAction
    after: StateSnapshot
    reward: float
    events: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Trajectory:
    trajectory_id: str
    policy_id: str
    task_name: str
    steps: list[TrajectoryStep]

    @property
    def start(self) -> StateSnapshot:
        return self.steps[0].before

    @property
    def end(self) -> StateSnapshot:
        return self.steps[-1].after

    @property
    def cost_ticks(self) -> int:
        return self.end.tick - self.start.tick


# ── Recorder helper ──────────────────────────────────────────────────────────


class TrajectoryRecorder:
    """Accumulates steps into a Trajectory.

    Usage inside the simulation loop::

        recorder = TrajectoryRecorder(
            trajectory_id="...", policy_id="...", task_name="..."
        )
        recorder.record(before_snapshot, action, after_snapshot, reward, events)
        trajectory = recorder.finish()
    """

    def __init__(self, trajectory_id: str, policy_id: str, task_name: str) -> None:
        self.trajectory_id = trajectory_id
        self.policy_id = policy_id
        self.task_name = task_name
        self._steps: list[TrajectoryStep] = []

    def record(
        self,
        before: StateSnapshot,
        action: PrimitiveAction,
        after: StateSnapshot,
        reward: float,
        events: list[str] | None = None,
    ) -> None:
        self._steps.append(
            TrajectoryStep(
                before=before,
                action=action,
                after=after,
                reward=reward,
                events=events or [],
            )
        )

    def finish(self) -> Trajectory:
        if not self._steps:
            raise ValueError("Cannot finish a trajectory with no steps.")
        return Trajectory(
            trajectory_id=self.trajectory_id,
            policy_id=self.policy_id,
            task_name=self.task_name,
            steps=list(self._steps),
        )
