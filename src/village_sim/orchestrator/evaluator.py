"""Task evaluators and trajectory clustering (§10, §11)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from village_sim.orchestrator.symbolic import SymbolicState
from village_sim.orchestrator.trajectory import Trajectory


class NeedName(StrEnum):
    HUNGER = "hunger"
    THIRST = "thirst"
    FATIGUE = "fatigue"
    COLD_STRESS = "cold_stress"


@dataclass(slots=True)
class TaskResult:
    success: bool
    death: bool
    timeout: bool
    main_need_delta: float


# ── Need task evaluators (§10) ───────────────────────────────────────────────


def _evaluate_need_task(
    trajectory: Trajectory,
    need: NeedName,
    improvement_threshold: float = -0.20,
    max_ticks: int = 500,
) -> TaskResult:
    start_value: float = getattr(trajectory.start.needs, need.value)
    end_value: float = getattr(trajectory.end.needs, need.value)
    delta: float = end_value - start_value

    death: bool = trajectory.end.needs.health <= 0.0
    timeout: bool = trajectory.cost_ticks > max_ticks
    success: bool = delta <= improvement_threshold and not death and not timeout

    return TaskResult(
        success=success,
        death=death,
        timeout=timeout,
        main_need_delta=delta,
    )


def evaluate_hunger_task(trajectory: Trajectory) -> TaskResult:
    return _evaluate_need_task(trajectory, NeedName.HUNGER)


def evaluate_thirst_task(trajectory: Trajectory) -> TaskResult:
    return _evaluate_need_task(trajectory, NeedName.THIRST)


def evaluate_fatigue_task(trajectory: Trajectory) -> TaskResult:
    return _evaluate_need_task(trajectory, NeedName.FATIGUE)


def evaluate_cold_stress_task(trajectory: Trajectory) -> TaskResult:
    return _evaluate_need_task(trajectory, NeedName.COLD_STRESS)


# ── Trajectory clustering (§11) ──────────────────────────────────────────────


def cluster_key_for_trajectory(trajectory: Trajectory) -> str:
    """Produce a stable cluster key for grouping similar successful trajectories.

    Format: ``<task_name>:<target_type>:<comma-joined changed needs>``
    """
    start: SymbolicState = trajectory.start.symbolic
    target_type: str = str(start.get("target_type", "none"))
    task_name: str = trajectory.task_name

    changed_needs: list[str] = []
    for need in NeedName:
        before: float = getattr(trajectory.start.needs, need.value)
        after: float = getattr(trajectory.end.needs, need.value)
        if abs(after - before) >= 0.10:
            changed_needs.append(need.value)

    return f"{task_name}:{target_type}:{','.join(changed_needs)}"
