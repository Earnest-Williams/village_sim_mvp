"""Simple GOAP planner over synthesized actions (§18)."""

from __future__ import annotations

from dataclasses import dataclass

from village_sim.orchestrator.action_model import (
    ActionLifecycle,
    SynthesizedAction,
)
from village_sim.orchestrator.symbolic import FactValue, SymbolicState


@dataclass
class PlanStep:
    action: SynthesizedAction
    expected_cost: float


def _action_applicable(
    action: SynthesizedAction,
    state: SymbolicState,
) -> bool:
    """Check hard preconditions against the current symbolic state."""
    for fact, required_value in action.preconditions.items():
        if state.get(fact) != required_value:
            return False
    return True


def _expected_cost(action: SynthesizedAction, failure_penalty: float = 50.0) -> float:
    """Simplified scoring formula from §18.

    expected_cost =
        base_ticks
        + failure_penalty * (1.0 - success_rate)
        + fatigue_weight * base_ticks * 0.1   (side-effect penalty proxy)
    """
    base = action.cost_model.base_ticks
    failure_term = failure_penalty * (1.0 - action.confidence.success_rate)
    side_effect_penalty = action.cost_model.fatigue_weight * base * 0.10
    return base + failure_term + side_effect_penalty


def _action_advances_goal(
    action: SynthesizedAction,
    goal: SymbolicState,
) -> bool:
    """True if at least one effect moves toward the goal."""
    for goal_fact, goal_value in goal.items():
        need_prefix = goal_fact.replace("_bucket", "")
        delta_key = f"{need_prefix}_delta"
        if delta_key in action.effects:
            estimate = action.effects[delta_key]
            if goal_value in ("low", "medium") and estimate.mean < 0.0:
                return True
    return False


def plan(
    state: SymbolicState,
    goal: SymbolicState,
    library: list[SynthesizedAction],
    agent_lifecycle_floor: ActionLifecycle = ActionLifecycle.CANDIDATE,
) -> list[PlanStep]:
    """Return a prioritised (lowest cost first) list of applicable plan steps.

    For the MVP this is a flat best-action selector, not a full tree search.
    A full A* GOAP tree search can replace this without changing the interface.
    """
    lifecycle_order = [
        ActionLifecycle.TRUSTED,
        ActionLifecycle.VALIDATED,
        ActionLifecycle.CANDIDATE,
        ActionLifecycle.DEPRECATED,
    ]
    min_rank = lifecycle_order.index(agent_lifecycle_floor)

    candidates: list[PlanStep] = []
    for action in library:
        action_rank = lifecycle_order.index(action.lifecycle)
        if action_rank > min_rank:
            continue
        if not _action_applicable(action, state):
            continue
        if not _action_advances_goal(action, goal):
            continue
        candidates.append(PlanStep(action=action, expected_cost=_expected_cost(action)))

    candidates.sort(key=lambda ps: ps.expected_cost)
    return candidates
