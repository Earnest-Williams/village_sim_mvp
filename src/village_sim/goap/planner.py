"""Bounded GOAP planner over synthesized actions (§18)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from village_sim.orchestrator.action_model import (
    ActionLifecycle,
    SynthesizedAction,
)
from village_sim.orchestrator.symbolic import FactValue, SymbolicState


@dataclass(slots=True)
class PlanStep:
    action: SynthesizedAction
    expected_cost: float


@dataclass(slots=True)
class _SearchNode:
    state: SymbolicState
    steps: list[PlanStep]
    total_cost: float


def _action_applicable(
    action: SynthesizedAction,
    state: Mapping[str, FactValue],
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
    state: Mapping[str, FactValue],
    goal: Mapping[str, FactValue],
) -> bool:
    """True if at least one effect changes state toward an unsatisfied goal."""
    after = _apply_action_effects(state, action, goal)
    if after == dict(state):
        return False
    before_score = _goal_match_count(state, goal)
    after_score = _goal_match_count(after, goal)
    if after_score > before_score:
        return True
    for fact, value in action.symbolic_effects.items():
        if state.get(fact) != value:
            return True
    return False


def plan(
    state: Mapping[str, FactValue],
    goal: Mapping[str, FactValue],
    library: list[SynthesizedAction],
    agent_lifecycle_floor: ActionLifecycle = ActionLifecycle.CANDIDATE,
    max_depth: int = 4,
    beam_width: int = 64,
) -> list[PlanStep]:
    """Return the lowest-cost bounded forward-search plan for a goal."""
    if _goal_satisfied(state, goal):
        return []

    actions = _eligible_actions(library, agent_lifecycle_floor)
    frontier: list[_SearchNode] = [
        _SearchNode(state=dict(state), steps=[], total_cost=0.0)
    ]
    visited: set[tuple[tuple[str, FactValue], ...]] = {_state_signature(state)}
    best_plan: list[PlanStep] = []
    best_cost: float = float("inf")

    for _depth in range(max_depth):
        next_frontier: list[_SearchNode] = []
        for node in frontier:
            for action in actions:
                if not _action_applicable(action, node.state):
                    continue
                if not _action_advances_goal(action, node.state, goal):
                    continue
                if any(
                    step.action.action_id == action.action_id for step in node.steps
                ):
                    continue
                after_state = _apply_action_effects(node.state, action, goal)
                if after_state == node.state:
                    continue
                signature = _state_signature(after_state)
                if signature in visited:
                    continue
                visited.add(signature)
                step_cost = _expected_cost(action)
                steps = [*node.steps, PlanStep(action=action, expected_cost=step_cost)]
                total_cost = node.total_cost + step_cost
                if _goal_satisfied(after_state, goal):
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_plan = steps
                    continue
                if total_cost < best_cost:
                    next_frontier.append(
                        _SearchNode(
                            state=after_state,
                            steps=steps,
                            total_cost=total_cost,
                        )
                    )
        next_frontier.sort(key=lambda item: item.total_cost)
        frontier = next_frontier[:beam_width]
        if not frontier:
            break

    return best_plan


def _eligible_actions(
    library: list[SynthesizedAction],
    agent_lifecycle_floor: ActionLifecycle,
) -> list[SynthesizedAction]:
    lifecycle_order = [
        ActionLifecycle.TRUSTED,
        ActionLifecycle.VALIDATED,
        ActionLifecycle.CANDIDATE,
        ActionLifecycle.DEPRECATED,
    ]
    min_rank = lifecycle_order.index(agent_lifecycle_floor)
    actions: list[SynthesizedAction] = []
    for action in library:
        action_rank = lifecycle_order.index(action.lifecycle)
        if action_rank <= min_rank:
            actions.append(action)
    actions.sort(key=lambda item: (_expected_cost(item), item.action_id))
    return actions


def _apply_action_effects(
    state: Mapping[str, FactValue],
    action: SynthesizedAction,
    goal: Mapping[str, FactValue],
) -> SymbolicState:
    next_state: SymbolicState = dict(state)
    for fact, value in action.symbolic_effects.items():
        next_state[fact] = value

    for goal_fact, goal_value in goal.items():
        need_prefix = goal_fact.replace("_bucket", "")
        delta_key = f"{need_prefix}_delta"
        estimate = action.effects.get(delta_key)
        if estimate is not None and goal_value in ("low", "medium"):
            if estimate.mean < 0.0:
                next_state[goal_fact] = goal_value
    return next_state


def _goal_satisfied(
    state: Mapping[str, FactValue],
    goal: Mapping[str, FactValue],
) -> bool:
    for fact, value in goal.items():
        if state.get(fact) != value:
            return False
    return True


def _goal_match_count(
    state: Mapping[str, FactValue],
    goal: Mapping[str, FactValue],
) -> int:
    matches = 0
    for fact, value in goal.items():
        if state.get(fact) == value:
            matches += 1
    return matches


def _state_signature(
    state: Mapping[str, FactValue],
) -> tuple[tuple[str, FactValue], ...]:
    return tuple(sorted(state.items()))
