"""Bounded GOAP planner over synthesized actions (§18)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from village_sim.orchestrator.action_model import (
    ActionLifecycle,
    SynthesizedAction,
)
from village_sim.orchestrator.symbolic import FactValue, SymbolicState

AGENT_ID = "agent_id"


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

    if not action.preconditions:
        return True
    frame: pl.DataFrame = symbolic_state_frame(state)
    return filter_agents_for_action(frame, action).height > 0


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
    """True if at least one effect moves toward an unsatisfied goal."""
    for goal_fact, goal_value in goal.items():
        if state.get(goal_fact) == goal_value:
            continue  # already satisfied – skip
        need_prefix = goal_fact.replace("_bucket", "")
        delta_key = f"{need_prefix}_delta"
        estimate = action.effects.get(delta_key)
        if estimate is not None and goal_value in ("low", "medium"):
            if estimate.mean < 0.0:
                return True
    # Allow symbolic-state changes that enable subsequent steps in a multi-step plan.
    for fact, value in action.symbolic_effects.items():
        if state.get(fact) != value:
            return True
    return False


def symbolic_state_frame(
    state: Mapping[str, FactValue], agent_id: int = 0
) -> pl.DataFrame:
    """Represent one symbolic agent state as a one-row Polars DataFrame."""

    row: dict[str, FactValue] = dict(state)
    row[AGENT_ID] = agent_id
    return pl.DataFrame([row], orient="row")


def filter_agents_for_action(
    frame: pl.DataFrame,
    action: SynthesizedAction,
) -> pl.DataFrame:
    """Vector-filter agents that satisfy an action's hard preconditions."""

    predicate: pl.Expr = _precondition_expr(action, frame.columns)
    return frame.filter(predicate)


def plan_batch(
    agent_states: pl.DataFrame,
    goal: Mapping[str, FactValue],
    library: list[SynthesizedAction],
    agent_lifecycle_floor: ActionLifecycle = ActionLifecycle.CANDIDATE,
    max_depth: int = 4,
    beam_width: int = 64,
) -> dict[int, list[PlanStep]]:
    """Plan only for agents passing vectorized GOAP precondition filters."""

    actions: list[SynthesizedAction] = _eligible_actions(library, agent_lifecycle_floor)
    if not actions or agent_states.is_empty():
        return {}

    candidate_mask: pl.Expr = pl.lit(False)
    for action in actions:
        candidate_mask = candidate_mask | _precondition_expr(
            action, agent_states.columns
        )
    goal_unsatisfied: pl.Expr = _goal_unsatisfied_expr(goal, agent_states.columns)
    candidates: pl.DataFrame = agent_states.filter(candidate_mask & goal_unsatisfied)
    plans: dict[int, list[PlanStep]] = {}
    if candidates.is_empty():
        return plans

    agent_ids: list[int] = [int(value) for value in candidates.get_column(AGENT_ID)]
    for agent_id in agent_ids:
        state: SymbolicState = _symbolic_state_from_frame(candidates, agent_id)
        steps: list[PlanStep] = _plan_with_actions(
            state=state,
            goal=goal,
            actions=actions,
            max_depth=max_depth,
            beam_width=beam_width,
        )
        if steps:
            plans[agent_id] = steps
    return plans


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

    actions: list[SynthesizedAction] = _eligible_actions(
        library,
        agent_lifecycle_floor,
    )
    return _plan_with_actions(
        state=dict(state),
        goal=goal,
        actions=actions,
        max_depth=max_depth,
        beam_width=beam_width,
    )


def _plan_with_actions(
    *,
    state: Mapping[str, FactValue],
    goal: Mapping[str, FactValue],
    actions: list[SynthesizedAction],
    max_depth: int,
    beam_width: int,
) -> list[PlanStep]:
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


def _precondition_expr(action: SynthesizedAction, columns: list[str]) -> pl.Expr:
    predicate: pl.Expr = pl.lit(True)
    for fact, required_value in action.preconditions.items():
        if fact not in columns:
            return pl.lit(False)
        predicate = predicate & (pl.col(fact) == required_value)
    return predicate


def _goal_unsatisfied_expr(
    goal: Mapping[str, FactValue], columns: list[str]
) -> pl.Expr:
    predicate: pl.Expr = pl.lit(False)
    for fact, value in goal.items():
        if fact not in columns:
            predicate = predicate | pl.lit(True)
        else:
            predicate = predicate | (pl.col(fact) != value)
    return predicate


def _symbolic_state_from_frame(frame: pl.DataFrame, agent_id: int) -> SymbolicState:
    row: pl.DataFrame = frame.filter(pl.col(AGENT_ID) == agent_id)
    if row.height != 1:
        raise ValueError("agent state frame must contain exactly one row for agent_id")
    values: dict[str, list[Any]] = row.to_dict(as_series=False)
    state: SymbolicState = {}
    for key, column in values.items():
        if key == AGENT_ID:
            continue
        if len(column) != 1:
            raise ValueError(f"symbolic frame missing scalar column: {key}")
        value: Any = column[0]
        if isinstance(value, bool | int | float | str):
            state[key] = value
    return state


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


def _state_signature(
    state: Mapping[str, FactValue],
) -> tuple[tuple[str, FactValue], ...]:
    return tuple(sorted(state.items()))
