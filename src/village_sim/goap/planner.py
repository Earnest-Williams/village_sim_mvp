"""Bounded GOAP planner over synthesized actions (§18)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
import polars as pl
from numpy.typing import NDArray

from village_sim.orchestrator.action_model import (
    ActionLifecycle,
    SynthesizedAction,
)
from village_sim.orchestrator.symbolic import FactValue, SymbolicState

AGENT_ID = "agent_id"
_NO_PARENT_NODE = -1
_NO_ACTION_INDEX = -1


@dataclass(slots=True)
class PlanStep:
    action: SynthesizedAction
    expected_cost: float


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
    node_states: list[SymbolicState] = [dict(state)]
    node_parent_indices: list[int] = [_NO_PARENT_NODE]
    node_action_indices: list[int] = [_NO_ACTION_INDEX]
    node_step_costs: list[float] = [0.0]
    node_total_costs: list[float] = [0.0]
    frontier_node_indices: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
    visited: set[tuple[tuple[str, FactValue], ...]] = {_state_signature(state)}
    best_plan_node_index: int = _NO_PARENT_NODE
    best_cost: float = float("inf")
    max_candidates: int = max(1, beam_width * max(1, len(actions)))
    candidate_node_indices: NDArray[np.int64] = np.empty(max_candidates, dtype=np.int64)
    candidate_costs: NDArray[np.float64] = np.empty(max_candidates, dtype=np.float64)
    candidate_action_ids: NDArray[np.int64] = np.empty(max_candidates, dtype=np.int64)

    for _depth in range(max_depth):
        candidate_count: int = 0
        for frontier_index in range(frontier_node_indices.size):
            node_index: int = int(frontier_node_indices[frontier_index])
            node_state: SymbolicState = node_states[node_index]
            node_cost: float = node_total_costs[node_index]
            for action_index, action in enumerate(actions):
                if _node_has_action(
                    node_index, action_index, node_parent_indices, node_action_indices
                ):
                    continue
                if not _action_applicable(action, node_state):
                    continue
                if not _action_advances_goal(action, node_state, goal):
                    continue
                after_state: SymbolicState = _apply_action_effects(
                    node_state, action, goal
                )
                if after_state == node_state:
                    continue
                signature: tuple[tuple[str, FactValue], ...] = _state_signature(
                    after_state
                )
                if signature in visited:
                    continue
                visited.add(signature)
                step_cost: float = _expected_cost(action)
                total_cost: float = node_cost + step_cost
                new_node_index: int = len(node_states)
                node_states.append(after_state)
                node_parent_indices.append(node_index)
                node_action_indices.append(action_index)
                node_step_costs.append(step_cost)
                node_total_costs.append(total_cost)
                if _goal_satisfied(after_state, goal):
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_plan_node_index = new_node_index
                    continue
                if total_cost >= best_cost:
                    continue
                if candidate_count >= max_candidates:
                    continue
                candidate_node_indices[candidate_count] = new_node_index
                candidate_costs[candidate_count] = total_cost
                candidate_action_ids[candidate_count] = action_index
                candidate_count += 1

        if candidate_count == 0:
            break
        ordered_indices: NDArray[np.int64] = np.lexsort(
            (
                candidate_action_ids[:candidate_count],
                candidate_costs[:candidate_count],
            )
        ).astype(np.int64, copy=False)
        keep_count: int = min(beam_width, candidate_count)
        frontier_node_indices = np.empty(keep_count, dtype=np.int64)
        for new_index in range(keep_count):
            old_index: int = int(ordered_indices[new_index])
            frontier_node_indices[new_index] = candidate_node_indices[old_index]

    if best_plan_node_index == _NO_PARENT_NODE:
        return []
    return _plan_steps_from_node(
        best_plan_node_index,
        node_parent_indices,
        node_action_indices,
        node_step_costs,
        actions,
    )


def _node_has_action(
    node_index: int,
    action_index: int,
    parent_indices: list[int],
    action_indices: list[int],
) -> bool:
    current_index: int = node_index
    while current_index != _NO_PARENT_NODE:
        if action_indices[current_index] == action_index:
            return True
        current_index = parent_indices[current_index]
    return False


def _plan_steps_from_node(
    node_index: int,
    parent_indices: list[int],
    action_indices: list[int],
    step_costs: list[float],
    actions: list[SynthesizedAction],
) -> list[PlanStep]:
    reverse_steps: list[PlanStep] = []
    current_index: int = node_index
    while current_index != _NO_PARENT_NODE:
        action_index: int = action_indices[current_index]
        if action_index != _NO_ACTION_INDEX:
            reverse_steps.append(
                PlanStep(
                    action=actions[action_index],
                    expected_cost=step_costs[current_index],
                )
            )
        current_index = parent_indices[current_index]
    reverse_steps.reverse()
    return reverse_steps


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
    state: SymbolicState = {}
    for key in row.columns:
        if key == AGENT_ID:
            continue
        value: object = row.get_column(key).item()
        if isinstance(value, bool | int | float | str):
            fact_value: FactValue = value
            state[key] = fact_value
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
