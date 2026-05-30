"""Utility-style survival policy."""

from __future__ import annotations

import random
from typing import List, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from village_sim.agent.actions import (
    execute_drink,
    execute_eat,
    execute_explore,
    execute_move_toward,
    execute_search_near,
    execute_sleep,
)
from village_sim.agent.decision import DecisionSource, DecisionTrace
from village_sim.agent.memory import AgentMemory, ResourceMemory
from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.time import SimClock
from village_sim.core.types import GoalKind, Position, ResourceKind
from village_sim.world.world import World


def choose_goal(agent: AgentState, clock: SimClock) -> GoalKind:
    """Pick a high-level goal from current needs and time of day."""
    if agent.thirst >= 0.64:
        return GoalKind.GET_WATER
    if agent.hunger >= 0.66:
        return GoalKind.GET_FOOD
    if clock.is_night and agent.thirst < 0.82 and agent.hunger < 0.82:
        return GoalKind.SLEEP
    if agent.fatigue >= 0.82 and agent.thirst < 0.86 and agent.hunger < 0.86:
        return GoalKind.SLEEP
    return GoalKind.EXPLORE


def choose_and_execute_action(
    agent: AgentState,
    memory: AgentMemory,
    visible_water_indices: NDArray[np.int64],
    visible_food_indices: NDArray[np.int64],
    world: World,
    clock: SimClock,
    rng: random.Random,
    config: SimConfig,
) -> str:
    """Resolve a goal to a concrete action and execute it over pure index arrays."""
    agent.decision_trace = DecisionTrace()
    agent.current_goal = choose_goal(agent, clock)

    if agent.current_goal is GoalKind.GET_WATER:
        return _resolve_water_goal(
            agent, memory, visible_water_indices, world, clock.tick, rng, config
        )
    if agent.current_goal is GoalKind.GET_FOOD:
        return _resolve_food_goal(
            agent, memory, visible_food_indices, world, clock.tick, rng, config
        )
    if agent.current_goal is GoalKind.SLEEP:
        return execute_sleep(agent)
    if agent.current_goal is GoalKind.EXPLORE:
        agent.decision_trace = DecisionTrace(
            source=DecisionSource.EXPLORE,
            reason="general exploration",
        )
        return execute_explore(agent, world, rng)
    return "idled"


def _resolve_water_goal(
    agent: AgentState,
    memory: AgentMemory,
    visible_water_indices: NDArray[np.int64],
    world: World,
    tick: int,
    rng: random.Random,
    config: SimConfig,
) -> str:
    drink_position: Position | None = world.nearest_drinkable_position(agent.position)
    if drink_position is not None:
        _set_position_trace(
            agent,
            DecisionSource.CURRENT_TILE_RESOURCE,
            ResourceKind.WATER,
            drink_position,
            "current or adjacent water is drinkable",
        )
        return execute_drink(agent, memory, world, tick, config)

    closest_idx: int = _closest_visible_idx(
        agent.position.x, agent.position.y, visible_water_indices, world.width
    )
    if closest_idx != -1:
        target_pos = Position(x=closest_idx % world.width, y=closest_idx // world.width)
        _set_position_trace(
            agent,
            DecisionSource.VISIBLE_RESOURCE,
            ResourceKind.WATER,
            target_pos,
            "visible water in range",
        )
        return execute_move_toward(agent, world, target_pos, rng)

    remembered: ResourceMemory | None = memory.best_memory(
        ResourceKind.WATER,
        agent.position,
        tick,
        config,
    )
    if remembered is not None:
        confidence: float = remembered.decayed_confidence(tick, config)
        if agent.position.manhattan_to(remembered.position) <= 1:
            _set_memory_trace(
                agent,
                DecisionSource.SEARCH_NEAR_MEMORY,
                remembered,
                confidence,
                "searching near remembered water",
            )
            memory.mark_failure(ResourceKind.WATER, remembered.position, tick)
            return execute_search_near(agent, world, remembered, rng)
        if confidence < 0.18:
            _set_memory_trace(
                agent,
                DecisionSource.SEARCH_NEAR_MEMORY,
                remembered,
                confidence,
                "searching near remembered water",
            )
            return execute_search_near(agent, world, remembered, rng)
        _set_memory_trace(
            agent,
            DecisionSource.REMEMBERED_RESOURCE,
            remembered,
            confidence,
            "returning to remembered water",
        )
        return execute_move_toward(agent, world, remembered.position, rng)

    agent.decision_trace = DecisionTrace(
        source=DecisionSource.EXPLORE,
        target_kind=ResourceKind.WATER.value,
        reason="no known water; exploring",
    )
    return execute_explore(agent, world, rng)


def _resolve_food_goal(
    agent: AgentState,
    memory: AgentMemory,
    visible_food_indices: NDArray[np.int64],
    world: World,
    tick: int,
    rng: random.Random,
    config: SimConfig,
) -> str:
    eat_position: Position | None = world.nearest_edible_position(agent.position)
    if eat_position is not None:
        _set_position_trace(
            agent,
            DecisionSource.CURRENT_TILE_RESOURCE,
            ResourceKind.FOOD,
            eat_position,
            "current or adjacent food is edible",
        )
        return execute_eat(agent, memory, world, tick, config)

    closest_idx: int = _closest_visible_idx(
        agent.position.x, agent.position.y, visible_food_indices, world.width
    )
    if closest_idx != -1:
        target_pos = Position(x=closest_idx % world.width, y=closest_idx // world.width)
        _set_position_trace(
            agent,
            DecisionSource.VISIBLE_RESOURCE,
            ResourceKind.FOOD,
            target_pos,
            "visible food in range",
        )
        return execute_move_toward(agent, world, target_pos, rng)

    remembered: ResourceMemory | None = memory.best_memory(
        ResourceKind.FOOD,
        agent.position,
        tick,
        config,
    )
    if remembered is not None:
        confidence: float = remembered.decayed_confidence(tick, config)
        if agent.position.manhattan_to(remembered.position) <= 1:
            _set_memory_trace(
                agent,
                DecisionSource.SEARCH_NEAR_MEMORY,
                remembered,
                confidence,
                "searching near remembered food",
            )
            memory.mark_failure(ResourceKind.FOOD, remembered.position, tick)
            return execute_search_near(agent, world, remembered, rng)
        if agent.position.manhattan_to(remembered.position) <= max(
            2, remembered.search_radius
        ):
            _set_memory_trace(
                agent,
                DecisionSource.SEARCH_NEAR_MEMORY,
                remembered,
                confidence,
                "searching near remembered food",
            )
            return execute_search_near(agent, world, remembered, rng)
        _set_memory_trace(
            agent,
            DecisionSource.REMEMBERED_RESOURCE,
            remembered,
            confidence,
            "returning to remembered food",
        )
        return execute_move_toward(agent, world, remembered.position, rng)

    agent.decision_trace = DecisionTrace(
        source=DecisionSource.EXPLORE,
        target_kind=ResourceKind.FOOD.value,
        reason="no known food; exploring",
    )
    return execute_explore(agent, world, rng)


def _set_position_trace(
    agent: AgentState,
    source: DecisionSource,
    kind: ResourceKind,
    position: Position,
    reason: str,
) -> None:
    agent.decision_trace = DecisionTrace(
        source=source,
        target_kind=kind.value,
        target_x=position.x,
        target_y=position.y,
        reason=reason,
    )


def _set_memory_trace(
    agent: AgentState,
    source: DecisionSource,
    memory: ResourceMemory,
    confidence: float,
    reason: str,
) -> None:
    agent.decision_trace = DecisionTrace(
        source=source,
        target_kind=memory.kind.value,
        target_x=memory.position.x,
        target_y=memory.position.y,
        memory_confidence=confidence,
        memory_successful_uses=memory.successful_uses,
        memory_failed_uses=memory.failed_uses,
        reason=reason,
    )


def _closest_visible_idx(
    agent_x: int,
    agent_y: int,
    indices: NDArray[np.int64],
    width: int,
) -> int:
    """Find the nearest tile index using purely vectorized Euclidean distance."""
    if indices.size == 0:
        return -1

    xs: NDArray[np.int64] = indices % width
    ys: NDArray[np.int64] = indices // width
    distances: NDArray[np.float64] = np.sqrt((xs - agent_x) ** 2 + (ys - agent_y) ** 2)
    best_loc: int = int(np.argmin(distances))
    return int(indices[best_loc])
