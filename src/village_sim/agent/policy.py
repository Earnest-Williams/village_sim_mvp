"""Utility-style survival policy."""

from __future__ import annotations

import random
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
from village_sim.agent.perception import Observation
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
    visible_water_indices: NDArray[np.int64] | Observation,
    visible_food_indices: NDArray[np.int64] | World,
    world: World | SimClock,
    clock: SimClock | random.Random,
    rng: random.Random | SimConfig,
    config: SimConfig | None = None,
) -> str:
    """Resolve a goal to a concrete action and execute it over pure index arrays."""

    if isinstance(visible_water_indices, Observation):
        actual_world, actual_clock, actual_rng, actual_config = _legacy_args(
            visible_food_indices,
            world,
            clock,
            rng,
            config,
        )
        actual_visible_water_indices: NDArray[np.int64] = _observation_indices(
            visible_water_indices,
            ResourceKind.WATER,
            actual_world.width,
        )
        actual_visible_food_indices: NDArray[np.int64] = _observation_indices(
            visible_water_indices,
            ResourceKind.FOOD,
            actual_world.width,
        )
    else:
        if not isinstance(visible_food_indices, np.ndarray):
            raise TypeError("visible_food_indices must be an ndarray")
        if not isinstance(world, World):
            raise TypeError("world must be a World")
        if not isinstance(clock, SimClock):
            raise TypeError("clock must be a SimClock")
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be a Random")
        if config is None:
            raise TypeError("config is required")
        actual_visible_water_indices = visible_water_indices
        actual_visible_food_indices = visible_food_indices
        actual_world = world
        actual_clock = clock
        actual_rng = rng
        actual_config = config

    agent.decision_trace = DecisionTrace()
    agent.current_goal = choose_goal(agent, actual_clock)

    if agent.current_goal is GoalKind.GET_WATER:
        return _resolve_water_goal(
            agent,
            memory,
            actual_visible_water_indices,
            actual_world,
            actual_clock.tick,
            actual_rng,
            actual_config,
        )
    if agent.current_goal is GoalKind.GET_FOOD:
        return _resolve_food_goal(
            agent,
            memory,
            actual_visible_food_indices,
            actual_world,
            actual_clock.tick,
            actual_rng,
            actual_config,
        )
    if agent.current_goal is GoalKind.SLEEP:
        return execute_sleep(agent)
    if agent.current_goal is GoalKind.EXPLORE:
        agent.decision_trace = DecisionTrace(
            source=DecisionSource.EXPLORE,
            reason="general exploration",
        )
        return execute_explore(agent, actual_world, actual_rng)
    return "idled"


def _legacy_args(
    visible_food_indices: NDArray[np.int64] | World,
    world: World | SimClock,
    clock: SimClock | random.Random,
    rng: random.Random | SimConfig,
    config: SimConfig | None,
) -> tuple[World, SimClock, random.Random, SimConfig]:
    if not isinstance(visible_food_indices, World):
        raise TypeError("legacy world argument must be a World")
    if not isinstance(world, SimClock):
        raise TypeError("legacy clock argument must be a SimClock")
    if not isinstance(clock, random.Random):
        raise TypeError("legacy rng argument must be a Random")
    if not isinstance(rng, SimConfig):
        raise TypeError("legacy config argument must be a SimConfig")
    if config is not None:
        raise TypeError("unexpected extra config argument")
    return visible_food_indices, world, clock, rng


def _observation_indices(
    observation: Observation,
    kind: ResourceKind,
    width: int,
) -> NDArray[np.int64]:
    if kind is ResourceKind.WATER:
        if observation.visible_water_indices.size > 0:
            return observation.visible_water_indices
        return np.asarray(
            [
                item.position.y * width + item.position.x
                for item in observation.visible_water
            ],
            dtype=np.int64,
        )
    if observation.visible_food_indices.size > 0:
        return observation.visible_food_indices
    return np.asarray(
        [
            item.position.y * width + item.position.x
            for item in observation.visible_food
        ],
        dtype=np.int64,
    )


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
    """Find the nearest tile index using vectorized squared distance."""
    if indices.size == 0:
        return -1

    xs: NDArray[np.int64] = indices % width
    ys: NDArray[np.int64] = indices // width
    dx: NDArray[np.int64] = xs - agent_x
    dy: NDArray[np.int64] = ys - agent_y
    squared_distances: NDArray[np.int64] = dx * dx + dy * dy
    best_loc: int = int(np.argmin(squared_distances))
    return int(indices[best_loc])
