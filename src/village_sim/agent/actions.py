"""Action execution helpers."""

from __future__ import annotations

import random
from collections import deque
from typing import List, Dict, Tuple, Any

from village_sim.agent.memory import AgentMemory, ResourceMemory
from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.types import (
    ActionKind,
    Position,
    ResourceKind,
    TerrainKind,
)
from village_sim.world.grid import index_of
from village_sim.world.world import World


def execute_drink(
    agent: AgentState,
    memory: AgentMemory,
    world: World,
    tick: int,
    config: SimConfig,
) -> str:
    drink_position: Position | None = world.nearest_drinkable_position(agent.position)
    agent.current_action = ActionKind.DRINK
    if drink_position is None:
        memory.mark_failure(ResourceKind.WATER, agent.position, tick)
        return "tried to drink but found no drinkable water"

    consumed: float = world.consume_water(drink_position, config.drink_amount_per_tick)
    if consumed <= 0.0:
        memory.mark_failure(ResourceKind.WATER, drink_position, tick)
        return "tried to drink but the water source was dry"

    agent.thirst = max(0.0, agent.thirst - consumed * 1.25)
    agent.health = min(1.0, agent.health + 0.004)
    memory.mark_success(
        ResourceKind.WATER, drink_position, tick, world.water_at(drink_position)
    )
    return "drank water"


def execute_eat(
    agent: AgentState,
    memory: AgentMemory,
    world: World,
    tick: int,
    config: SimConfig,
) -> str:
    eat_position: Position | None = world.nearest_edible_position(agent.position)
    agent.current_action = ActionKind.EAT
    if eat_position is None:
        memory.mark_failure(ResourceKind.FOOD, agent.position, tick)
        return "tried to eat but found no edible food"

    consumed: float = world.consume_food(eat_position, config.eat_amount_per_tick)
    if consumed <= 0.0:
        memory.mark_failure(ResourceKind.FOOD, eat_position, tick)
        return "tried to eat but the food source was empty"

    agent.hunger = max(0.0, agent.hunger - consumed * 0.95)
    agent.health = min(1.0, agent.health + 0.002)
    memory.mark_success(
        ResourceKind.FOOD, eat_position, tick, world.food_at(eat_position)
    )
    return "ate food"


def execute_sleep(agent: AgentState) -> str:
    agent.current_action = ActionKind.SLEEP
    agent.target = None
    return "slept"


def execute_move_toward(
    agent: AgentState,
    world: World,
    target: Position,
    rng: random.Random,
) -> str:
    agent.current_action = ActionKind.MOVE
    if agent.target != target:
        agent.path = []
    agent.target = target
    if agent.position == target:
        agent.path = []
        return "arrived at target"

    candidate: Position | None = choose_step_toward(agent, world, target, rng)
    if candidate is None:
        return "could not find a passable movement step"

    agent.position = candidate
    agent.distance_walked += 1
    return "moved toward target"


def execute_explore(
    agent: AgentState,
    world: World,
    rng: random.Random,
) -> str:
    agent.current_action = ActionKind.EXPLORE
    candidate: Position | None = choose_exploration_step(agent, world, rng)
    if candidate is None:
        agent.current_action = ActionKind.IDLE
        return "idled because no exploration step was available"

    agent.position = candidate
    agent.distance_walked += 1
    agent.target = None
    agent.path = []
    return "explored"


def execute_search_near(
    agent: AgentState,
    world: World,
    memory: ResourceMemory,
    rng: random.Random,
) -> str:
    agent.current_action = ActionKind.SEARCH
    if agent.position.manhattan_to(memory.position) > max(2, memory.search_radius):
        return execute_move_toward(agent, world, memory.position, rng)

    best_position: Position | None = None
    best_score: float = -1_000_000.0

    mx: int = memory.position.x
    my: int = memory.position.y
    rad: int = memory.search_radius

    min_x: int = max(0, mx - rad)
    max_x: int = min(world.width - 1, mx + rad)
    min_y: int = max(0, my - rad)
    max_y: int = min(world.height - 1, my + rad)

    for y in range(min_y, max_y + 1):
        y_dist: int = abs(y - my)
        x_rad: int = rad - y_dist
        min_x_row: int = max(0, mx - x_rad)
        max_x_row: int = min(world.width - 1, mx + x_rad)
        for x in range(min_x_row, max_x_row + 1):
            idx: int = y * world.width + x
            if world.terrain[idx] == int(TerrainKind.ROCK):
                continue

            visit_count: int = agent.visited_counts[idx]
            distance: int = abs(agent.position.x - x) + abs(agent.position.y - y)
            score: float = (
                -float(visit_count) * 0.55 - float(distance) * 0.12 + rng.random() * 0.15
            )
            if score > best_score:
                best_score = score
                best_position = Position(x=x, y=y)

    if best_position is None:
        return execute_explore(agent, world, rng)
    return execute_move_toward(agent, world, best_position, rng)


def choose_step_toward(
    agent: AgentState,
    world: World,
    target: Position,
    rng: random.Random,
) -> Position | None:
    """Choose one step using a small search over integer grids."""
    if agent.position == target:
        agent.path = []
        return agent.position

    if len(agent.path) >= 2 and agent.path[0] == agent.position:
        next_step: Position = agent.path[1]
        if world.is_passable(next_step):
            agent.path = agent.path[1:]
            return next_step

    reachable_target: Position | None = _reachable_target(world, target)
    if reachable_target is None:
        return _choose_greedy_step(agent, world, target, rng)

    path: List[Position] | None = _find_path(world, agent.position, reachable_target)
    if path is not None and len(path) >= 2:
        agent.path = path[1:]
        return path[1]

    agent.path = []
    return _choose_greedy_step(agent, world, target, rng)


def _reachable_target(world: World, target: Position) -> Position | None:
    tx: int = target.x
    ty: int = target.y
    width: int = world.width
    height: int = world.height
    t_idx: int = ty * width + tx

    if 0 <= tx < width and 0 <= ty < height and world.terrain[t_idx] != int(TerrainKind.ROCK):
        return target

    best_idx: int = -1
    best_dist: float = 1_000_000.0

    neighbors: List[int] = []
    if ty > 0:
        neighbors.append(t_idx - width)
    if ty < height - 1:
        neighbors.append(t_idx + width)
    if tx > 0:
        neighbors.append(t_idx - 1)
    if tx < width - 1:
        neighbors.append(t_idx + 1)

    for n_idx in neighbors:
        if world.terrain[n_idx] == int(TerrainKind.ROCK):
            continue
        nx: int = n_idx % width
        ny: int = n_idx // width
        dist: float = ((nx - tx) ** 2 + (ny - ty) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_idx = n_idx

    if best_idx == -1:
        return None
    return Position(x=best_idx % width, y=best_idx // width)


def _find_path(
    world: World, start: Position, target: Position
) -> List[Position] | None:
    """Find an unweighted passable-grid path with bounded breadth-first search."""
    width: int = world.width
    height: int = world.height
    start_idx: int = start.y * width + start.x
    target_idx: int = target.y * width + target.x

    frontier: deque[int] = deque([start_idx])
    came_from: Dict[int, int] = {start_idx: start_idx}
    expansions: int = 0
    max_expansions: int = min(width * height, 2_000)

    while frontier and expansions < max_expansions:
        current_idx: int = frontier.popleft()
        expansions += 1
        if current_idx == target_idx:
            return _reconstruct_path(came_from, current_idx, start_idx, width)

        cx: int = current_idx % width
        cy: int = current_idx // width

        neighbors: List[int] = []
        if cy > 0:
            neighbors.append(current_idx - width)
        if cy < height - 1:
            neighbors.append(current_idx + width)
        if cx > 0:
            neighbors.append(current_idx - 1)
        if cx < width - 1:
            neighbors.append(current_idx + 1)

        for n_idx in neighbors:
            if n_idx in came_from:
                continue
            if world.terrain[n_idx] == int(TerrainKind.ROCK):
                continue
            came_from[n_idx] = current_idx
            frontier.append(n_idx)

    return None


def _reconstruct_path(
    came_from: Dict[int, int],
    current_idx: int,
    start_idx: int,
    width: int,
) -> List[Position]:
    path_indices: List[int] = [current_idx]
    while current_idx != start_idx:
        current_idx = came_from[current_idx]
        path_indices.append(current_idx)
    path_indices.reverse()
    return [Position(x=idx % width, y=idx // width) for idx in path_indices]


def _choose_greedy_step(
    agent: AgentState,
    world: World,
    target: Position,
    rng: random.Random,
) -> Position | None:
    cx: int = agent.position.x
    cy: int = agent.position.y
    width: int = world.width
    height: int = world.height
    current_idx: int = cy * width + cx

    current_distance: float = agent.position.distance_to(target)
    current_height: float = float(world.height_map[current_idx])

    best_idx: int = -1
    best_score: float = -1_000_000.0

    neighbors: List[int] = []
    if cy > 0:
        neighbors.append(current_idx - width)
    if cy < height - 1:
        neighbors.append(current_idx + width)
    if cx > 0:
        neighbors.append(current_idx - 1)
    if cx < width - 1:
        neighbors.append(current_idx + 1)

    for n_idx in neighbors:
        if world.terrain[n_idx] == int(TerrainKind.ROCK):
            continue
        nx: int = n_idx % width
        ny: int = n_idx // width
        dist: float = ((nx - target.x) ** 2 + (ny - target.y) ** 2) ** 0.5
        cost: float = float(world.movement_costs[n_idx])
        h_delta: float = max(0.0, float(world.height_map[n_idx]) - current_height)

        score: float = (
            (current_distance - dist) * 4.0 - cost * 0.35 - h_delta * 1.6
        )
        score += rng.random() * 0.03

        if score > best_score:
            best_score = score
            best_idx = n_idx

    if best_idx == -1:
        return None
    return Position(x=best_idx % width, y=best_idx // width)


def choose_exploration_step(
    agent: AgentState,
    world: World,
    rng: random.Random,
) -> Position | None:
    cx: int = agent.position.x
    cy: int = agent.position.y
    width: int = world.width
    height: int = world.height
    current_idx: int = cy * width + cx
    current_height: float = float(world.height_map[current_idx])

    best_idx: int = -1
    best_score: float = -1_000_000.0

    neighbors: List[int] = []
    if cy > 0:
        neighbors.append(current_idx - width)
    if cy < height - 1:
        neighbors.append(current_idx + width)
    if cx > 0:
        neighbors.append(current_idx - 1)
    if cx < width - 1:
        neighbors.append(current_idx + 1)

    for n_idx in neighbors:
        kind_val: int = int(world.terrain[n_idx])
        if kind_val == int(TerrainKind.ROCK):
            continue

        visit_count: int = agent.visited_counts[n_idx]
        terrain_bonus: float = 0.0

        if agent.thirst > 0.55 and float(world.height_map[n_idx]) < current_height:
            terrain_bonus += 0.45
        if agent.hunger > 0.45 and kind_val == int(TerrainKind.FOREST):
            terrain_bonus += 0.28
        if kind_val == int(TerrainKind.WATER):
            terrain_bonus += 0.18

        score: float = (
            terrain_bonus
            - float(visit_count) * 0.30
            - float(world.movement_costs[n_idx]) * 0.08
        )
        score += rng.random() * 0.20

        if score > best_score:
            best_score = score
            best_idx = n_idx

    if best_idx == -1:
        return None
    return Position(x=best_idx % width, y=best_idx // width)
