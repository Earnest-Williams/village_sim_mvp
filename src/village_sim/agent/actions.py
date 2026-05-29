"""Action execution helpers."""

from __future__ import annotations

import random
from collections import deque

from village_sim.agent.memory import AgentMemory, ResourceMemory
from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.types import ActionKind, MoveCandidate, Position, ResourceKind, TerrainKind
from village_sim.world.grid import index_of, iter_neighbor_positions, iter_positions_in_radius
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
    memory.mark_success(ResourceKind.WATER, drink_position, tick, world.water_at(drink_position))
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
    memory.mark_success(ResourceKind.FOOD, eat_position, tick, world.food_at(eat_position))
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

    candidates: list[Position] = []
    for position in iter_positions_in_radius(
        world.width,
        world.height,
        memory.position,
        memory.search_radius,
    ):
        if world.is_passable(position):
            candidates.append(position)
    if not candidates:
        return execute_explore(agent, world, rng)

    best_position: Position | None = None
    best_score: float = -1_000_000.0
    for position in candidates:
        visit_count: int = agent.visited_counts[index_of(world.width, position)]
        distance: int = agent.position.manhattan_to(position)
        score: float = -float(visit_count) * 0.55 - float(distance) * 0.12 + rng.random() * 0.15
        if score > best_score:
            best_score = score
            best_position = position

    if best_position is None:
        return execute_explore(agent, world, rng)
    return execute_move_toward(agent, world, best_position, rng)


def choose_step_toward(
    agent: AgentState,
    world: World,
    target: Position,
    rng: random.Random,
) -> Position | None:
    """Choose one step using a small A* search.

    This is intentionally still simple and portable. It prevents the MVP agent from
    getting trapped by rocks while trying to walk greedily toward water or food.
    """

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

    path: list[Position] | None = _find_path(world, agent.position, reachable_target)
    if path is not None and len(path) >= 2:
        agent.path = path[1:]
        return path[1]

    agent.path = []
    return _choose_greedy_step(agent, world, target, rng)


def _reachable_target(world: World, target: Position) -> Position | None:
    if world.in_bounds(target) and world.is_passable(target):
        return target
    best: Position | None = None
    best_distance: float = 1_000_000.0
    for neighbor in iter_neighbor_positions(world.width, world.height, target, False):
        if not world.is_passable(neighbor):
            continue
        distance: float = neighbor.distance_to(target)
        if distance < best_distance:
            best_distance = distance
            best = neighbor
    return best


def _find_path(world: World, start: Position, target: Position) -> list[Position] | None:
    """Find an unweighted passable-grid path with bounded breadth-first search."""

    frontier: deque[Position] = deque([start])
    came_from: dict[Position, Position | None] = {start: None}
    expansions: int = 0
    max_expansions: int = min(world.width * world.height, 2_000)

    while frontier and expansions < max_expansions:
        current: Position = frontier.popleft()
        expansions += 1
        if current == target:
            return _reconstruct_path(came_from, current)

        neighbors: list[Position] = list(
            iter_neighbor_positions(world.width, world.height, current, False)
        )
        neighbors.sort(key=lambda candidate: candidate.distance_to(target))
        for neighbor in neighbors:
            if neighbor in came_from:
                continue
            if not world.is_passable(neighbor):
                continue
            came_from[neighbor] = current
            frontier.append(neighbor)

    return None


def _reconstruct_path(
    came_from: dict[Position, Position | None],
    current: Position,
) -> list[Position]:
    path: list[Position] = [current]
    while came_from[current] is not None:
        previous: Position | None = came_from[current]
        assert previous is not None
        current = previous
        path.append(current)
    path.reverse()
    return path


def _choose_greedy_step(
    agent: AgentState,
    world: World,
    target: Position,
    rng: random.Random,
) -> Position | None:
    candidates: list[MoveCandidate] = []
    current_distance: float = agent.position.distance_to(target)
    current_height: float = world.height_at(agent.position)
    for neighbor in iter_neighbor_positions(world.width, world.height, agent.position, False):
        if not world.is_passable(neighbor):
            continue
        distance: float = neighbor.distance_to(target)
        cost: float = world.movement_cost(neighbor)
        height_delta: float = max(0.0, world.height_at(neighbor) - current_height)
        score: float = (current_distance - distance) * 4.0 - cost * 0.35 - height_delta * 1.6
        score += rng.random() * 0.03
        candidates.append(MoveCandidate(position=neighbor, score=score))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[0].position


def choose_exploration_step(
    agent: AgentState,
    world: World,
    rng: random.Random,
) -> Position | None:
    candidates: list[MoveCandidate] = []
    current_height: float = world.height_at(agent.position)
    for neighbor in iter_neighbor_positions(world.width, world.height, agent.position, False):
        if not world.is_passable(neighbor):
            continue
        index: int = index_of(world.width, neighbor)
        visit_count: int = agent.visited_counts[index]
        kind: TerrainKind = world.terrain_at(neighbor)
        terrain_bonus: float = 0.0
        if agent.thirst > 0.55 and world.height_at(neighbor) < current_height:
            terrain_bonus += 0.45
        if agent.hunger > 0.45 and kind is TerrainKind.FOREST:
            terrain_bonus += 0.28
        if kind is TerrainKind.WATER:
            terrain_bonus += 0.18
        score: float = terrain_bonus - float(visit_count) * 0.30 - world.movement_cost(neighbor) * 0.08
        score += rng.random() * 0.20
        candidates.append(MoveCandidate(position=neighbor, score=score))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[0].position
