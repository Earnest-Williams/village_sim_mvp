"""World state and world-level operations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from village_sim.core.config import SimConfig
from village_sim.core.types import Position, TerrainKind
from village_sim.world.discoverables import Discoverable, update_discoverables_daily
from village_sim.world.grid import index_of, iter_neighbor_positions, iter_positions
from village_sim.world.hydrology import step_hydrology
from village_sim.world.resources import initialize_food, initialize_water, regrow_food
from village_sim.world.terrain import classify_terrain, generate_height_map, estimate_slope, walk_cost


@dataclass(slots=True)
class World:
    """Portable world state.

    Arrays are flat row-major lists. This makes the Python MVP simple while keeping a
    direct migration path to NumPy arrays or a native memory layout later.
    """

    width: int
    height: int
    height_map: list[float]
    terrain: list[int]
    water: list[float]
    food: list[float]
    food_capacity: list[float]
    discoverables: dict[str, Discoverable] = field(default_factory=dict)

    def index(self, position: Position) -> int:
        return index_of(self.width, position)

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def terrain_at(self, position: Position) -> TerrainKind:
        return TerrainKind(self.terrain[self.index(position)])

    def height_at(self, position: Position) -> float:
        return self.height_map[self.index(position)]

    def water_at(self, position: Position) -> float:
        return self.water[self.index(position)]

    def food_at(self, position: Position) -> float:
        return self.food[self.index(position)]

    def consume_water(self, position: Position, amount: float) -> float:
        index: int = self.index(position)
        available: float = self.water[index]
        consumed: float = min(available, amount)
        if TerrainKind(self.terrain[index]) is TerrainKind.WATER:
            # Permanent water terrain can be drunk from without fully draining.
            self.water[index] = max(0.65, available - consumed * 0.05)
        else:
            self.water[index] = max(0.0, available - consumed)
        return consumed

    def consume_food(self, position: Position, amount: float) -> float:
        index: int = self.index(position)
        consumed: float = min(self.food[index], amount)
        self.food[index] = max(0.0, self.food[index] - consumed)
        return consumed

    def is_passable(self, position: Position) -> bool:
        if not self.in_bounds(position):
            return False
        kind: TerrainKind = self.terrain_at(position)
        return kind is not TerrainKind.ROCK

    def movement_cost(self, position: Position) -> float:
        index: int = self.index(position)
        slope: float = estimate_slope(self.width, self.height, self.height_map, index)
        return walk_cost(self.terrain[index], slope)

    def step_environment(self, rng: random.Random, config: SimConfig, tick_of_day: int = -1) -> bool:
        raining: bool = step_hydrology(
            self.width,
            self.height,
            self.height_map,
            self.terrain,
            self.water,
            rng,
            config,
        )
        regrow_food(self.width, self.height, self.food, self.food_capacity, config)
        if tick_of_day == 0:
            update_discoverables_daily(self)
        return raining

    def nearest_drinkable_position(self, position: Position) -> Position | None:
        if self.water_at(position) >= 0.20:
            return position
        best_position: Position | None = None
        best_distance: int = 999_999
        for neighbor in iter_neighbor_positions(self.width, self.height, position, False):
            if self.water_at(neighbor) >= 0.20:
                distance: int = position.manhattan_to(neighbor)
                if distance < best_distance:
                    best_distance = distance
                    best_position = neighbor
        return best_position

    def nearest_edible_position(self, position: Position) -> Position | None:
        if self.food_at(position) >= 0.12:
            return position
        best_position: Position | None = None
        best_distance: int = 999_999
        for neighbor in iter_neighbor_positions(self.width, self.height, position, False):
            if self.food_at(neighbor) >= 0.12:
                distance: int = position.manhattan_to(neighbor)
                if distance < best_distance:
                    best_distance = distance
                    best_position = neighbor
        return best_position


def generate_world(
    config: SimConfig,
    rng: random.Random,
    discoverables: dict[str, Discoverable] | None = None,
) -> World:
    config.validate()
    height_map: list[float] = generate_height_map(config.width, config.height, rng)
    terrain: list[int] = classify_terrain(config.width, config.height, height_map, rng)
    water: list[float] = initialize_water(terrain)
    food, food_capacity = initialize_food(config.width, config.height, terrain, rng)
    return World(
        width=config.width,
        height=config.height,
        height_map=height_map,
        terrain=terrain,
        water=water,
        food=food,
        food_capacity=food_capacity,
        discoverables=discoverables or {},
    )


def choose_spawn_position(world: World, rng: random.Random) -> Position:
    """Pick a passable starting position with survivable nearby geography.

    This does not give the agent any resource knowledge. It only prevents the MVP
    from starting a stone-age human in an obviously nonviable corner of the map.
    """

    center: Position = Position(x=world.width // 2, y=world.height // 2)
    water_positions: list[Position] = [
        position
        for position in iter_positions(world.width, world.height)
        if world.water_at(position) >= 0.65
    ]
    food_positions: list[Position] = [
        position
        for position in iter_positions(world.width, world.height)
        if world.food_at(position) >= 0.18
    ]

    max_water_distance: int = max(10, min(world.width, world.height) // 4)
    max_food_distance: int = max(12, min(world.width, world.height) // 3)

    candidates: list[Position] = []
    fallback_candidates: list[Position] = []
    for position in iter_positions(world.width, world.height):
        if not world.is_passable(position):
            continue
        kind: TerrainKind = world.terrain_at(position)
        if kind is TerrainKind.WATER:
            continue
        if position.manhattan_to(center) > (world.width + world.height) // 3:
            continue

        fallback_candidates.append(position)
        nearest_water_distance: int = _nearest_distance(position, water_positions)
        nearest_food_distance: int = _nearest_distance(position, food_positions)
        if (
            nearest_water_distance <= max_water_distance
            and nearest_food_distance <= max_food_distance
        ):
            candidates.append(position)

    if not candidates:
        candidates = fallback_candidates

    if not candidates:
        for position in iter_positions(world.width, world.height):
            if world.is_passable(position) and world.terrain_at(position) is not TerrainKind.WATER:
                candidates.append(position)

    if not candidates:
        raise ValueError("world has no valid spawn positions")

    return candidates[rng.randrange(len(candidates))]


def _nearest_distance(position: Position, targets: list[Position]) -> int:
    if not targets:
        return 999_999
    return min(position.manhattan_to(target) for target in targets)
