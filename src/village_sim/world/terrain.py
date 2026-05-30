"""World state and world-level operations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from village_sim.core.config import SimConfig
from village_sim.core.types import Position, TerrainKind
from village_sim.world.discoverables import Discoverable, update_discoverables_daily
from village_sim.world.grid import index_of, iter_neighbor_positions, iter_positions
from village_sim.world.resources import initialize_food, initialize_water, regrow_food
from village_sim.world.water_system import (
    RainEvent,
    RainSystem,
    WaterSystemState,
    build_water_system_state,
    step_active_water_flow,
    step_sampled_rain,
    step_water_maintenance_bank,
)
from village_sim.world.terrain import (
    carve_stream,
    classify_terrain,
    compute_all_movement_costs,
    estimate_slope,
    generate_height_map,
    walk_cost,
)

FloatGrid = NDArray[np.float64]
IntGrid = NDArray[np.int64]


def _new_discoverables() -> dict[str, Discoverable]:
    return {}


@dataclass(slots=True)
class World:
    """Portable world state.

    Arrays are flat row-major NumPy buffers for cache-local environmental updates.
    Legacy list inputs are accepted at API boundaries and normalized during setup.
    """

    width: int
    height: int
    height_map: list[float] | FloatGrid
    terrain: list[int] | IntGrid
    water: list[float] | FloatGrid
    food: list[float] | FloatGrid
    food_capacity: list[float] | FloatGrid
    movement_costs: list[float] | FloatGrid = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    water_system: WaterSystemState = field(init=False)
    rain_system: RainSystem = field(default_factory=RainSystem)
    discoverables: dict[str, Discoverable] = field(default_factory=_new_discoverables)
    seed: int = 0
    tile_size_meters: float = 2.0

    def __post_init__(self) -> None:
        self.water_system = build_water_system_state(
            width=self.width,
            height=self.height,
            terrain=self.terrain,
            water=self.water,
        )
        if len(self.movement_costs) == 0 and self.width > 0 and self.height > 0:
            self.movement_costs = compute_all_movement_costs(
                self.width, self.height, self.height_map, self.terrain
            )

    def index(self, position: Position) -> int:
        return index_of(self.width, position)

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def terrain_at(self, position: Position) -> TerrainKind:
        return TerrainKind(self.terrain[self.index(position)])

    def height_at(self, position: Position) -> float:
        return float(self.height_map[self.index(position)])

    def water_at(self, position: Position) -> float:
        return float(self.water[self.index(position)])

    def food_at(self, position: Position) -> float:
        return float(self.food[self.index(position)])

    def consume_water(self, position: Position, amount: float) -> float:
        index: int = self.index(position)
        available: float = float(self.water[index])
        consumed: float = min(available, amount)
        if TerrainKind(self.terrain[index]) is TerrainKind.WATER:
            # Permanent water terrain can be drunk from without fully draining.
            self.water[index] = max(0.65, available - consumed * 0.05)
        else:
            self.water[index] = max(0.0, available - consumed)
        return consumed

    def consume_food(self, position: Position, amount: float) -> float:
        index: int = self.index(position)
        consumed: float = min(float(self.food[index]), amount)
        self.food[index] = max(0.0, float(self.food[index]) - consumed)
        return consumed

    def is_passable(self, position: Position) -> bool:
        if not self.in_bounds(position):
            return False
        kind: TerrainKind = self.terrain_at(position)
        return kind is not TerrainKind.ROCK

    def movement_cost(self, position: Position) -> float:
        return float(self.movement_costs[self.index(position)])

    def step_environment(
        self, rng: random.Random, config: SimConfig, tick: int = -1
    ) -> bool:
        event: RainEvent | None = self.rain_system.tick(rng, config)
        rain_tick: int = max(0, tick)
        if event is not None and event.should_apply(rain_tick):
            step_sampled_rain(
                rng=rng,
                water=self.water,
                state=self.water_system,
                event=event,
                config=config,
            )
        step_active_water_flow(
            water=self.water,
            height_map=self.height_map,
            state=self.water_system,
            config=config,
        )
        step_water_maintenance_bank(
            tick=rain_tick,
            water=self.water,
            state=self.water_system,
            config=config,
        )
        regrow_food(self.width, self.height, self.food, self.food_capacity, config)
        if tick >= 0 and rain_tick % config.ticks_per_day == 0:
            update_discoverables_daily(self)
        return event is not None and event.is_raining

    def nearest_drinkable_position(self, position: Position) -> Position | None:
        if self.water_at(position) >= 0.20:
            return position
        best_position: Position | None = None
        best_distance: int = 999_999
        for neighbor in iter_neighbor_positions(
            self.width, self.height, position, False
        ):
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
        for neighbor in iter_neighbor_positions(
            self.width, self.height, position, False
        ):
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
    height_map: FloatGrid = generate_height_map(config.width, config.height, rng)
    terrain: IntGrid = classify_terrain(config.width, config.height, height_map, rng)
    carve_stream(config.width, config.height, height_map, terrain, rng)
    water: FloatGrid = initialize_water(terrain)
    food_values, food_capacity_values = initialize_food(
        config.width, config.height, terrain, rng
    )
    food: FloatGrid = np.asarray(food_values, dtype=np.float64)
    food_capacity: FloatGrid = np.asarray(food_capacity_values, dtype=np.float64)
    movement_costs: FloatGrid = compute_all_movement_costs(
        config.width, config.height, height_map, terrain
    )
    return World(
        width=config.width,
        height=config.height,
        height_map=height_map,
        terrain=terrain,
        water=water,
        food=food,
        food_capacity=food_capacity,
        movement_costs=movement_costs,
        discoverables=discoverables or {},
        seed=config.seed,
        tile_size_meters=config.tile_size_meters,
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
            if (
                world.is_passable(position)
                and world.terrain_at(position) is not TerrainKind.WATER
            ):
                candidates.append(position)

    if not candidates:
        raise ValueError("world has no valid spawn positions")

    return candidates[rng.randrange(len(candidates))]


def _nearest_distance(position: Position, targets: list[Position]) -> int:
    if not targets:
        return 999_999
    return min(position.manhattan_to(target) for target in targets)
