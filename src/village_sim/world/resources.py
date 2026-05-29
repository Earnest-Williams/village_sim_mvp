"""Food and water resource initialization/update helpers."""

from __future__ import annotations

import random

from village_sim.core.config import SimConfig
from village_sim.core.types import TerrainKind
from village_sim.world.grid import index_of, iter_neighbor_positions, iter_positions


def initialize_water(terrain: list[int]) -> list[float]:
    """Create initial water levels from terrain classes."""

    water: list[float] = []
    for terrain_value in terrain:
        if TerrainKind(terrain_value) is TerrainKind.WATER:
            water.append(1.0)
        else:
            water.append(0.0)
    return water


def initialize_food(
    width: int,
    height: int,
    terrain: list[int],
    rng: random.Random,
) -> tuple[list[float], list[float]]:
    """Place food in plausible forest/grass-edge cells.

    Returns `(food_amount, food_capacity)`. Capacity lets food regrow only on
    real food patches instead of turning every viable cell into food over time.
    """

    food: list[float] = [0.0 for _ in terrain]
    capacity: list[float] = [0.0 for _ in terrain]
    for position in iter_positions(width, height):
        index: int = index_of(width, position)
        kind: TerrainKind = TerrainKind(terrain[index])
        if kind is TerrainKind.WATER or kind is TerrainKind.ROCK:
            continue

        forest_neighbors: int = 0
        grass_neighbors: int = 0
        for neighbor in iter_neighbor_positions(width, height, position, True):
            neighbor_kind: TerrainKind = TerrainKind(terrain[index_of(width, neighbor)])
            if neighbor_kind is TerrainKind.FOREST:
                forest_neighbors += 1
            elif neighbor_kind is TerrainKind.GRASS:
                grass_neighbors += 1

        edge_bonus: float = 0.050 if forest_neighbors > 0 and grass_neighbors > 0 else 0.0
        base_chance: float = 0.012
        if kind is TerrainKind.FOREST:
            base_chance = 0.050
        elif kind is TerrainKind.GRASS:
            base_chance = 0.026
        elif kind is TerrainKind.HILL:
            base_chance = 0.010

        if rng.random() < base_chance + edge_bonus:
            patch_capacity: float = 0.40 + rng.random() * 0.60
            capacity[index] = patch_capacity
            food[index] = patch_capacity * (0.45 + rng.random() * 0.55)

    return food, capacity


def regrow_food(
    width: int,
    height: int,
    food: list[float],
    food_capacity: list[float],
    config: SimConfig,
) -> None:
    """Regrow a small amount of food on established food patches."""

    del width, height

    for index, capacity in enumerate(food_capacity):
        if capacity <= 0.0:
            continue
        food[index] = min(capacity, food[index] + config.food_regrowth_per_tick)
