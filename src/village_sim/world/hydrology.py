"""Simple downhill water simulation."""

from __future__ import annotations

import random

from village_sim.core.config import SimConfig
from village_sim.core.types import TerrainKind
from village_sim.world.grid import index_of, iter_neighbor_positions, iter_positions


def step_hydrology(
    width: int,
    height: int,
    height_map: list[float],
    terrain: list[int],
    water: list[float],
    rng: random.Random,
    config: SimConfig,
) -> bool:
    """Advance water one tick and return whether rain occurred."""

    raining: bool = rng.random() < config.rain_chance_per_tick
    if raining:
        for rain_index in range(len(water)):
            if TerrainKind(terrain[rain_index]) is not TerrainKind.ROCK:
                water[rain_index] = min(2.0, water[rain_index] + config.rain_amount)

    delta: list[float] = [0.0 for _ in water]
    for position in iter_positions(width, height):
        index: int = index_of(width, position)
        amount: float = water[index]
        if amount < config.min_flow_water:
            continue

        current_height: float = height_map[index] + amount * 0.025
        best_neighbor_index: int | None = None
        best_neighbor_height: float = current_height
        for neighbor in iter_neighbor_positions(width, height, position, False):
            neighbor_index: int = index_of(width, neighbor)
            neighbor_height: float = (
                height_map[neighbor_index] + water[neighbor_index] * 0.025
            )
            if neighbor_height < best_neighbor_height:
                best_neighbor_height = neighbor_height
                best_neighbor_index = neighbor_index

        if best_neighbor_index is None:
            continue

        height_drop: float = current_height - best_neighbor_height
        flow: float = min(
            amount,
            amount * config.downhill_flow_fraction * max(0.20, height_drop * 12.0),
        )
        if flow <= 0.0:
            continue
        delta[index] -= flow
        delta[best_neighbor_index] += flow

    for index, change in enumerate(delta):
        water[index] = max(0.0, water[index] + change)
        if TerrainKind(terrain[index]) is TerrainKind.WATER:
            water[index] = max(water[index], 0.65)
        else:
            water[index] = max(0.0, water[index] - config.evaporation_per_tick)

    return raining
