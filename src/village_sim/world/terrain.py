"""Terrain generation."""

from __future__ import annotations

import random

from village_sim.core.types import TerrainKind
from village_sim.world.grid import index_of, iter_neighbor_positions, iter_positions


def generate_height_map(width: int, height: int, rng: random.Random) -> list[float]:
    """Generate a smoothed, normalized height map."""

    size: int = width * height
    values: list[float] = [rng.random() for _ in range(size)]

    # Several smoothing passes produce broad terrain features without dependencies.
    for _ in range(7):
        next_values: list[float] = values.copy()
        for position in iter_positions(width, height):
            total: float = values[index_of(width, position)]
            count: int = 1
            for neighbor in iter_neighbor_positions(width, height, position, True):
                total += values[index_of(width, neighbor)]
                count += 1
            next_values[index_of(width, position)] = total / float(count)
        values = next_values

    min_value: float = min(values)
    max_value: float = max(values)
    spread: float = max_value - min_value
    if spread <= 0.000_001:
        return [0.5 for _ in range(size)]

    return [(value - min_value) / spread for value in values]


def estimate_slope(
    width: int, height: int, height_map: list[float], index: int
) -> float:
    position = divmod(index, width)
    y: int = position[0]
    x: int = position[1]
    center: float = height_map[index]
    max_delta: float = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx: int = x + dx
            ny: int = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbor_index: int = ny * width + nx
                max_delta = max(max_delta, abs(center - height_map[neighbor_index]))
    return max_delta


def classify_terrain(
    width: int,
    height: int,
    height_map: list[float],
    rng: random.Random,
) -> list[int]:
    """Classify terrain kinds from height, slope, and deterministic randomness."""

    terrain: list[int] = []
    for index, height_value in enumerate(height_map):
        slope: float = estimate_slope(width, height, height_map, index)
        if height_value < 0.165:
            terrain.append(int(TerrainKind.WATER))
        elif height_value > 0.86 or slope > 0.145:
            terrain.append(int(TerrainKind.ROCK))
        elif height_value > 0.70:
            terrain.append(int(TerrainKind.HILL))
        elif 0.24 < height_value < 0.74 and rng.random() < 0.43:
            terrain.append(int(TerrainKind.FOREST))
        else:
            terrain.append(int(TerrainKind.GRASS))
    return terrain


def walk_cost(terrain_kind: int, slope: float) -> float:
    kind: TerrainKind = TerrainKind(terrain_kind)
    base: float
    if kind is TerrainKind.WATER:
        base = 2.8
    elif kind is TerrainKind.GRASS:
        base = 1.0
    elif kind is TerrainKind.FOREST:
        base = 1.35
    elif kind is TerrainKind.HILL:
        base = 1.75
    else:
        base = 3.8
    return base + slope * 5.0
