"""Vectorized downhill water simulation."""

from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray

from village_sim.core.config import SimConfig
from village_sim.core.types import TerrainKind

FloatGrid = NDArray[np.float64]
IntGrid = NDArray[np.int64]
BoolGrid = NDArray[np.bool_]
MutableFloatGrid = list[float] | FloatGrid
TerrainGrid = list[int] | IntGrid

_NEIGHBOR_COUNT = 4


def _as_float_grid(values: MutableFloatGrid) -> FloatGrid:
    return np.asarray(values, dtype=np.float64)


def _as_int_grid(values: TerrainGrid) -> IntGrid:
    return np.asarray(values, dtype=np.int64)


def _sync_float_grid(target: MutableFloatGrid, values: FloatGrid) -> None:
    if isinstance(target, np.ndarray):
        target[...] = values
        return
    target[:] = values.astype(np.float64, copy=False).tolist()


def _neighbor_indices(width: int, height: int) -> IntGrid:
    indices: IntGrid = np.arange(width * height, dtype=np.int64).reshape(height, width)
    neighbors: IntGrid = np.full((width * height, _NEIGHBOR_COUNT), -1, dtype=np.int64)
    north_rows: IntGrid = indices[1:, :].reshape(-1)
    west_rows: IntGrid = indices[:, 1:].reshape(-1)
    east_rows: IntGrid = indices[:, :-1].reshape(-1)
    south_rows: IntGrid = indices[:-1, :].reshape(-1)
    neighbors[north_rows, 0] = indices[:-1, :].reshape(-1)
    neighbors[west_rows, 1] = indices[:, :-1].reshape(-1)
    neighbors[east_rows, 2] = indices[:, 1:].reshape(-1)
    neighbors[south_rows, 3] = indices[1:, :].reshape(-1)
    return neighbors


def step_hydrology(
    width: int,
    height: int,
    height_map: MutableFloatGrid,
    terrain: TerrainGrid,
    water: MutableFloatGrid,
    rng: random.Random,
    config: SimConfig,
) -> bool:
    """Advance water one tick and return whether rain occurred."""

    cell_count: int = width * height
    heights: FloatGrid = _as_float_grid(height_map).reshape(cell_count)
    terrain_values: IntGrid = _as_int_grid(terrain).reshape(cell_count)
    water_values: FloatGrid = _as_float_grid(water).reshape(cell_count).copy()

    raining: bool = rng.random() < config.rain_chance_per_tick
    rock_mask: BoolGrid = terrain_values == int(TerrainKind.ROCK)
    if raining:
        water_values[~rock_mask] = np.minimum(
            2.0, water_values[~rock_mask] + config.rain_amount
        )

    active_mask: BoolGrid = water_values >= config.min_flow_water
    if np.any(active_mask):
        neighbors: IntGrid = _neighbor_indices(width, height)
        valid_neighbors: BoolGrid = neighbors >= 0
        safe_neighbors: IntGrid = np.where(valid_neighbors, neighbors, 0)
        surface: FloatGrid = heights + water_values * 0.025
        neighbor_surface: FloatGrid = surface[safe_neighbors]
        neighbor_surface = np.where(valid_neighbors, neighbor_surface, np.inf)
        best_slots: IntGrid = np.argmin(neighbor_surface, axis=1)
        best_surface: FloatGrid = neighbor_surface[np.arange(cell_count), best_slots]
        downhill_mask: BoolGrid = active_mask & (best_surface < surface)
        source_indices: IntGrid = np.flatnonzero(downhill_mask).astype(np.int64)
        if source_indices.size > 0:
            target_indices: IntGrid = safe_neighbors[
                source_indices, best_slots[source_indices]
            ]
            height_drop: FloatGrid = (
                surface[source_indices] - best_surface[source_indices]
            )
            flow_scale: FloatGrid = config.downhill_flow_fraction * np.maximum(
                0.20, height_drop * 12.0
            )
            flows: FloatGrid = np.minimum(
                water_values[source_indices], water_values[source_indices] * flow_scale
            )
            positive_mask: BoolGrid = flows > 0.0
            if np.any(positive_mask):
                source_indices = source_indices[positive_mask]
                target_indices = target_indices[positive_mask]
                flows = flows[positive_mask]
                delta: FloatGrid = np.zeros(cell_count, dtype=np.float64)
                np.add.at(delta, source_indices, -flows)
                np.add.at(delta, target_indices, flows)
                water_values = np.maximum(0.0, water_values + delta)

    water_terrain_mask: BoolGrid = terrain_values == int(TerrainKind.WATER)
    water_values = np.where(
        water_terrain_mask,
        np.maximum(water_values, 0.65),
        np.maximum(0.0, water_values - config.evaporation_per_tick),
    )
    _sync_float_grid(water, water_values)
    return raining
