"""Terrain generation backed by contiguous NumPy grids."""

from __future__ import annotations

import random
from typing import Literal, List, Dict, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from village_sim.core.types import Position, TerrainKind
from village_sim.world.grid import index_of

FloatGrid = NDArray[np.float64]
IntGrid = NDArray[np.int64]
BoolGrid = NDArray[np.bool_]
MutableIntGrid = list[int] | IntGrid
HeightGrid = list[float] | FloatGrid
StreamEdge = Literal["north", "south", "west", "east"]

_SMOOTHING_PASSES = 7
_EPSILON = 0.000_001
_NEIGHBOR_KERNEL_SIZE = 3


def _as_float_grid(values: HeightGrid) -> FloatGrid:
    return np.asarray(values, dtype=np.float64)


def _as_int_grid(values: MutableIntGrid) -> IntGrid:
    return np.asarray(values, dtype=np.int64)


def _sync_int_grid(target: MutableIntGrid, values: IntGrid) -> None:
    if isinstance(target, np.ndarray):
        target[...] = values
        return
    target[:] = values.astype(np.int64, copy=False).tolist()


def generate_height_map(width: int, height: int, rng: random.Random) -> FloatGrid:
    """Generate a smoothed, normalized, flat row-major height map."""

    size: int = width * height
    if size <= 0:
        return np.empty(0, dtype=np.float64)

    values: FloatGrid = np.fromiter(
        (rng.random() for _ in range(size)), dtype=np.float64, count=size
    ).reshape(height, width)

    counts: FloatGrid = np.ones_like(values)
    for _ in range(_SMOOTHING_PASSES):
        padded: FloatGrid = np.pad(values, 1, mode="constant", constant_values=0.0)
        padded_counts: FloatGrid = np.pad(
            counts, 1, mode="constant", constant_values=0.0
        )
        total: FloatGrid = np.zeros_like(values)
        total_counts: FloatGrid = np.zeros_like(values)
        for y_offset in range(_NEIGHBOR_KERNEL_SIZE):
            for x_offset in range(_NEIGHBOR_KERNEL_SIZE):
                total += padded[
                    y_offset : y_offset + height, x_offset : x_offset + width
                ]
                total_counts += padded_counts[
                    y_offset : y_offset + height, x_offset : x_offset + width
                ]
        values = total / total_counts

    min_value: float = float(values.min())
    max_value: float = float(values.max())
    spread: float = max_value - min_value
    if spread <= _EPSILON:
        return np.full(size, 0.5, dtype=np.float64)

    normalized: FloatGrid = (values - min_value) / spread
    return normalized.reshape(size)


def estimate_slope(
    width: int, height: int, height_map: HeightGrid, index: int
) -> float:
    """Return the maximum adjacent height delta for one tile."""

    if width <= 0 or height <= 0:
        return 0.0
    heights: FloatGrid = _as_float_grid(height_map).reshape(height, width)
    y: int = index // width
    x: int = index % width
    if not (0 <= x < width and 0 <= y < height):
        raise IndexError("slope index is outside the height map")

    center: float = float(heights[y, x])
    y_min: int = max(0, y - 1)
    y_max: int = min(height, y + 2)
    x_min: int = max(0, x - 1)
    x_max: int = min(width, x + 2)
    window: FloatGrid = heights[y_min:y_max, x_min:x_max]
    return float(np.max(np.abs(window - center)))


def _estimate_all_slopes(width: int, height: int, height_map: HeightGrid) -> FloatGrid:
    heights: FloatGrid = _as_float_grid(height_map).reshape(height, width)
    padded: FloatGrid = np.pad(heights, 1, mode="edge")
    max_delta: FloatGrid = np.zeros_like(heights)
    for y_offset in range(_NEIGHBOR_KERNEL_SIZE):
        for x_offset in range(_NEIGHBOR_KERNEL_SIZE):
            if x_offset == 1 and y_offset == 1:
                continue
            neighbor: FloatGrid = padded[
                y_offset : y_offset + height, x_offset : x_offset + width
            ]
            max_delta = np.maximum(max_delta, np.abs(heights - neighbor))
    return max_delta.reshape(width * height)


def classify_terrain(
    width: int,
    height: int,
    height_map: HeightGrid,
    rng: random.Random,
) -> IntGrid:
    """Classify terrain kinds from height, slope, and deterministic randomness."""

    size: int = width * height
    heights: FloatGrid = _as_float_grid(height_map).reshape(size)
    slopes: FloatGrid = _estimate_all_slopes(width, height, heights)
    terrain: IntGrid = np.full(size, int(TerrainKind.GRASS), dtype=np.int64)
    rock_mask: BoolGrid = (heights > 0.88) | (slopes > 0.155)
    hill_mask: BoolGrid = (heights > 0.70) & ~rock_mask
    terrain[hill_mask] = int(TerrainKind.HILL)
    forest_candidate_mask: BoolGrid = (
        (heights > 0.20) & (heights < 0.76) & ~rock_mask & ~hill_mask
    )
    forest_candidate_indices: IntGrid = np.flatnonzero(forest_candidate_mask).astype(
        np.int64
    )
    if forest_candidate_indices.size > 0:
        forest_rolls: FloatGrid = np.fromiter(
            (rng.random() for _ in range(forest_candidate_indices.size)),
            dtype=np.float64,
            count=int(forest_candidate_indices.size),
        )
        forest_indices: IntGrid = forest_candidate_indices[forest_rolls < 0.38]
        terrain[forest_indices] = int(TerrainKind.FOREST)
    terrain[rock_mask] = int(TerrainKind.ROCK)
    return terrain


def carve_stream(
    width: int,
    height: int,
    height_map: HeightGrid,
    terrain: MutableIntGrid,
    rng: random.Random,
) -> None:
    """Carve a narrow connected stream from one map edge to the opposite edge."""

    if width <= 0 or height <= 0:
        return
    if len(height_map) != width * height or len(terrain) != width * height:
        raise ValueError("height_map and terrain must match width * height")

    height_values: FloatGrid = _as_float_grid(height_map).reshape(height, width)
    terrain_values: IntGrid = _as_int_grid(terrain).reshape(height, width).copy()
    vertical: bool = rng.random() < 0.5
    if vertical:
        start_x: int = _lowest_edge_coordinate(width, height, height_values, "north")
        end_x: int = _lowest_edge_coordinate(width, height, height_values, "south")
        x: int = start_x
        terrain_values[0, x] = int(TerrainKind.WATER)
        for y in range(1, height):
            previous_x: int = x
            x = _choose_next_stream_axis_value(
                current=x,
                target=end_x,
                cross_limit=width,
                forward=y,
                forward_limit=height,
                vertical=True,
                width=width,
                height_map=height_values,
                rng=rng,
            )
            min_x: int = min(previous_x, x)
            max_x: int = max(previous_x, x)
            terrain_values[y, min_x : max_x + 1] = int(TerrainKind.WATER)
    else:
        start_y = _lowest_edge_coordinate(width, height, height_values, "west")
        end_y = _lowest_edge_coordinate(width, height, height_values, "east")
        y = start_y
        terrain_values[y, 0] = int(TerrainKind.WATER)
        for x in range(1, width):
            previous_y: int = y
            y = _choose_next_stream_axis_value(
                current=y,
                target=end_y,
                cross_limit=height,
                forward=x,
                forward_limit=width,
                vertical=False,
                width=width,
                height_map=height_values,
                rng=rng,
            )
            min_y: int = min(previous_y, y)
            max_y: int = max(previous_y, y)
            terrain_values[min_y : max_y + 1, x] = int(TerrainKind.WATER)

    _sync_int_grid(terrain, terrain_values.reshape(width * height))


def _lowest_edge_coordinate(
    width: int, height: int, height_map: FloatGrid, edge: StreamEdge
) -> int:
    heights: FloatGrid = height_map.reshape(height, width)
    if edge == "north":
        return int(np.argmin(heights[0, :]))
    if edge == "south":
        return int(np.argmin(heights[height - 1, :]))
    if edge == "west":
        return int(np.argmin(heights[:, 0]))
    return int(np.argmin(heights[:, width - 1]))


def _choose_next_stream_axis_value(
    *,
    current: int,
    target: int,
    cross_limit: int,
    forward: int,
    forward_limit: int,
    vertical: bool,
    width: int,
    height_map: FloatGrid,
    rng: random.Random,
) -> int:
    remaining: int = max(1, forward_limit - 1 - forward)
    candidate_values: list[int] = [current]
    if current > 0:
        candidate_values.append(current - 1)
    if current + 1 < cross_limit:
        candidate_values.append(current + 1)
    candidates: IntGrid = np.asarray(candidate_values, dtype=np.int64)
    target_pressure: FloatGrid = np.abs(candidates - target).astype(np.float64) / float(
        remaining
    )
    meander: FloatGrid = np.fromiter(
        (rng.random() * 0.20 for _ in range(candidates.size)),
        dtype=np.float64,
        count=int(candidates.size),
    )
    if vertical:
        heights: FloatGrid = height_map[forward, candidates]
    else:
        heights = height_map[candidates, forward]
    scores: FloatGrid = target_pressure + heights * 0.85 + meander
    return int(candidates[int(np.argmin(scores))])


def _mark_stream_cell(width: int, terrain: MutableIntGrid, position: Position) -> None:
    terrain[index_of(width, position)] = int(TerrainKind.WATER)


def _mark_stream_connector(
    width: int, terrain: MutableIntGrid, start: Position, end: Position
) -> None:
    terrain_values: IntGrid = _as_int_grid(terrain).copy()
    if start.y == end.y:
        min_x: int = min(start.x, end.x)
        max_x: int = max(start.x, end.x)
        indices: IntGrid = start.y * width + np.arange(min_x, max_x + 1, dtype=np.int64)
        terrain_values[indices] = int(TerrainKind.WATER)
        _sync_int_grid(terrain, terrain_values)
        return
    if start.x == end.x:
        min_y: int = min(start.y, end.y)
        max_y: int = max(start.y, end.y)
        indices = np.arange(min_y, max_y + 1, dtype=np.int64) * width + start.x
        terrain_values[indices] = int(TerrainKind.WATER)
        _sync_int_grid(terrain, terrain_values)
        return
    raise ValueError("stream connector endpoints must share an axis")


def compute_all_movement_costs(
    width: int, height: int, height_map: HeightGrid, terrain: MutableIntGrid
) -> FloatGrid:
    """Precompute movement costs for the entire grid using vectorized arrays."""
    if width <= 0 or height <= 0:
        return np.empty(0, dtype=np.float64)

    size: int = width * height
    heights: FloatGrid = _as_float_grid(height_map).reshape(size)
    terrain_grid: IntGrid = _as_int_grid(terrain).reshape(size)
    slopes: FloatGrid = _estimate_all_slopes(width, height, heights)
    costs: FloatGrid = np.full(size, 3.8, dtype=np.float64)

    costs[terrain_grid == int(TerrainKind.WATER)] = 2.8
    costs[terrain_grid == int(TerrainKind.GRASS)] = 1.0
    costs[terrain_grid == int(TerrainKind.FOREST)] = 1.35
    costs[terrain_grid == int(TerrainKind.HILL)] = 1.75

    costs += slopes * 5.0
    return costs


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
