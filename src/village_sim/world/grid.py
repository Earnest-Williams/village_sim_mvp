"""Vectorized world-grid buffers and coordinate helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from village_sim.core.types import Position, TerrainKind

STRUCTURE_NONE: np.int32 = np.int32(0)
STRUCTURE_FARM: np.int32 = np.int32(1)
STRUCTURE_SHELTER: np.int32 = np.int32(2)
STRUCTURE_WORKSHOP: np.int32 = np.int32(3)


@dataclass(slots=True)
class WorldGrids:
    """Flat, cache-local grid buffers for vectorized environment mutation.

    Every per-cell plane is a one-dimensional row-major NumPy array with exactly
    ``width * height`` elements. Categorical planes use ``np.int32`` and
    continuous planes use ``np.float32`` for RL observation throughput.
    """

    width: int
    height: int
    terrain_kind: NDArray[np.int32] = field(init=False)
    structure_kind: NDArray[np.int32] = field(init=False)
    elevation: NDArray[np.float32] = field(init=False)
    structure_health: NDArray[np.float32] = field(init=False)
    crop_growth: NDArray[np.float32] = field(init=False)
    water_table: NDArray[np.float32] = field(init=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("WorldGrids dimensions must be positive")
        cell_count: int = self.width * self.height
        self.terrain_kind = np.empty(cell_count, dtype=np.int32)
        self.structure_kind = np.empty(cell_count, dtype=np.int32)
        self.elevation = np.empty(cell_count, dtype=np.float32)
        self.structure_health = np.empty(cell_count, dtype=np.float32)
        self.crop_growth = np.empty(cell_count, dtype=np.float32)
        self.water_table = np.empty(cell_count, dtype=np.float32)
        self.reset()

    @property
    def cell_count(self) -> int:
        """Return the number of cells in O(1)."""

        return int(self.terrain_kind.size)

    def reset(self) -> None:
        """Reinitialize buffers in place without reallocating grid memory."""

        self.terrain_kind.fill(np.int32(TerrainKind.GRASS))
        self.structure_kind.fill(STRUCTURE_NONE)
        self.elevation.fill(np.float32(0.0))
        self.structure_health.fill(np.float32(0.0))
        self.crop_growth.fill(np.float32(0.0))
        self.water_table.fill(np.float32(0.35))

    def copy_from_world_arrays(
        self,
        *,
        terrain: NDArray[np.int64] | NDArray[np.int32],
        elevation: NDArray[np.float64] | NDArray[np.float32],
        water: NDArray[np.float64] | NDArray[np.float32],
        food: NDArray[np.float64] | NDArray[np.float32],
    ) -> None:
        """Normalize legacy world arrays into the RL grid planes."""

        if terrain.size != self.cell_count:
            raise ValueError("terrain array size must match WorldGrids cell count")
        if elevation.size != self.cell_count:
            raise ValueError("elevation array size must match WorldGrids cell count")
        if water.size != self.cell_count:
            raise ValueError("water array size must match WorldGrids cell count")
        if food.size != self.cell_count:
            raise ValueError("food array size must match WorldGrids cell count")
        np.copyto(self.terrain_kind, terrain.ravel(), casting="unsafe")
        np.copyto(self.elevation, elevation.ravel(), casting="unsafe")
        np.copyto(self.water_table, water.ravel(), casting="unsafe")
        np.copyto(self.crop_growth, food.ravel(), casting="unsafe")
        self.structure_kind.fill(STRUCTURE_NONE)
        self.structure_health.fill(np.float32(0.0))

    def unravel_indices(
        self, indices: NDArray[np.int64] | NDArray[np.int32]
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return vectorized x/y coordinates for flat indices."""

        x_coords: NDArray[np.int32] = np.asarray(indices % self.width, dtype=np.int32)
        y_coords: NDArray[np.int32] = np.asarray(indices // self.width, dtype=np.int32)
        return x_coords, y_coords


def index_of(width: int, position: Position) -> int:
    return position.y * width + position.x


def position_of(width: int, index: int) -> Position:
    return Position(x=index % width, y=index // width)


def in_bounds(width: int, height: int, position: Position) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def iter_positions(width: int, height: int) -> Iterator[Position]:
    for y in range(height):
        for x in range(width):
            yield Position(x=x, y=y)


def iter_neighbor_positions(
    width: int,
    height: int,
    position: Position,
    include_diagonal: bool = False,
) -> Iterator[Position]:
    deltas: tuple[tuple[int, int], ...]
    if include_diagonal:
        deltas = (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        )
    else:
        deltas = ((0, -1), (-1, 0), (1, 0), (0, 1))

    for dx, dy in deltas:
        candidate: Position = Position(x=position.x + dx, y=position.y + dy)
        if in_bounds(width, height, candidate):
            yield candidate


def iter_positions_in_radius(
    width: int,
    height: int,
    center: Position,
    radius: int,
) -> Iterator[Position]:
    min_x: int = max(0, center.x - radius)
    max_x: int = min(width - 1, center.x + radius)
    min_y: int = max(0, center.y - radius)
    max_y: int = min(height - 1, center.y + radius)
    radius_sq: int = radius * radius

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            dx: int = x - center.x
            dy: int = y - center.y
            if dx * dx + dy * dy <= radius_sq:
                yield Position(x=x, y=y)
