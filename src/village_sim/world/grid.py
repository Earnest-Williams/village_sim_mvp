"""Grid helpers."""

from __future__ import annotations

from collections.abc import Iterator

from village_sim.core.types import Position


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
