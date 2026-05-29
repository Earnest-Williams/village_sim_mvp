"""Terrain generation."""

from __future__ import annotations

import random

from village_sim.core.types import Position, TerrainKind
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
        if height_value > 0.88 or slope > 0.155:
            terrain.append(int(TerrainKind.ROCK))
        elif height_value > 0.70:
            terrain.append(int(TerrainKind.HILL))
        elif 0.20 < height_value < 0.76 and rng.random() < 0.38:
            terrain.append(int(TerrainKind.FOREST))
        else:
            terrain.append(int(TerrainKind.GRASS))
    return terrain


def carve_stream(
    width: int,
    height: int,
    height_map: list[float],
    terrain: list[int],
    rng: random.Random,
) -> None:
    """Carve a narrow connected stream from one map edge to the opposite edge."""

    if width <= 0 or height <= 0:
        return
    if len(height_map) != width * height or len(terrain) != width * height:
        raise ValueError("height_map and terrain must match width * height")

    vertical: bool = rng.random() < 0.5
    if vertical:
        start_x: int = _lowest_edge_coordinate(width, height, height_map, "north")
        end_x: int = _lowest_edge_coordinate(width, height, height_map, "south")
        x: int = start_x
        _mark_stream_cell(width, terrain, Position(x=x, y=0))
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
                height_map=height_map,
                rng=rng,
            )
            _mark_stream_connector(
                width, terrain, Position(x=previous_x, y=y), Position(x=x, y=y)
            )
    else:
        start_y = _lowest_edge_coordinate(width, height, height_map, "west")
        end_y = _lowest_edge_coordinate(width, height, height_map, "east")
        y = start_y
        _mark_stream_cell(width, terrain, Position(x=0, y=y))
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
                height_map=height_map,
                rng=rng,
            )
            _mark_stream_connector(
                width, terrain, Position(x=x, y=previous_y), Position(x=x, y=y)
            )


def _lowest_edge_coordinate(
    width: int, height: int, height_map: list[float], edge: str
) -> int:
    best_coordinate: int = 0
    best_height: float = 999_999.0
    if edge == "north":
        for x in range(width):
            value: float = height_map[index_of(width, Position(x=x, y=0))]
            if value < best_height:
                best_height = value
                best_coordinate = x
    elif edge == "south":
        y: int = height - 1
        for x in range(width):
            value = height_map[index_of(width, Position(x=x, y=y))]
            if value < best_height:
                best_height = value
                best_coordinate = x
    elif edge == "west":
        for y in range(height):
            value = height_map[index_of(width, Position(x=0, y=y))]
            if value < best_height:
                best_height = value
                best_coordinate = y
    elif edge == "east":
        x = width - 1
        for y in range(height):
            value = height_map[index_of(width, Position(x=x, y=y))]
            if value < best_height:
                best_height = value
                best_coordinate = y
    else:
        raise ValueError(f"unknown stream edge: {edge}")
    return best_coordinate


def _choose_next_stream_axis_value(
    *,
    current: int,
    target: int,
    cross_limit: int,
    forward: int,
    forward_limit: int,
    vertical: bool,
    width: int,
    height_map: list[float],
    rng: random.Random,
) -> int:
    remaining: int = max(1, forward_limit - 1 - forward)
    candidates: list[int] = [current]
    if current > 0:
        candidates.append(current - 1)
    if current + 1 < cross_limit:
        candidates.append(current + 1)

    best_value: int = current
    best_score: float = 999_999.0
    for candidate in candidates:
        if vertical:
            position = Position(x=candidate, y=forward)
        else:
            position = Position(x=forward, y=candidate)
        target_pressure: float = abs(candidate - target) / float(remaining)
        meander: float = rng.random() * 0.20
        height_score: float = height_map[index_of(width, position)] * 0.85
        score: float = target_pressure + height_score + meander
        if score < best_score:
            best_score = score
            best_value = candidate
    return best_value


def _mark_stream_cell(width: int, terrain: list[int], position: Position) -> None:
    terrain[index_of(width, position)] = int(TerrainKind.WATER)


def _mark_stream_connector(
    width: int, terrain: list[int], start: Position, end: Position
) -> None:
    if start.y == end.y:
        min_x: int = min(start.x, end.x)
        max_x: int = max(start.x, end.x)
        for x in range(min_x, max_x + 1):
            _mark_stream_cell(width, terrain, Position(x=x, y=start.y))
        return
    if start.x == end.x:
        min_y: int = min(start.y, end.y)
        max_y: int = max(start.y, end.y)
        for y in range(min_y, max_y + 1):
            _mark_stream_cell(width, terrain, Position(x=start.x, y=y))
        return
    raise ValueError("stream connector endpoints must share an axis")


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
