"""Agent perception."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit
from numpy.typing import NDArray

from village_sim.core.config import SimConfig
from village_sim.core.time import SimClock
from village_sim.core.types import Position, ResourceKind, ResourceSighting
from village_sim.world.discoverables import (
    DiscoverableKind,
    DiscoverableObservation,
    discoverable_at_or_adjacent,
    perceive_discoverables,
)
from village_sim.world.weather import WeatherState, make_weather_state
from village_sim.world.world import World

RESOURCE_KIND_WATER: int = 0
RESOURCE_KIND_FOOD: int = 1
RESOURCE_WATER_THRESHOLD: float = 0.25
RESOURCE_FOOD_THRESHOLD: float = 0.18


@njit(cache=True)
def _extract_visible_resources_kernel(
    agent_ids: NDArray[np.int64],
    agent_x: NDArray[np.int32],
    agent_y: NDArray[np.int32],
    agent_radii: NDArray[np.int32],
    agent_alive: NDArray[np.bool_],
    world_width: int,
    world_height: int,
    water_grid: NDArray[np.float64],
    food_grid: NDArray[np.float64],
    water_threshold: float,
    food_threshold: float,
    out_agent_ids: NDArray[np.int64],
    out_tile_indices: NDArray[np.int64],
    out_kinds: NDArray[np.int32],
    out_amounts: NDArray[np.float64],
) -> int:
    """Fill pre-allocated output arrays with visible resource rows."""

    max_sightings: int = out_tile_indices.shape[0]
    count: int = 0

    for i in range(agent_ids.shape[0]):
        if not agent_alive[i]:
            continue

        ax: int = agent_x[i]
        ay: int = agent_y[i]
        radius: int = agent_radii[i]
        agent_id: int = agent_ids[i]

        min_y: int = max(0, ay - radius)
        max_y: int = min(world_height - 1, ay + radius)
        for tile_y in range(min_y, max_y + 1):
            y_dist: int = abs(tile_y - ay)
            x_radius: int = radius - y_dist
            min_x: int = max(0, ax - x_radius)
            max_x: int = min(world_width - 1, ax + x_radius)

            for tile_x in range(min_x, max_x + 1):
                tile_index: int = tile_y * world_width + tile_x

                water_amount: float = water_grid[tile_index]
                if water_amount >= water_threshold:
                    if count >= max_sightings:
                        return count
                    out_agent_ids[count] = agent_id
                    out_tile_indices[count] = tile_index
                    out_kinds[count] = RESOURCE_KIND_WATER
                    out_amounts[count] = water_amount
                    count += 1

                food_amount: float = food_grid[tile_index]
                if food_amount >= food_threshold:
                    if count >= max_sightings:
                        return count
                    out_agent_ids[count] = agent_id
                    out_tile_indices[count] = tile_index
                    out_kinds[count] = RESOURCE_KIND_FOOD
                    out_amounts[count] = food_amount
                    count += 1

    return count


def max_resource_sightings(agent_count: int, radius: int) -> int:
    """Return a safe output row capacity for diamond-radius perception."""

    tiles_per_agent: int = 1 + 2 * radius * (radius + 1)
    return agent_count * tiles_per_agent * 2


def perceive_batch_resources(
    agent_ids: NDArray[np.int64],
    agent_x: NDArray[np.int32],
    agent_y: NDArray[np.int32],
    agent_alive: NDArray[np.bool_],
    world: World,
    clock: SimClock,
    config: SimConfig,
    out_agent_ids: NDArray[np.int64] | None = None,
    out_tile_indices: NDArray[np.int64] | None = None,
    out_kinds: NDArray[np.int32] | None = None,
    out_amounts: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.int32], NDArray[np.float64]
]:
    """Batched perception entrypoint. Emits flat arrays of visible resources."""

    radius: int = (
        config.vision_radius_day if clock.is_daylight else config.vision_radius_night
    )
    agent_radii: NDArray[np.int32] = np.full(agent_ids.shape, radius, dtype=np.int32)
    capacity: int = max_resource_sightings(agent_ids.shape[0], radius)

    if out_agent_ids is None:
        out_agent_ids = np.empty(capacity, dtype=np.int64)
    if out_tile_indices is None:
        out_tile_indices = np.empty(capacity, dtype=np.int64)
    if out_kinds is None:
        out_kinds = np.empty(capacity, dtype=np.int32)
    if out_amounts is None:
        out_amounts = np.empty(capacity, dtype=np.float64)

    if (
        out_agent_ids.shape[0] < capacity
        or out_tile_indices.shape[0] < capacity
        or out_kinds.shape[0] < capacity
        or out_amounts.shape[0] < capacity
    ):
        raise ValueError("resource perception output buffers are too small")

    water_grid: NDArray[np.float64] = np.asarray(world.water, dtype=np.float64)
    food_grid: NDArray[np.float64] = np.asarray(world.food, dtype=np.float64)

    count: int = _extract_visible_resources_kernel(
        agent_ids,
        agent_x,
        agent_y,
        agent_radii,
        agent_alive,
        world.width,
        world.height,
        water_grid,
        food_grid,
        RESOURCE_WATER_THRESHOLD,
        RESOURCE_FOOD_THRESHOLD,
        out_agent_ids,
        out_tile_indices,
        out_kinds,
        out_amounts,
    )
    return (
        out_agent_ids[:count],
        out_tile_indices[:count],
        out_kinds[:count],
        out_amounts[:count],
    )


@dataclass(slots=True)
class Observation:
    """The agent's local view of the world for one tick."""

    visible_water: list[ResourceSighting] = field(default_factory=list)
    visible_food: list[ResourceSighting] = field(default_factory=list)
    discoverables: list[DiscoverableObservation] = field(default_factory=list)
    is_daylight: bool = True
    is_night: bool = False
    is_raining: bool = False
    temperature_c: float = 18.0
    feels_cold: bool = False
    is_sheltered: bool = False
    visible_water_indices: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    visible_water_amounts: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    visible_food_indices: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    visible_food_amounts: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    world_width: int = 0

    def all_sightings(self) -> list[ResourceSighting]:
        """Return legacy ResourceSighting objects for non-hot compatibility paths."""

        if self.visible_water or self.visible_food:
            return [*self.visible_water, *self.visible_food]
        if self.world_width <= 0:
            return []

        sightings: list[ResourceSighting] = []
        for i in range(self.visible_water_indices.shape[0]):
            tile_index: int = int(self.visible_water_indices[i])
            sightings.append(
                ResourceSighting(
                    position=Position(
                        x=tile_index % self.world_width,
                        y=tile_index // self.world_width,
                    ),
                    kind=ResourceKind.WATER,
                    amount=float(self.visible_water_amounts[i]),
                )
            )
        for i in range(self.visible_food_indices.shape[0]):
            tile_index = int(self.visible_food_indices[i])
            sightings.append(
                ResourceSighting(
                    position=Position(
                        x=tile_index % self.world_width,
                        y=tile_index // self.world_width,
                    ),
                    kind=ResourceKind.FOOD,
                    amount=float(self.visible_food_amounts[i]),
                )
            )
        return sightings


def perceive(
    world: World,
    position: Position,
    clock: SimClock,
    config: SimConfig,
    weather: WeatherState | None = None,
    is_sheltered: bool | None = None,
) -> Observation:
    """Build the agent observation for one tick without hot-loop sighting objects.

    Omitted weather or shelter context is a legacy test convenience. Production
    engine and GOAP paths pass both values explicitly so observations reflect
    live simulation state.
    """

    if weather is None:
        weather = make_weather_state(
            is_raining=False,
            is_night=clock.is_night,
            config=config,
        )
    sheltered: bool = False
    if is_sheltered is None:
        item = discoverable_at_or_adjacent(world, position.x, position.y)
        sheltered = item is not None and item.kind is DiscoverableKind.CAVE
    else:
        sheltered = is_sheltered

    radius: int = (
        config.vision_radius_day if clock.is_daylight else config.vision_radius_night
    )
    min_x: int = max(0, position.x - radius)
    max_x: int = min(world.width - 1, position.x + radius)
    min_y: int = max(0, position.y - radius)
    max_y: int = min(world.height - 1, position.y + radius)
    xs: NDArray[np.int64] = np.arange(min_x, max_x + 1, dtype=np.int64)
    ys: NDArray[np.int64] = np.arange(min_y, max_y + 1, dtype=np.int64)
    grid_xs, grid_ys = np.meshgrid(xs, ys)
    dx: NDArray[np.int64] = grid_xs - position.x
    dy: NDArray[np.int64] = grid_ys - position.y
    visible_mask: NDArray[np.bool_] = dx * dx + dy * dy <= radius * radius
    tile_indices: NDArray[np.int64] = (grid_ys * world.width + grid_xs)[visible_mask]
    water_values: NDArray[np.float64] = np.asarray(world.water, dtype=np.float64)[
        tile_indices
    ]
    food_values: NDArray[np.float64] = np.asarray(world.food, dtype=np.float64)[
        tile_indices
    ]
    water_mask: NDArray[np.bool_] = water_values >= RESOURCE_WATER_THRESHOLD
    food_mask: NDArray[np.bool_] = food_values >= RESOURCE_FOOD_THRESHOLD
    disc_obs: list[DiscoverableObservation] = perceive_discoverables(
        world,
        position.x,
        position.y,
        radius,
    )
    return Observation(
        discoverables=disc_obs,
        is_daylight=clock.is_daylight,
        is_night=clock.is_night,
        is_raining=weather.is_raining,
        temperature_c=weather.temperature_c,
        feels_cold=weather.feels_cold,
        is_sheltered=sheltered,
        visible_water_indices=tile_indices[water_mask],
        visible_water_amounts=water_values[water_mask],
        visible_food_indices=tile_indices[food_mask],
        visible_food_amounts=food_values[food_mask],
        world_width=world.width,
    )
