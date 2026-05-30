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
from village_sim.world.grid import iter_positions_in_radius
from village_sim.world.weather import WeatherState, make_weather_state
from village_sim.world.world import World

RESOURCE_KIND_WATER: int = 0
RESOURCE_KIND_FOOD: int = 1


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
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.int32], NDArray[np.float64]
]:
    """High-throughput Numba kernel for resource perception across all live agents."""
    max_sightings: int = agent_ids.shape[0] * 1024
    out_agent_ids: NDArray[np.int64] = np.empty(max_sightings, dtype=np.int64)
    out_tile_indices: NDArray[np.int64] = np.empty(max_sightings, dtype=np.int64)
    out_kinds: NDArray[np.int32] = np.empty(max_sightings, dtype=np.int32)
    out_amounts: NDArray[np.float64] = np.empty(max_sightings, dtype=np.float64)

    count: int = 0

    for i in range(agent_ids.shape[0]):
        if not agent_alive[i]:
            continue

        ax: int = agent_x[i]
        ay: int = agent_y[i]
        r: int = agent_radii[i]
        aid: int = agent_ids[i]

        min_y: int = max(0, ay - r)
        max_y: int = min(world_height - 1, ay + r)
        for dy in range(min_y, max_y + 1):
            y_dist: int = abs(dy - ay)
            x_radius: int = r - y_dist
            min_x: int = max(0, ax - x_radius)
            max_x: int = min(world_width - 1, ax + x_radius)

            for dx in range(min_x, max_x + 1):
                idx: int = dy * world_width + dx

                w_amt: float = water_grid[idx]
                if w_amt >= water_threshold:
                    if count < max_sightings:
                        out_agent_ids[count] = aid
                        out_tile_indices[count] = idx
                        out_kinds[count] = RESOURCE_KIND_WATER
                        out_amounts[count] = w_amt
                        count += 1

                f_amt: float = food_grid[idx]
                if f_amt >= food_threshold:
                    if count < max_sightings:
                        out_agent_ids[count] = aid
                        out_tile_indices[count] = idx
                        out_kinds[count] = RESOURCE_KIND_FOOD
                        out_amounts[count] = f_amt
                        count += 1

    return (
        out_agent_ids[:count],
        out_tile_indices[:count],
        out_kinds[:count],
        out_amounts[:count],
    )


def perceive_batch_resources(
    agent_ids: NDArray[np.int64],
    agent_x: NDArray[np.int32],
    agent_y: NDArray[np.int32],
    agent_alive: NDArray[np.bool_],
    world: World,
    clock: SimClock,
    config: SimConfig,
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.int32], NDArray[np.float64]
]:
    """Batched perception entrypoint. Emits flat arrays of visible resources."""
    radius: int = (
        config.vision_radius_day if clock.is_daylight else config.vision_radius_night
    )
    agent_radii: NDArray[np.int32] = np.full(agent_ids.shape, radius, dtype=np.int32)

    water_grid: NDArray[np.float64] = np.asarray(world.water, dtype=np.float64)
    food_grid: NDArray[np.float64] = np.asarray(world.food, dtype=np.float64)

    return _extract_visible_resources_kernel(
        agent_ids=agent_ids,
        agent_x=agent_x,
        agent_y=agent_y,
        agent_radii=agent_radii,
        agent_alive=agent_alive,
        world_width=world.width,
        world_height=world.height,
        water_grid=water_grid,
        food_grid=food_grid,
        water_threshold=0.25,
        food_threshold=0.18,
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

    def all_sightings(self) -> list[ResourceSighting]:
        return [*self.visible_water, *self.visible_food]


def perceive(
    world: World,
    position: Position,
    clock: SimClock,
    config: SimConfig,
    weather: WeatherState | None = None,
    is_sheltered: bool | None = None,
) -> Observation:
    """Build the agent observation for one tick.

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
    visible_water: list[ResourceSighting] = []
    visible_food: list[ResourceSighting] = []
    for seen_position in iter_positions_in_radius(
        world.width, world.height, position, radius
    ):
        water_amount: float = world.water_at(seen_position)
        if water_amount >= 0.25:
            visible_water.append(
                ResourceSighting(
                    position=seen_position,
                    kind=ResourceKind.WATER,
                    amount=water_amount,
                )
            )
        food_amount: float = world.food_at(seen_position)
        if food_amount >= 0.18:
            visible_food.append(
                ResourceSighting(
                    position=seen_position,
                    kind=ResourceKind.FOOD,
                    amount=food_amount,
                )
            )
    disc_obs = perceive_discoverables(world, position.x, position.y, radius)
    return Observation(
        visible_water=visible_water,
        visible_food=visible_food,
        discoverables=disc_obs,
        is_daylight=clock.is_daylight,
        is_night=clock.is_night,
        is_raining=weather.is_raining,
        temperature_c=weather.temperature_c,
        feels_cold=weather.feels_cold,
        is_sheltered=sheltered,
    )
