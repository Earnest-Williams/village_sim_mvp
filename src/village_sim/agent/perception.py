"""Agent perception."""

from __future__ import annotations

from dataclasses import dataclass, field

from village_sim.core.config import SimConfig
from village_sim.core.time import SimClock
from village_sim.core.types import Position, ResourceKind, ResourceSighting
from village_sim.world.discoverables import DiscoverableObservation, perceive_discoverables
from village_sim.world.grid import iter_positions_in_radius
from village_sim.world.world import World


@dataclass(slots=True)
class Observation:
    """The agent's local view of the world for one tick."""

    visible_water: list[ResourceSighting] = field(default_factory=list)
    visible_food: list[ResourceSighting] = field(default_factory=list)
    discoverables: list[DiscoverableObservation] = field(default_factory=list)
    is_daylight: bool = True
    is_night: bool = False

    def all_sightings(self) -> list[ResourceSighting]:
        return [*self.visible_water, *self.visible_food]


def perceive(
    world: World,
    position: Position,
    clock: SimClock,
    config: SimConfig,
) -> Observation:
    radius: int = config.vision_radius_day if clock.is_daylight else config.vision_radius_night
    visible_water: list[ResourceSighting] = []
    visible_food: list[ResourceSighting] = []
    for seen_position in iter_positions_in_radius(world.width, world.height, position, radius):
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
    )
