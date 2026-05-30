"""Sampled rain, active water flow, flooding, and soil saturation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum

from village_sim.core.config import SimConfig
from village_sim.core.types import TerrainKind


class RainKind(StrEnum):
    """Storm intensity class."""

    DRIZZLE = "drizzle"
    RAIN = "rain"
    DOWNPOUR = "downpour"


@dataclass(frozen=True, slots=True)
class RainProfile:
    """Rain behavior for one active storm class."""

    kind: RainKind
    application_interval_ticks: int
    sample_fraction: float
    rain_min: float
    rain_max: float
    puddle_rate: float
    catchment_gain: float
    min_duration_ticks: int
    max_duration_ticks: int


@dataclass(slots=True)
class RainEvent:
    """Mutable active rain event."""

    profile: RainProfile
    ticks_remaining: int

    @property
    def is_raining(self) -> bool:
        return self.ticks_remaining > 0

    def should_apply(self, tick: int) -> bool:
        interval: int = max(1, self.profile.application_interval_ticks)
        return self.ticks_remaining > 0 and tick % interval == 0


@dataclass(slots=True)
class RainSystem:
    """Weather-owned rain event state."""

    active_event: RainEvent | None = None

    def tick(self, rng: random.Random, config: SimConfig) -> RainEvent | None:
        """Advance storm lifecycle and return the active event, if any."""

        if self.active_event is not None:
            self.active_event.ticks_remaining -= 1
            if self.active_event.ticks_remaining <= 0:
                self.active_event = None
            return self.active_event

        if rng.random() >= config.rain_event_start_chance_per_tick:
            return None

        profile: RainProfile = choose_rain_profile(rng, config)
        duration: int = rng.randint(
            profile.min_duration_ticks,
            profile.max_duration_ticks,
        )
        self.active_event = RainEvent(profile=profile, ticks_remaining=duration)
        return self.active_event


def _new_active_set() -> set[int]:
    return set()


@dataclass(slots=True)
class WaterSystemState:
    """Persistent hydrology state stored as flat row-major arrays."""

    wetness: list[float]
    soil_water: list[float]
    soil_capacity: list[float]
    soil_drainage: list[float]
    infiltration: list[float]
    runoff: list[float]
    base_water: list[float]
    bankfull_level: list[float]
    catchment: list[bool]
    permanent_water: list[bool]
    outdoor_indices: list[int]
    bank_indices: list[list[int]]
    neighbors4: list[tuple[int, ...]]
    fluid_active: set[int] = field(default_factory=_new_active_set)
    flood_active: set[int] = field(default_factory=_new_active_set)


def choose_rain_profile(rng: random.Random, config: SimConfig) -> RainProfile:
    """Choose a configured rain profile from weighted storm kinds."""

    roll: float = rng.random()
    if roll < config.drizzle_event_weight:
        return RainProfile(
            kind=RainKind.DRIZZLE,
            application_interval_ticks=5,
            sample_fraction=config.drizzle_sample_fraction,
            rain_min=0.10,
            rain_max=0.30,
            puddle_rate=0.10,
            catchment_gain=1.00,
            min_duration_ticks=24,
            max_duration_ticks=96,
        )
    if roll < config.drizzle_event_weight + config.downpour_event_weight:
        return RainProfile(
            kind=RainKind.DOWNPOUR,
            application_interval_ticks=1,
            sample_fraction=config.downpour_sample_fraction,
            rain_min=0.50,
            rain_max=0.80,
            puddle_rate=0.85,
            catchment_gain=1.00,
            min_duration_ticks=8,
            max_duration_ticks=32,
        )
    return RainProfile(
        kind=RainKind.RAIN,
        application_interval_ticks=2,
        sample_fraction=config.rain_sample_fraction,
        rain_min=0.25,
        rain_max=0.55,
        puddle_rate=0.45,
        catchment_gain=1.00,
        min_duration_ticks=18,
        max_duration_ticks=72,
    )


def build_water_system_state(
    *,
    width: int,
    height: int,
    terrain: list[int],
    water: list[float],
) -> WaterSystemState:
    """Build flat-array hydrology state from terrain and current water."""

    cell_count: int = width * height
    wetness: list[float] = [0.0 for _ in range(cell_count)]
    soil_water: list[float] = [0.0 for _ in range(cell_count)]
    soil_capacity: list[float] = [0.0 for _ in range(cell_count)]
    soil_drainage: list[float] = [0.0 for _ in range(cell_count)]
    infiltration: list[float] = [0.0 for _ in range(cell_count)]
    runoff: list[float] = [0.0 for _ in range(cell_count)]
    base_water: list[float] = [0.0 for _ in range(cell_count)]
    bankfull_level: list[float] = [0.0 for _ in range(cell_count)]
    catchment: list[bool] = [False for _ in range(cell_count)]
    permanent_water: list[bool] = [False for _ in range(cell_count)]
    outdoor_indices: list[int] = []
    bank_indices: list[list[int]] = [[], [], [], []]
    neighbors4: list[tuple[int, ...]] = []

    for index in range(cell_count):
        terrain_kind: TerrainKind = TerrainKind(terrain[index])
        outdoor_indices.append(index)
        bank_indices[index & 3].append(index)
        soil_capacity[index] = _soil_capacity_for(terrain_kind)
        soil_drainage[index] = _soil_drainage_for(terrain_kind)
        infiltration[index] = _infiltration_for(terrain_kind)
        runoff[index] = _runoff_for(terrain_kind)

        if terrain_kind is TerrainKind.WATER:
            permanent_water[index] = True
            catchment[index] = True
            base_water[index] = 0.65
            bankfull_level[index] = 1.25
            water[index] = max(water[index], base_water[index])
        else:
            catchment[index] = _is_non_water_catchment(terrain_kind)

        neighbors4.append(_neighbors4(width, height, index))

    return WaterSystemState(
        wetness=wetness,
        soil_water=soil_water,
        soil_capacity=soil_capacity,
        soil_drainage=soil_drainage,
        infiltration=infiltration,
        runoff=runoff,
        base_water=base_water,
        bankfull_level=bankfull_level,
        catchment=catchment,
        permanent_water=permanent_water,
        outdoor_indices=outdoor_indices,
        bank_indices=bank_indices,
        neighbors4=neighbors4,
    )


def step_sampled_rain(
    *,
    rng: random.Random,
    water: list[float],
    state: WaterSystemState,
    event: RainEvent,
    config: SimConfig,
) -> None:
    """Apply active rain to a random sample of outdoor tile indices."""

    if not event.is_raining or not state.outdoor_indices:
        return

    sample_count: int = max(
        1,
        int(float(len(state.outdoor_indices)) * event.profile.sample_fraction),
    )
    for _ in range(sample_count):
        sampled_index: int = state.outdoor_indices[
            rng.randrange(len(state.outdoor_indices))
        ]
        if state.catchment[sampled_index]:
            water[sampled_index] = min(
                config.max_surface_water_depth,
                water[sampled_index] + event.profile.catchment_gain,
            )
            state.fluid_active.add(sampled_index)
            if state.permanent_water[sampled_index]:
                state.flood_active.add(sampled_index)
            continue

        if rng.random() >= event.profile.puddle_rate:
            continue

        rain_amount: float = rng.uniform(event.profile.rain_min, event.profile.rain_max)
        apply_rain_to_tile(
            index=sampled_index,
            rain_amount=rain_amount,
            water=water,
            state=state,
            config=config,
        )


def apply_rain_to_tile(
    *,
    index: int,
    rain_amount: float,
    water: list[float],
    state: WaterSystemState,
    config: SimConfig,
) -> None:
    """Apply one sampled rainfall amount to one tile."""

    if state.permanent_water[index]:
        water[index] = min(config.max_surface_water_depth, water[index] + rain_amount)
        state.fluid_active.add(index)
        state.flood_active.add(index)
        return

    state.wetness[index] = min(1.0, state.wetness[index] + rain_amount)
    soil_room: float = max(0.0, state.soil_capacity[index] - state.soil_water[index])
    absorbed: float = min(soil_room, rain_amount * state.infiltration[index])
    state.soil_water[index] += absorbed
    runoff_amount: float = max(0.0, rain_amount - absorbed) * state.runoff[index]
    if runoff_amount <= 0.0:
        return

    water[index] = min(config.max_surface_water_depth, water[index] + runoff_amount)
    if water[index] >= config.surface_water_active_threshold:
        state.fluid_active.add(index)


def step_active_water_flow(
    *,
    water: list[float],
    height_map: list[float],
    state: WaterSystemState,
    config: SimConfig,
) -> None:
    """Process only active surface water and active flood source tiles."""

    if not state.fluid_active and not state.flood_active:
        return

    delta: dict[int, float] = {}
    next_active: set[int] = set()
    active_indices: set[int] = set(state.fluid_active)
    active_indices.update(state.flood_active)
    state.fluid_active.clear()
    state.flood_active.clear()

    for index in active_indices:
        current_water: float = water[index]

        if state.permanent_water[index]:
            if current_water > state.bankfull_level[index]:
                _spill_floodwater(
                    index=index,
                    water=water,
                    height_map=height_map,
                    state=state,
                    config=config,
                    delta=delta,
                    next_active=next_active,
                )
            else:
                _restore_base_water(index, water, state)
            continue

        movable_water: float = max(0.0, current_water - state.base_water[index])
        if movable_water < config.surface_water_active_threshold:
            _restore_base_water(index, water, state)
            continue

        moved: bool = _move_downhill_or_sideways(
            index=index,
            water=water,
            height_map=height_map,
            state=state,
            config=config,
            delta=delta,
            next_active=next_active,
        )
        if not moved:
            _infiltrate_surface_water(
                index=index,
                water=water,
                state=state,
                config=config,
                next_active=next_active,
            )

    for index, change in delta.items():
        water[index] = max(state.base_water[index], water[index] + change)
        if water[index] >= config.surface_water_active_threshold:
            next_active.add(index)
        if state.permanent_water[index] and water[index] > state.bankfull_level[index]:
            state.flood_active.add(index)

    state.fluid_active.update(
        index
        for index in next_active
        if not state.permanent_water[index]
        and water[index] >= config.surface_water_active_threshold
    )


def step_water_maintenance_bank(
    *,
    tick: int,
    water: list[float],
    state: WaterSystemState,
    config: SimConfig,
) -> None:
    """Process one quarter-map bank of slow water maintenance."""

    bank: int = tick & 3
    for index in state.bank_indices[bank]:
        if state.permanent_water[index]:
            _restore_base_water(index, water, state)
            if water[index] > state.bankfull_level[index]:
                state.flood_active.add(index)
            continue

        state.soil_water[index] = max(
            0.0,
            state.soil_water[index] - state.soil_drainage[index],
        )
        state.wetness[index] = max(0.0, state.wetness[index] - config.wetness_decay)
        if index in state.fluid_active or index in state.flood_active:
            continue

        if 0.0 < water[index] <= config.shallow_water_threshold:
            water[index] = max(0.0, water[index] - config.shallow_evaporation)


def _move_downhill_or_sideways(
    *,
    index: int,
    water: list[float],
    height_map: list[float],
    state: WaterSystemState,
    config: SimConfig,
    delta: dict[int, float],
    next_active: set[int],
) -> bool:
    current_surface: float = (
        height_map[index] + water[index] * config.water_height_scale
    )
    best_downhill: int = -1
    best_downhill_surface: float = current_surface
    for neighbor_index in state.neighbors4[index]:
        neighbor_surface: float = (
            height_map[neighbor_index]
            + water[neighbor_index] * config.water_height_scale
        )
        if (
            height_map[neighbor_index] < height_map[index]
            and neighbor_surface < best_downhill_surface
        ):
            best_downhill = neighbor_index
            best_downhill_surface = neighbor_surface

    if best_downhill >= 0:
        height_drop: float = current_surface - best_downhill_surface
        flow: float = min(
            water[index] - state.base_water[index],
            water[index]
            * config.downhill_flow_fraction
            * max(0.20, height_drop * 12.0),
        )
        if flow > 0.0:
            _add_delta(delta, index, -flow)
            _add_delta(delta, best_downhill, flow)
            next_active.add(index)
            next_active.add(best_downhill)
            return True

    best_spread: int = -1
    best_depth: float = water[index]
    for neighbor_index in state.neighbors4[index]:
        if height_map[neighbor_index] > height_map[index] + config.sideways_height_slop:
            continue
        if water[neighbor_index] < best_depth:
            best_spread = neighbor_index
            best_depth = water[neighbor_index]

    if best_spread < 0:
        return False

    depth_difference: float = water[index] - water[best_spread]
    if depth_difference <= config.sideways_min_depth_difference:
        return False

    spread: float = min(
        water[index] - state.base_water[index],
        depth_difference * config.sideways_spread_fraction,
    )
    if spread <= 0.0:
        return False

    _add_delta(delta, index, -spread)
    _add_delta(delta, best_spread, spread)
    next_active.add(index)
    next_active.add(best_spread)
    return True


def _spill_floodwater(
    *,
    index: int,
    water: list[float],
    height_map: list[float],
    state: WaterSystemState,
    config: SimConfig,
    delta: dict[int, float],
    next_active: set[int],
) -> None:
    excess: float = water[index] - state.bankfull_level[index]
    if excess <= 0.0:
        return

    target_index: int = -1
    target_height: float = 999_999.0
    source_height: float = height_map[index]
    for neighbor_index in state.neighbors4[index]:
        if state.permanent_water[neighbor_index]:
            continue
        neighbor_height: float = height_map[neighbor_index]
        if neighbor_height <= source_height and neighbor_height < target_height:
            target_index = neighbor_index
            target_height = neighbor_height

    if target_index < 0:
        drain: float = min(excess, config.off_map_drain_rate)
        _add_delta(delta, index, -drain)
        next_active.add(index)
        return

    spill: float = excess * config.flood_spill_fraction
    _add_delta(delta, index, -spill)
    _add_delta(delta, target_index, spill)
    next_active.add(index)
    next_active.add(target_index)


def _infiltrate_surface_water(
    *,
    index: int,
    water: list[float],
    state: WaterSystemState,
    config: SimConfig,
    next_active: set[int],
) -> None:
    if state.permanent_water[index]:
        return

    soil_room: float = max(0.0, state.soil_capacity[index] - state.soil_water[index])
    absorbed: float = min(soil_room, config.surface_infiltration_per_fluid_tick)
    if absorbed > 0.0:
        state.soil_water[index] += absorbed
        water[index] = max(0.0, water[index] - absorbed)
    if water[index] >= config.surface_water_active_threshold:
        next_active.add(index)


def _restore_base_water(
    index: int, water: list[float], state: WaterSystemState
) -> None:
    if water[index] < state.base_water[index]:
        water[index] = state.base_water[index]


def _add_delta(delta: dict[int, float], index: int, change: float) -> None:
    delta[index] = delta.get(index, 0.0) + change


def _neighbors4(width: int, height: int, index: int) -> tuple[int, ...]:
    x: int = index % width
    y: int = index // width
    neighbors: list[int] = []
    if y > 0:
        neighbors.append(index - width)
    if x > 0:
        neighbors.append(index - 1)
    if x < width - 1:
        neighbors.append(index + 1)
    if y < height - 1:
        neighbors.append(index + width)
    return tuple(neighbors)


def _soil_capacity_for(terrain_kind: TerrainKind) -> float:
    if terrain_kind is TerrainKind.WATER:
        return 0.0
    if terrain_kind is TerrainKind.ROCK:
        return 0.10
    if terrain_kind is TerrainKind.HILL:
        return 0.45
    if terrain_kind is TerrainKind.FOREST:
        return 0.90
    return 0.70


def _soil_drainage_for(terrain_kind: TerrainKind) -> float:
    if terrain_kind is TerrainKind.WATER:
        return 0.0
    if terrain_kind is TerrainKind.ROCK:
        return 0.01
    if terrain_kind is TerrainKind.HILL:
        return 0.06
    if terrain_kind is TerrainKind.FOREST:
        return 0.04
    return 0.05


def _infiltration_for(terrain_kind: TerrainKind) -> float:
    if terrain_kind is TerrainKind.WATER:
        return 0.0
    if terrain_kind is TerrainKind.ROCK:
        return 0.10
    if terrain_kind is TerrainKind.HILL:
        return 0.35
    if terrain_kind is TerrainKind.FOREST:
        return 0.75
    return 0.60


def _runoff_for(terrain_kind: TerrainKind) -> float:
    if terrain_kind is TerrainKind.WATER:
        return 1.00
    if terrain_kind is TerrainKind.ROCK:
        return 0.90
    if terrain_kind is TerrainKind.HILL:
        return 0.60
    if terrain_kind is TerrainKind.FOREST:
        return 0.20
    return 0.35


def _is_non_water_catchment(terrain_kind: TerrainKind) -> bool:
    return terrain_kind is TerrainKind.HILL
