"""Sampled rain, active water flow, flooding, and soil saturation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from village_sim.core.config import SimConfig
from village_sim.core.types import TerrainKind

FloatGrid = NDArray[np.float64]
IntGrid = NDArray[np.int64]
BoolGrid = NDArray[np.bool_]
MutableFloatGrid = list[float] | FloatGrid
TerrainGrid = list[int] | IntGrid


class ComparableFloatArray(np.ndarray):
    """Float ndarray with test-friendly scalar equality for lists."""

    def __eq__(self, other: object) -> bool:
        return bool(
            np.array_equal(np.asarray(self), np.asarray(other, dtype=np.float64))
        )


def _comparable_float(values: FloatGrid) -> ComparableFloatArray:
    return values.astype(np.float64, copy=False).view(ComparableFloatArray)


_NEIGHBOR_COUNT = 4
_NO_NEIGHBOR = -1


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


@dataclass(slots=True)
class ActiveMask:
    """Set-like active tile mask backed by one contiguous boolean array."""

    mask: BoolGrid

    def add(self, index: int) -> None:
        self.mask[index] = True

    def clear(self) -> None:
        self.mask.fill(False)

    def update(self, indices: Iterable[int] | IntGrid | BoolGrid) -> None:
        if isinstance(indices, np.ndarray) and indices.dtype == np.bool_:
            self.mask |= np.asarray(indices, dtype=np.bool_)
            return
        index_values: IntGrid = np.fromiter(indices, dtype=np.int64)
        if index_values.size > 0:
            self.mask[index_values] = True

    def to_indices(self) -> IntGrid:
        return np.flatnonzero(self.mask).astype(np.int64)

    def __contains__(self, index: object) -> bool:
        return isinstance(index, int) and bool(self.mask[index])

    def __bool__(self) -> bool:
        return bool(np.any(self.mask))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, set):
            return set(self.to_indices().tolist()) == other
        if isinstance(other, ActiveMask):
            return bool(np.array_equal(self.mask, other.mask))
        return False


def _new_active_mask(cell_count: int) -> ActiveMask:
    return ActiveMask(np.zeros(cell_count, dtype=np.bool_))


@dataclass(slots=True)
class WaterSystemState:
    """Persistent hydrology state stored as flat row-major NumPy arrays."""

    wetness: FloatGrid
    soil_water: FloatGrid
    soil_capacity: FloatGrid
    soil_drainage: FloatGrid
    infiltration: FloatGrid
    runoff: FloatGrid
    base_water: FloatGrid
    bankfull_level: FloatGrid
    catchment: BoolGrid
    permanent_water: BoolGrid
    outdoor_indices: IntGrid
    bank_indices: tuple[IntGrid, IntGrid, IntGrid, IntGrid]
    neighbors4: IntGrid
    fluid_active: ActiveMask = field(init=False)
    flood_active: ActiveMask = field(init=False)
    flow_delta: FloatGrid = field(init=False)
    next_active_buffer: BoolGrid = field(init=False)
    active_mask_buffer: BoolGrid = field(init=False)
    moved_mask_buffer: BoolGrid = field(init=False)
    changed_mask_buffer: BoolGrid = field(init=False)

    def __post_init__(self) -> None:
        self.wetness = _comparable_float(np.asarray(self.wetness, dtype=np.float64))
        self.soil_water = _comparable_float(
            np.asarray(self.soil_water, dtype=np.float64)
        )
        self.soil_capacity = _comparable_float(
            np.asarray(self.soil_capacity, dtype=np.float64)
        )
        self.soil_drainage = _comparable_float(
            np.asarray(self.soil_drainage, dtype=np.float64)
        )
        self.infiltration = _comparable_float(
            np.asarray(self.infiltration, dtype=np.float64)
        )
        self.runoff = _comparable_float(np.asarray(self.runoff, dtype=np.float64))
        self.base_water = _comparable_float(
            np.asarray(self.base_water, dtype=np.float64)
        )
        self.bankfull_level = _comparable_float(
            np.asarray(self.bankfull_level, dtype=np.float64)
        )
        cell_count: int = int(self.wetness.size)
        self.fluid_active = _new_active_mask(cell_count)
        self.flood_active = _new_active_mask(cell_count)
        self.flow_delta = np.zeros(cell_count, dtype=np.float64)
        self.next_active_buffer = np.zeros(cell_count, dtype=np.bool_)
        self.active_mask_buffer = np.zeros(cell_count, dtype=np.bool_)
        self.moved_mask_buffer = np.zeros(cell_count, dtype=np.bool_)
        self.changed_mask_buffer = np.zeros(cell_count, dtype=np.bool_)

    def __setattr__(self, name: str, value: object) -> None:
        float_fields: set[str] = {
            "wetness",
            "soil_water",
            "soil_capacity",
            "soil_drainage",
            "infiltration",
            "runoff",
            "base_water",
            "bankfull_level",
            "flow_delta",
        }
        if name in float_fields:
            array_value: ComparableFloatArray = _comparable_float(
                np.asarray(value, dtype=np.float64)
            )
            object.__setattr__(self, name, array_value)
            return
        object.__setattr__(self, name, value)


def _as_float_grid(values: MutableFloatGrid) -> FloatGrid:
    return np.asarray(values, dtype=np.float64)


def _as_int_grid(values: TerrainGrid) -> IntGrid:
    return np.asarray(values, dtype=np.int64)


def _sync_float_grid(target: MutableFloatGrid, values: FloatGrid) -> None:
    if isinstance(target, np.ndarray):
        target[...] = values
        return
    target[:] = values.astype(np.float64, copy=False).tolist()


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


def _neighbor_indices(width: int, height: int) -> IntGrid:
    indices: IntGrid = np.arange(width * height, dtype=np.int64).reshape(height, width)
    neighbors: IntGrid = np.full((width * height, _NEIGHBOR_COUNT), _NO_NEIGHBOR)
    north_rows: IntGrid = indices[1:, :].reshape(-1)
    west_rows: IntGrid = indices[:, 1:].reshape(-1)
    east_rows: IntGrid = indices[:, :-1].reshape(-1)
    south_rows: IntGrid = indices[:-1, :].reshape(-1)
    neighbors[north_rows, 0] = indices[:-1, :].reshape(-1)
    neighbors[west_rows, 1] = indices[:, :-1].reshape(-1)
    neighbors[east_rows, 2] = indices[:, 1:].reshape(-1)
    neighbors[south_rows, 3] = indices[1:, :].reshape(-1)
    return neighbors


def build_water_system_state(
    *,
    width: int,
    height: int,
    terrain: TerrainGrid,
    water: MutableFloatGrid,
) -> WaterSystemState:
    """Build flat-array hydrology state from terrain and current water."""

    cell_count: int = width * height
    terrain_values: IntGrid = _as_int_grid(terrain).reshape(cell_count)
    wetness: FloatGrid = np.zeros(cell_count, dtype=np.float64)
    soil_water: FloatGrid = np.zeros(cell_count, dtype=np.float64)
    soil_capacity: FloatGrid = np.full(cell_count, 0.70, dtype=np.float64)
    soil_drainage: FloatGrid = np.full(cell_count, 0.05, dtype=np.float64)
    infiltration: FloatGrid = np.full(cell_count, 0.60, dtype=np.float64)
    runoff: FloatGrid = np.full(cell_count, 0.35, dtype=np.float64)
    base_water: FloatGrid = np.zeros(cell_count, dtype=np.float64)
    bankfull_level: FloatGrid = np.zeros(cell_count, dtype=np.float64)

    water_mask: BoolGrid = terrain_values == int(TerrainKind.WATER)
    rock_mask: BoolGrid = terrain_values == int(TerrainKind.ROCK)
    hill_mask: BoolGrid = terrain_values == int(TerrainKind.HILL)
    forest_mask: BoolGrid = terrain_values == int(TerrainKind.FOREST)

    soil_capacity[water_mask] = 0.0
    soil_capacity[rock_mask] = 0.10
    soil_capacity[hill_mask] = 0.45
    soil_capacity[forest_mask] = 0.90

    soil_drainage[water_mask] = 0.0
    soil_drainage[rock_mask] = 0.01
    soil_drainage[hill_mask] = 0.06
    soil_drainage[forest_mask] = 0.04

    infiltration[water_mask] = 0.0
    infiltration[rock_mask] = 0.10
    infiltration[hill_mask] = 0.35
    infiltration[forest_mask] = 0.75

    runoff[water_mask] = 1.00
    runoff[rock_mask] = 0.90
    runoff[hill_mask] = 0.60
    runoff[forest_mask] = 0.20

    catchment: BoolGrid = water_mask | hill_mask
    base_water[water_mask] = 0.65
    bankfull_level[water_mask] = 1.25
    water_values: FloatGrid = _as_float_grid(water).reshape(cell_count).copy()
    water_values[water_mask] = np.maximum(
        water_values[water_mask], base_water[water_mask]
    )
    _sync_float_grid(water, water_values)

    indices: IntGrid = np.arange(cell_count, dtype=np.int64)
    bank_indices: tuple[IntGrid, IntGrid, IntGrid, IntGrid] = (
        indices[(indices & 3) == 0],
        indices[(indices & 3) == 1],
        indices[(indices & 3) == 2],
        indices[(indices & 3) == 3],
    )

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
        permanent_water=water_mask.copy(),
        outdoor_indices=indices,
        bank_indices=bank_indices,
        neighbors4=_neighbor_indices(width, height),
    )


def step_sampled_rain(
    *,
    rng: random.Random,
    water: MutableFloatGrid,
    state: WaterSystemState,
    event: RainEvent,
    config: SimConfig,
) -> None:
    """Apply active rain to a random sample of outdoor tile indices."""

    if not event.is_raining or state.outdoor_indices.size == 0:
        return

    sample_count: int = max(
        1,
        int(float(state.outdoor_indices.size) * event.profile.sample_fraction),
    )
    sampled_indices: IntGrid = np.asarray(
        rng.sample(
            state.outdoor_indices.tolist(),
            min(int(state.outdoor_indices.size), sample_count),
        ),
        dtype=np.int64,
    )
    if sampled_indices.size == 0:
        return

    water_values: FloatGrid = _as_float_grid(water).reshape(state.wetness.size).copy()
    catchment_mask: BoolGrid = state.catchment[sampled_indices]
    catchment_indices: IntGrid = sampled_indices[catchment_mask]
    if catchment_indices.size > 0:
        water_values[catchment_indices] = np.minimum(
            config.max_surface_water_depth,
            water_values[catchment_indices] + event.profile.catchment_gain,
        )
        state.fluid_active.update(catchment_indices)
        state.flood_active.update(
            catchment_indices[state.permanent_water[catchment_indices]]
        )

    puddle_candidates: IntGrid = sampled_indices[~catchment_mask]
    if puddle_candidates.size > 0:
        puddle_indices_list: list[int] = []
        rain_amounts_list: list[float] = []
        for candidate in puddle_candidates.tolist():
            if rng.random() < event.profile.puddle_rate:
                puddle_indices_list.append(int(candidate))
                rain_amounts_list.append(
                    rng.uniform(event.profile.rain_min, event.profile.rain_max)
                )
        if puddle_indices_list:
            _apply_rain_to_indices(
                indices=np.asarray(puddle_indices_list, dtype=np.int64),
                rain_amounts=np.asarray(rain_amounts_list, dtype=np.float64),
                water_values=water_values,
                state=state,
                config=config,
            )

    _sync_float_grid(water, water_values)


def apply_rain_to_tile(
    *,
    index: int,
    rain_amount: float,
    water: MutableFloatGrid,
    state: WaterSystemState,
    config: SimConfig,
) -> None:
    """Apply one sampled rainfall amount to one tile."""

    water_values: FloatGrid = _as_float_grid(water).reshape(state.wetness.size).copy()
    _apply_rain_to_indices(
        indices=np.asarray([index], dtype=np.int64),
        rain_amounts=np.asarray([rain_amount], dtype=np.float64),
        water_values=water_values,
        state=state,
        config=config,
    )
    _sync_float_grid(water, water_values)


def _apply_rain_to_indices(
    *,
    indices: IntGrid,
    rain_amounts: FloatGrid,
    water_values: FloatGrid,
    state: WaterSystemState,
    config: SimConfig,
) -> None:
    permanent_mask: BoolGrid = state.permanent_water[indices]
    permanent_indices: IntGrid = indices[permanent_mask]
    if permanent_indices.size > 0:
        permanent_amounts: FloatGrid = rain_amounts[permanent_mask]
        water_values[permanent_indices] = np.minimum(
            config.max_surface_water_depth,
            water_values[permanent_indices] + permanent_amounts,
        )
        state.fluid_active.update(permanent_indices)
        state.flood_active.update(permanent_indices)

    soil_indices: IntGrid = indices[~permanent_mask]
    if soil_indices.size == 0:
        return
    soil_amounts: FloatGrid = rain_amounts[~permanent_mask]
    state.wetness[soil_indices] = np.minimum(
        1.0, state.wetness[soil_indices] + soil_amounts
    )
    soil_room: FloatGrid = np.maximum(
        0.0, state.soil_capacity[soil_indices] - state.soil_water[soil_indices]
    )
    absorbed: FloatGrid = np.minimum(
        soil_room, soil_amounts * state.infiltration[soil_indices]
    )
    state.soil_water[soil_indices] += absorbed
    runoff_amount: FloatGrid = (
        np.maximum(0.0, soil_amounts - absorbed) * state.runoff[soil_indices]
    )
    runoff_mask: BoolGrid = runoff_amount > 0.0
    runoff_indices: IntGrid = soil_indices[runoff_mask]
    if runoff_indices.size == 0:
        return
    water_values[runoff_indices] = np.minimum(
        config.max_surface_water_depth,
        water_values[runoff_indices] + runoff_amount[runoff_mask],
    )
    active_mask: BoolGrid = (
        water_values[runoff_indices] >= config.surface_water_active_threshold
    )
    state.fluid_active.update(runoff_indices[active_mask])


def step_active_water_flow(
    *,
    water: MutableFloatGrid,
    height_map: MutableFloatGrid,
    state: WaterSystemState,
    config: SimConfig,
) -> None:
    """Process active surface water and active flood source tiles."""

    if not state.fluid_active and not state.flood_active:
        return

    water_values: FloatGrid = _as_float_grid(water).reshape(state.wetness.size).copy()
    heights: FloatGrid = _as_float_grid(height_map).reshape(state.wetness.size)
    active_mask: BoolGrid = state.active_mask_buffer
    active_mask[...] = state.fluid_active.mask
    active_mask |= state.flood_active.mask
    state.fluid_active.clear()
    state.flood_active.clear()

    delta: FloatGrid = state.flow_delta
    next_active: BoolGrid = state.next_active_buffer
    moved_mask: BoolGrid = state.moved_mask_buffer
    changed: BoolGrid = state.changed_mask_buffer
    delta.fill(0.0)
    next_active.fill(False)
    moved_mask.fill(False)
    changed.fill(False)

    _spill_floodwater_numba(
        active_mask,
        water_values,
        heights,
        state.base_water,
        state.bankfull_level,
        state.permanent_water,
        state.neighbors4,
        config.off_map_drain_rate,
        config.flood_spill_fraction,
        delta,
        next_active,
    )
    _move_downhill_or_sideways_numba(
        active_mask,
        water_values,
        heights,
        state.base_water,
        state.permanent_water,
        state.neighbors4,
        config.surface_water_active_threshold,
        config.water_height_scale,
        config.downhill_flow_fraction,
        config.sideways_height_slop,
        config.sideways_min_depth_difference,
        config.sideways_spread_fraction,
        delta,
        next_active,
        moved_mask,
    )
    _infiltrate_surface_water_numba(
        active_mask,
        moved_mask,
        water_values,
        state.soil_water,
        state.soil_capacity,
        state.base_water,
        state.permanent_water,
        config.surface_water_active_threshold,
        config.surface_infiltration_per_fluid_tick,
        next_active,
    )
    _commit_water_flow_numba(
        water_values,
        state.base_water,
        state.bankfull_level,
        state.permanent_water,
        active_mask,
        delta,
        config.surface_water_active_threshold,
        changed,
        next_active,
        state.flood_active.mask,
        state.fluid_active.mask,
    )
    _sync_float_grid(water, water_values)


@njit(cache=True, fastmath=True)
def _spill_floodwater_numba(
    active_mask: BoolGrid,
    water_values: FloatGrid,
    heights: FloatGrid,
    base_water: FloatGrid,
    bankfull_level: FloatGrid,
    permanent_water: BoolGrid,
    neighbors4: IntGrid,
    off_map_drain_rate: float,
    flood_spill_fraction: float,
    delta: FloatGrid,
    next_active: BoolGrid,
) -> None:
    cell_count: int = active_mask.size
    for source in range(cell_count):
        if not active_mask[source] or not permanent_water[source]:
            continue
        excess: float = water_values[source] - bankfull_level[source]
        if excess <= 0.0:
            if water_values[source] <= bankfull_level[source]:
                restored: float = base_water[source]
                if water_values[source] < restored:
                    water_values[source] = restored
            continue

        best_target: int = _NO_NEIGHBOR
        best_height: float = np.inf
        source_height: float = heights[source]
        for slot in range(_NEIGHBOR_COUNT):
            target: int = neighbors4[source, slot]
            if target < 0 or permanent_water[target]:
                continue
            target_height: float = heights[target]
            if target_height <= source_height and target_height < best_height:
                best_height = target_height
                best_target = target

        if best_target < 0:
            drain: float = excess
            if drain > off_map_drain_rate:
                drain = off_map_drain_rate
            if drain > 0.0:
                delta[source] -= drain
                next_active[source] = True
            continue

        spill: float = excess * flood_spill_fraction
        if spill > 0.0:
            delta[source] -= spill
            delta[best_target] += spill
            next_active[source] = True
            next_active[best_target] = True


@njit(cache=True, fastmath=True)
def _move_downhill_or_sideways_numba(
    active_mask: BoolGrid,
    water_values: FloatGrid,
    heights: FloatGrid,
    base_water: FloatGrid,
    permanent_water: BoolGrid,
    neighbors4: IntGrid,
    active_threshold: float,
    water_height_scale: float,
    downhill_flow_fraction: float,
    sideways_height_slop: float,
    sideways_min_depth_difference: float,
    sideways_spread_fraction: float,
    delta: FloatGrid,
    next_active: BoolGrid,
    moved_mask: BoolGrid,
) -> None:
    cell_count: int = active_mask.size
    for source in range(cell_count):
        if not active_mask[source] or permanent_water[source]:
            continue
        available: float = water_values[source] - base_water[source]
        if available < active_threshold:
            continue

        source_surface: float = (
            heights[source] + water_values[source] * water_height_scale
        )
        best_downhill_target: int = _NO_NEIGHBOR
        best_downhill_surface: float = np.inf
        source_height: float = heights[source]
        for slot in range(_NEIGHBOR_COUNT):
            target: int = neighbors4[source, slot]
            if target < 0 or heights[target] >= source_height:
                continue
            target_surface: float = (
                heights[target] + water_values[target] * water_height_scale
            )
            if target_surface < best_downhill_surface:
                best_downhill_surface = target_surface
                best_downhill_target = target

        if best_downhill_target >= 0 and best_downhill_surface < source_surface:
            height_drop: float = source_surface - best_downhill_surface
            flow_scale: float = height_drop * 12.0
            if flow_scale < 0.20:
                flow_scale = 0.20
            flow: float = water_values[source] * downhill_flow_fraction * flow_scale
            if flow > available:
                flow = available
            if flow > 0.0:
                delta[source] -= flow
                delta[best_downhill_target] += flow
                next_active[source] = True
                next_active[best_downhill_target] = True
                moved_mask[source] = True
                continue

        allowed_height_limit: float = source_height + sideways_height_slop
        best_sideways_target: int = _NO_NEIGHBOR
        best_sideways_surface: float = np.inf
        for slot in range(_NEIGHBOR_COUNT):
            target = neighbors4[source, slot]
            if target < 0 or heights[target] > allowed_height_limit:
                continue
            target_surface = heights[target] + water_values[target] * water_height_scale
            if target_surface < best_sideways_surface:
                best_sideways_surface = target_surface
                best_sideways_target = target

        if best_sideways_target < 0 or best_sideways_surface >= source_surface:
            continue
        depth_difference: float = (
            water_values[source] - water_values[best_sideways_target]
        )
        if depth_difference <= sideways_min_depth_difference:
            continue
        spread: float = depth_difference * sideways_spread_fraction
        if spread > available:
            spread = available
        if spread > 0.0:
            delta[source] -= spread
            delta[best_sideways_target] += spread
            next_active[source] = True
            next_active[best_sideways_target] = True
            moved_mask[source] = True


@njit(cache=True, fastmath=True)
def _infiltrate_surface_water_numba(
    active_mask: BoolGrid,
    moved_mask: BoolGrid,
    water_values: FloatGrid,
    soil_water: FloatGrid,
    soil_capacity: FloatGrid,
    base_water: FloatGrid,
    permanent_water: BoolGrid,
    active_threshold: float,
    surface_infiltration_per_fluid_tick: float,
    next_active: BoolGrid,
) -> None:
    cell_count: int = active_mask.size
    for index in range(cell_count):
        if not active_mask[index] or moved_mask[index] or permanent_water[index]:
            continue
        available: float = water_values[index] - base_water[index]
        if available < active_threshold:
            continue
        soil_room: float = soil_capacity[index] - soil_water[index]
        if soil_room < 0.0:
            soil_room = 0.0
        absorbed: float = surface_infiltration_per_fluid_tick
        if absorbed > soil_room:
            absorbed = soil_room
        if absorbed > water_values[index]:
            absorbed = water_values[index]
        if absorbed > 0.0:
            soil_water[index] += absorbed
            water_values[index] -= absorbed
            if water_values[index] < 0.0:
                water_values[index] = 0.0
        if water_values[index] >= active_threshold:
            next_active[index] = True


@njit(cache=True, fastmath=True)
def _commit_water_flow_numba(
    water_values: FloatGrid,
    base_water: FloatGrid,
    bankfull_level: FloatGrid,
    permanent_water: BoolGrid,
    active_mask: BoolGrid,
    delta: FloatGrid,
    active_threshold: float,
    changed: BoolGrid,
    next_active: BoolGrid,
    flood_active: BoolGrid,
    fluid_active: BoolGrid,
) -> None:
    cell_count: int = water_values.size
    for index in range(cell_count):
        delta_value: float = delta[index]
        changed[index] = delta_value != 0.0
        if changed[index]:
            updated: float = water_values[index] + delta_value
            if updated < base_water[index]:
                updated = base_water[index]
            water_values[index] = updated
        if next_active[index] or (
            changed[index] and water_values[index] >= active_threshold
        ):
            next_active[index] = water_values[index] >= active_threshold
        if permanent_water[index]:
            flood_active[index] = (
                changed[index] and water_values[index] > bankfull_level[index]
            )
            continue
        fluid_active[index] = (
            next_active[index] and water_values[index] >= active_threshold
        )


def step_water_maintenance_bank(
    *,
    tick: int,
    water: MutableFloatGrid,
    state: WaterSystemState,
    config: SimConfig,
) -> None:
    """Process one quarter-map bank of slow water maintenance."""

    bank: int = tick & 3
    indices: IntGrid = state.bank_indices[bank]
    if indices.size == 0:
        return
    water_values: FloatGrid = _as_float_grid(water).reshape(state.wetness.size).copy()
    permanent_indices: IntGrid = indices[state.permanent_water[indices]]
    if permanent_indices.size > 0:
        water_values[permanent_indices] = np.maximum(
            water_values[permanent_indices], state.base_water[permanent_indices]
        )
        state.flood_active.update(
            permanent_indices[
                water_values[permanent_indices]
                > state.bankfull_level[permanent_indices]
            ]
        )

    soil_indices: IntGrid = indices[~state.permanent_water[indices]]
    if soil_indices.size > 0:
        state.soil_water[soil_indices] = np.maximum(
            0.0,
            state.soil_water[soil_indices] - state.soil_drainage[soil_indices],
        )
        state.wetness[soil_indices] = np.maximum(
            0.0, state.wetness[soil_indices] - config.wetness_decay
        )
        inactive_mask: BoolGrid = (
            ~state.fluid_active.mask[soil_indices]
            & ~state.flood_active.mask[soil_indices]
        )
        evaporation_indices: IntGrid = soil_indices[
            inactive_mask
            & (water_values[soil_indices] > 0.0)
            & (water_values[soil_indices] <= config.shallow_water_threshold)
        ]
        if evaporation_indices.size > 0:
            water_values[evaporation_indices] = np.maximum(
                0.0, water_values[evaporation_indices] - config.shallow_evaporation
            )
    _sync_float_grid(water, water_values)
