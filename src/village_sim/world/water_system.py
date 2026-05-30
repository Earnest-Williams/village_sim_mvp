"""Sampled rain, active water flow, flooding, and soil saturation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from village_sim.core.config import SimConfig
from village_sim.core.types import TerrainKind

FloatGrid = NDArray[np.float64]
IntGrid = NDArray[np.int64]
BoolGrid = NDArray[np.bool_]
MutableFloatGrid = list[float] | FloatGrid
TerrainGrid = list[int] | IntGrid


class ComparableFloatArray(np.ndarray[Any, np.dtype[np.float64]]):
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
    active_mask: BoolGrid = state.fluid_active.mask | state.flood_active.mask
    state.fluid_active.clear()
    state.flood_active.clear()

    permanent_active: BoolGrid = active_mask & state.permanent_water
    flood_sources: IntGrid = np.flatnonzero(
        permanent_active & (water_values > state.bankfull_level)
    ).astype(np.int64)
    next_active: BoolGrid = np.zeros_like(active_mask)
    delta: FloatGrid = np.zeros_like(water_values)
    if flood_sources.size > 0:
        _spill_floodwater_vectorized(
            indices=flood_sources,
            water_values=water_values,
            heights=heights,
            state=state,
            config=config,
            delta=delta,
            next_active=next_active,
        )
    restore_permanent: BoolGrid = permanent_active & (
        water_values <= state.bankfull_level
    )
    water_values[restore_permanent] = np.maximum(
        water_values[restore_permanent], state.base_water[restore_permanent]
    )

    movable_mask: BoolGrid = (
        active_mask
        & ~state.permanent_water
        & ((water_values - state.base_water) >= config.surface_water_active_threshold)
    )
    movable_indices: IntGrid = np.flatnonzero(movable_mask).astype(np.int64)
    moved_mask: BoolGrid = np.zeros_like(active_mask)
    if movable_indices.size > 0:
        moved_indices: IntGrid = _move_downhill_or_sideways_vectorized(
            indices=movable_indices,
            water_values=water_values,
            heights=heights,
            state=state,
            config=config,
            delta=delta,
            next_active=next_active,
        )
        moved_mask[moved_indices] = True

    still_indices: IntGrid = movable_indices[~moved_mask[movable_indices]]
    if still_indices.size > 0:
        _infiltrate_surface_water_vectorized(
            indices=still_indices,
            water_values=water_values,
            state=state,
            config=config,
            next_active=next_active,
        )

    changed: BoolGrid = delta != 0.0
    water_values[changed] = np.maximum(
        state.base_water[changed], water_values[changed] + delta[changed]
    )
    next_active |= changed & (water_values >= config.surface_water_active_threshold)
    state.flood_active.update(
        state.permanent_water & (water_values > state.bankfull_level) & changed
    )
    state.fluid_active.update(
        next_active
        & ~state.permanent_water
        & (water_values >= config.surface_water_active_threshold)
    )
    _sync_float_grid(water, water_values)


def _move_downhill_or_sideways_vectorized(
    *,
    indices: IntGrid,
    water_values: FloatGrid,
    heights: FloatGrid,
    state: WaterSystemState,
    config: SimConfig,
    delta: FloatGrid,
    next_active: BoolGrid,
) -> IntGrid:
    neighbors: IntGrid = state.neighbors4[indices]
    valid: BoolGrid = neighbors >= 0
    safe_neighbors: IntGrid = np.where(valid, neighbors, 0)
    surface: FloatGrid = heights + water_values * config.water_height_scale
    neighbor_surface: FloatGrid = np.where(valid, surface[safe_neighbors], np.inf)
    lower_height: BoolGrid = np.where(
        valid, heights[safe_neighbors] < heights[indices, None], False
    )
    downhill_surface: FloatGrid = np.where(lower_height, neighbor_surface, np.inf)
    best_downhill_slot: IntGrid = np.argmin(downhill_surface, axis=1)
    row_numbers: IntGrid = np.arange(indices.size, dtype=np.int64)
    best_downhill_surface: FloatGrid = downhill_surface[row_numbers, best_downhill_slot]
    downhill_mask: BoolGrid = np.isfinite(best_downhill_surface) & (
        best_downhill_surface < surface[indices]
    )
    moved_indices: list[IntGrid] = []

    if np.any(downhill_mask):
        sources: IntGrid = indices[downhill_mask]
        targets: IntGrid = safe_neighbors[
            row_numbers[downhill_mask], best_downhill_slot[downhill_mask]
        ]
        height_drop: FloatGrid = surface[sources] - best_downhill_surface[downhill_mask]
        flows: FloatGrid = np.minimum(
            water_values[sources] - state.base_water[sources],
            water_values[sources]
            * config.downhill_flow_fraction
            * np.maximum(0.20, height_drop * 12.0),
        )
        positive: BoolGrid = flows > 0.0
        if np.any(positive):
            sources = sources[positive]
            targets = targets[positive]
            flows = flows[positive]
            np.add.at(delta, sources, -flows)
            np.add.at(delta, targets, flows)
            next_active[sources] = True
            next_active[targets] = True
            moved_indices.append(sources)

    remaining_indices: IntGrid = indices[~downhill_mask]
    if remaining_indices.size > 0:
        moved_sideways: IntGrid = _move_sideways_vectorized(
            indices=remaining_indices,
            water_values=water_values,
            heights=heights,
            surface=surface,
            state=state,
            config=config,
            delta=delta,
            next_active=next_active,
        )
        if moved_sideways.size > 0:
            moved_indices.append(moved_sideways)

    if not moved_indices:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(moved_indices).astype(np.int64)


def _move_sideways_vectorized(
    *,
    indices: IntGrid,
    water_values: FloatGrid,
    heights: FloatGrid,
    surface: FloatGrid,
    state: WaterSystemState,
    config: SimConfig,
    delta: FloatGrid,
    next_active: BoolGrid,
) -> IntGrid:
    neighbors: IntGrid = state.neighbors4[indices]
    valid: BoolGrid = neighbors >= 0
    safe_neighbors: IntGrid = np.where(valid, neighbors, 0)
    allowed_height: BoolGrid = np.where(
        valid,
        heights[safe_neighbors] <= heights[indices, None] + config.sideways_height_slop,
        False,
    )
    neighbor_surface: FloatGrid = np.where(
        allowed_height, surface[safe_neighbors], np.inf
    )
    best_slot: IntGrid = np.argmin(neighbor_surface, axis=1)
    rows: IntGrid = np.arange(indices.size, dtype=np.int64)
    best_surface: FloatGrid = neighbor_surface[rows, best_slot]
    spread_candidates: BoolGrid = np.isfinite(best_surface) & (
        best_surface < surface[indices]
    )
    if not np.any(spread_candidates):
        return np.empty(0, dtype=np.int64)

    sources: IntGrid = indices[spread_candidates]
    targets: IntGrid = safe_neighbors[
        rows[spread_candidates], best_slot[spread_candidates]
    ]
    depth_difference: FloatGrid = water_values[sources] - water_values[targets]
    spread_mask: BoolGrid = depth_difference > config.sideways_min_depth_difference
    if not np.any(spread_mask):
        return np.empty(0, dtype=np.int64)

    sources = sources[spread_mask]
    targets = targets[spread_mask]
    spreads: FloatGrid = np.minimum(
        water_values[sources] - state.base_water[sources],
        depth_difference[spread_mask] * config.sideways_spread_fraction,
    )
    positive: BoolGrid = spreads > 0.0
    if not np.any(positive):
        return np.empty(0, dtype=np.int64)

    sources = sources[positive]
    targets = targets[positive]
    spreads = spreads[positive]
    np.add.at(delta, sources, -spreads)
    np.add.at(delta, targets, spreads)
    next_active[sources] = True
    next_active[targets] = True
    return sources


def _spill_floodwater_vectorized(
    *,
    indices: IntGrid,
    water_values: FloatGrid,
    heights: FloatGrid,
    state: WaterSystemState,
    config: SimConfig,
    delta: FloatGrid,
    next_active: BoolGrid,
) -> None:
    excess: FloatGrid = water_values[indices] - state.bankfull_level[indices]
    positive: BoolGrid = excess > 0.0
    if not np.any(positive):
        return
    sources: IntGrid = indices[positive]
    excess = excess[positive]
    neighbors: IntGrid = state.neighbors4[sources]
    valid: BoolGrid = neighbors >= 0
    safe_neighbors: IntGrid = np.where(valid, neighbors, 0)
    spill_allowed: BoolGrid = (
        valid
        & ~state.permanent_water[safe_neighbors]
        & (heights[safe_neighbors] <= heights[sources, None])
    )
    neighbor_heights: FloatGrid = np.where(
        spill_allowed, heights[safe_neighbors], np.inf
    )
    best_slot: IntGrid = np.argmin(neighbor_heights, axis=1)
    rows: IntGrid = np.arange(sources.size, dtype=np.int64)
    target_height: FloatGrid = neighbor_heights[rows, best_slot]
    has_target: BoolGrid = np.isfinite(target_height)

    drain_sources: IntGrid = sources[~has_target]
    if drain_sources.size > 0:
        drains: FloatGrid = np.minimum(excess[~has_target], config.off_map_drain_rate)
        np.add.at(delta, drain_sources, -drains)
        next_active[drain_sources] = True

    spill_sources: IntGrid = sources[has_target]
    if spill_sources.size == 0:
        return
    targets: IntGrid = safe_neighbors[rows[has_target], best_slot[has_target]]
    spills: FloatGrid = excess[has_target] * config.flood_spill_fraction
    np.add.at(delta, spill_sources, -spills)
    np.add.at(delta, targets, spills)
    next_active[spill_sources] = True
    next_active[targets] = True


def _infiltrate_surface_water_vectorized(
    *,
    indices: IntGrid,
    water_values: FloatGrid,
    state: WaterSystemState,
    config: SimConfig,
    next_active: BoolGrid,
) -> None:
    soil_room: FloatGrid = np.maximum(
        0.0, state.soil_capacity[indices] - state.soil_water[indices]
    )
    absorbed: FloatGrid = np.minimum(
        soil_room, config.surface_infiltration_per_fluid_tick
    )
    absorb_mask: BoolGrid = absorbed > 0.0
    absorb_indices: IntGrid = indices[absorb_mask]
    if absorb_indices.size > 0:
        state.soil_water[absorb_indices] += absorbed[absorb_mask]
        water_values[absorb_indices] = np.maximum(
            0.0, water_values[absorb_indices] - absorbed[absorb_mask]
        )
    active_indices: IntGrid = indices[
        water_values[indices] >= config.surface_water_active_threshold
    ]
    next_active[active_indices] = True


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
