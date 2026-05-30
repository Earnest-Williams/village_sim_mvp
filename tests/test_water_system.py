"""Tests for sampled rain and active water flow."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from village_sim.core.config import SimConfig
from village_sim.core.types import TerrainKind
from village_sim.world.water_system import (
    RainEvent,
    RainKind,
    RainProfile,
    RainSystem,
    apply_rain_to_tile,
    build_water_system_state,
    step_active_water_flow,
    step_sampled_rain,
    step_water_maintenance_bank,
)


class RainSystemTests(unittest.TestCase):
    def test_no_active_event_if_random_roll_misses(self) -> None:
        config = replace(SimConfig(), rain_event_start_chance_per_tick=0.0)
        system = RainSystem()

        event = system.tick(random.Random(1), config)

        self.assertIsNone(event)
        self.assertIsNone(system.active_event)

    def test_active_event_starts_when_random_roll_hits(self) -> None:
        config = replace(SimConfig(), rain_event_start_chance_per_tick=1.0)
        system = RainSystem()

        event = system.tick(random.Random(1), config)

        self.assertIsNotNone(event)
        self.assertIs(system.active_event, event)
        self.assertGreater(event.ticks_remaining if event is not None else 0, 0)

    def test_event_duration_decrements_and_eventually_ends(self) -> None:
        system = RainSystem(active_event=RainEvent(_test_profile(), 2))

        first = system.tick(random.Random(1), SimConfig())
        first_ticks_remaining = first.ticks_remaining if first is not None else -1
        second = system.tick(random.Random(1), SimConfig())

        self.assertIsNotNone(first)
        self.assertEqual(first_ticks_remaining, 1)
        self.assertIsNone(second)
        self.assertIsNone(system.active_event)


class SampledRainTests(unittest.TestCase):
    def test_catchment_tiles_add_water_up_to_max_depth(self) -> None:
        config = replace(SimConfig(), max_surface_water_depth=2.0)
        terrain = [int(TerrainKind.HILL)]
        water = [1.75]
        state = build_water_system_state(
            width=1, height=1, terrain=terrain, water=water
        )
        event = RainEvent(
            replace(_test_profile(), sample_fraction=1.0, catchment_gain=1.0),
            1,
        )

        step_sampled_rain(
            rng=random.Random(4),
            water=water,
            state=state,
            event=event,
            config=config,
        )

        self.assertEqual(water[0], 2.0)
        self.assertIn(0, state.fluid_active)

    def test_non_catchment_tiles_update_soil_wetness_and_surface_water(self) -> None:
        config = SimConfig()
        terrain = [int(TerrainKind.ROCK)]
        water = [0.0]
        state = build_water_system_state(
            width=1, height=1, terrain=terrain, water=water
        )

        apply_rain_to_tile(
            index=0,
            rain_amount=1.0,
            water=water,
            state=state,
            config=config,
        )

        self.assertEqual(state.wetness[0], 1.0)
        self.assertAlmostEqual(state.soil_water[0], 0.1)
        self.assertAlmostEqual(water[0], 0.81)
        self.assertIn(0, state.fluid_active)

    def test_sampling_without_replacement_covers_all_tiles(self) -> None:
        terrain = [int(TerrainKind.ROCK), int(TerrainKind.ROCK)]
        water = [0.0, 0.0]
        state = build_water_system_state(
            width=2, height=1, terrain=terrain, water=water
        )
        event = RainEvent(
            replace(_test_profile(), sample_fraction=1.0, puddle_rate=1.0),
            1,
        )

        step_sampled_rain(
            rng=random.Random(1),
            water=water,
            state=state,
            event=event,
            config=SimConfig(),
        )

        self.assertGreater(water[0], 0.0)
        self.assertGreater(water[1], 0.0)


class ActiveWaterFlowTests(unittest.TestCase):
    def test_water_moves_to_lower_height_neighbor(self) -> None:
        terrain = [int(TerrainKind.GRASS), int(TerrainKind.GRASS)]
        water = [1.0, 0.0]
        state = build_water_system_state(
            width=2, height=1, terrain=terrain, water=water
        )
        state.fluid_active.add(0)

        step_active_water_flow(
            water=water,
            height_map=[1.0, 0.0],
            state=state,
            config=SimConfig(),
        )

        self.assertLess(water[0], 1.0)
        self.assertGreater(water[1], 0.0)

    def test_water_spreads_sideways_when_no_lower_height_neighbor_exists(self) -> None:
        terrain = [int(TerrainKind.GRASS), int(TerrainKind.GRASS)]
        water = [1.0, 0.0]
        state = build_water_system_state(
            width=2, height=1, terrain=terrain, water=water
        )
        state.fluid_active.add(0)

        step_active_water_flow(
            water=water,
            height_map=[0.0, 0.0],
            state=state,
            config=SimConfig(),
        )

        self.assertAlmostEqual(water[0], 0.75)
        self.assertAlmostEqual(water[1], 0.25)

    def test_inactive_dry_tiles_are_not_processed(self) -> None:
        terrain = [int(TerrainKind.GRASS), int(TerrainKind.GRASS)]
        water = [0.0, 1.0]
        state = build_water_system_state(
            width=2, height=1, terrain=terrain, water=water
        )

        step_active_water_flow(
            water=water,
            height_map=[0.0, -1.0],
            state=state,
            config=SimConfig(),
        )

        self.assertEqual(water, [0.0, 1.0])
        self.assertEqual(state.soil_water, [0.0, 0.0])

    def test_sideways_flow_does_not_move_water_uphill_surface(self) -> None:
        terrain = [int(TerrainKind.GRASS), int(TerrainKind.GRASS)]
        water = [0.20, 0.0]
        state = build_water_system_state(
            width=2, height=1, terrain=terrain, water=water
        )
        state.fluid_active.add(0)

        step_active_water_flow(
            water=water,
            height_map=[0.0, 0.009],
            state=state,
            config=SimConfig(),
        )

        self.assertEqual(water[1], 0.0)

    def test_permanent_water_below_bankfull_does_not_flow(self) -> None:
        terrain = [int(TerrainKind.WATER), int(TerrainKind.GRASS)]
        water = [1.0, 0.0]
        state = build_water_system_state(
            width=2, height=1, terrain=terrain, water=water
        )
        state.fluid_active.add(0)

        step_active_water_flow(
            water=water,
            height_map=[1.0, 0.0],
            state=state,
            config=SimConfig(),
        )

        self.assertEqual(water, [1.0, 0.0])
        self.assertNotIn(0, state.fluid_active)


class BankedMaintenanceTests(unittest.TestCase):
    def test_only_one_bank_is_processed_per_tick_and_soil_drains(self) -> None:
        terrain = [int(TerrainKind.GRASS) for _ in range(4)]
        water = [0.0 for _ in range(4)]
        state = build_water_system_state(
            width=4, height=1, terrain=terrain, water=water
        )
        state.soil_water = [0.5, 0.5, 0.5, 0.5]
        state.wetness = [0.5, 0.5, 0.5, 0.5]

        step_water_maintenance_bank(
            tick=1, water=water, state=state, config=SimConfig()
        )

        self.assertEqual(state.soil_water[0], 0.5)
        self.assertAlmostEqual(state.soil_water[1], 0.45)
        self.assertEqual(state.wetness[0], 0.5)
        self.assertAlmostEqual(state.wetness[1], 0.45)

    def test_permanent_water_restores_to_base_water(self) -> None:
        terrain = [int(TerrainKind.WATER)]
        water = [0.1]
        state = build_water_system_state(
            width=1, height=1, terrain=terrain, water=water
        )
        water[0] = 0.1

        step_water_maintenance_bank(
            tick=0, water=water, state=state, config=SimConfig()
        )

        self.assertEqual(water[0], 0.65)

    def test_maintenance_does_not_activate_inactive_deep_water(self) -> None:
        terrain = [int(TerrainKind.GRASS)]
        water = [1.0]
        state = build_water_system_state(
            width=1, height=1, terrain=terrain, water=water
        )

        step_water_maintenance_bank(
            tick=0, water=water, state=state, config=SimConfig()
        )

        self.assertEqual(water[0], 1.0)
        self.assertEqual(state.fluid_active, set())


class FloodBehaviorTests(unittest.TestCase):
    def test_permanent_water_above_bankfull_spills_to_neighbor(self) -> None:
        terrain = [int(TerrainKind.WATER), int(TerrainKind.GRASS)]
        water = [2.0, 0.0]
        state = build_water_system_state(
            width=2, height=1, terrain=terrain, water=water
        )
        state.flood_active.add(0)

        step_active_water_flow(
            water=water,
            height_map=[1.0, 1.0],
            state=state,
            config=SimConfig(),
        )

        self.assertLess(water[0], 2.0)
        self.assertGreater(water[1], 0.0)
        self.assertIn(1, state.fluid_active)


def _test_profile() -> RainProfile:
    return RainProfile(
        kind=RainKind.RAIN,
        application_interval_ticks=1,
        sample_fraction=1.0,
        rain_min=1.0,
        rain_max=1.0,
        puddle_rate=1.0,
        catchment_gain=1.0,
        min_duration_ticks=1,
        max_duration_ticks=1,
    )


if __name__ == "__main__":
    unittest.main()
