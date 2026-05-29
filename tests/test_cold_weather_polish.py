"""Polish tests for cold/weather/shelter summaries."""

from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout

from village_sim.agent.perception import perceive
from village_sim.core.config import SimConfig
from village_sim.core.time import clock_from_tick
from village_sim.core.types import Position
from village_sim.run import print_cold_summary, run_batch
from village_sim.sim.engine import Simulation
from village_sim.sim.event_summary import (
    count_cold_status_events,
    count_cold_weather_events,
    count_shelter_events,
)
from village_sim.sim.events import TickEvent
from village_sim.sim.metrics import SimResult
from village_sim.world.discoverables import Discoverable


def _event(kind: str, message: str) -> TickEvent:
    return TickEvent(
        tick=0,
        day=0,
        actor="agent:1",
        kind=kind,
        message=message,
        x=0,
        y=0,
    )


def _result() -> SimResult:
    return SimResult(
        seed=1,
        days_elapsed=1.0,
        survived=True,
        death_reason=None,
        final_health=1.0,
        final_thirst=0.1,
        final_hunger=0.2,
        final_fatigue=0.3,
        final_cold_stress=0.4,
        final_temperature_c=3.0,
        final_feels_cold=True,
        final_is_sheltered=True,
        final_cold_status="cold",
        cold_weather_events=5,
        cold_status_events=6,
        shelter_events=7,
        water_discoveries=0,
        food_discoveries=0,
        distance_walked=0,
        remembered_water_sites=0,
        remembered_food_sites=0,
    )


class ColdWeatherPolishTests(unittest.TestCase):
    def test_event_summary_helpers_count_stable_messages(self) -> None:
        events = [
            _event("weather", "cold night"),
            _event("weather", "cold rain"),
            _event("weather", "cold night rain"),
            _event("weather", "cold day"),
            _event("weather", "rain fell"),
            _event("weather", "coldish debug"),
            _event("status", "agent is cold"),
            _event("status", "agent is severely cold"),
            _event("status", "not cold substring"),
            _event("action", "seeking shelter at cave_001"),
            _event("action", "sheltered at cave_001"),
            _event("action", "shelter substring only"),
        ]

        self.assertEqual(count_cold_weather_events(events), 4)
        self.assertEqual(count_cold_status_events(events), 2)
        self.assertEqual(count_shelter_events(events), 2)

    def test_print_cold_summary_uses_result_fields(self) -> None:
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_cold_summary(_result())

        self.assertEqual(
            buffer.getvalue().strip(),
            "Cold/weather: temp_c=3.0 feels_cold=True sheltered=True "
            "cold_weather_events=5 cold_status_events=6 shelter_events=7",
        )

    def test_sim_result_includes_final_weather_shelter_and_event_counts(self) -> None:
        config = SimConfig(
            width=32,
            height=32,
            max_days=1,
            seed=11,
            enable_initial_discoverables=True,
        )
        sim = Simulation(config)
        result = sim.run()
        payload = result.to_json_obj()

        self.assertIn("final_temperature_c", payload)
        self.assertIn("final_feels_cold", payload)
        self.assertIn("final_is_sheltered", payload)
        self.assertIn("final_cold_status", payload)
        self.assertIn("cold_weather_events", payload)
        self.assertIn("cold_status_events", payload)
        self.assertIn("shelter_events", payload)
        self.assertEqual(
            result.cold_weather_events, count_cold_weather_events(sim.events)
        )
        self.assertEqual(
            result.cold_status_events, count_cold_status_events(sim.events)
        )
        self.assertEqual(result.shelter_events, count_shelter_events(sim.events))
        self.assertEqual(result.final_is_sheltered, sim.agent_is_sheltered())

    def test_batch_output_includes_cold_weather_columns(self) -> None:
        args = argparse.Namespace(
            width=16,
            height=16,
            days=1,
            seed=1,
            discoverables=False,
            goap=False,
            batch=2,
        )
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            run_batch(args)

        output = buffer.getvalue()
        self.assertIn("final_cold_stress", output)
        self.assertIn("final_temperature_c", output)
        self.assertIn("final_feels_cold", output)
        self.assertIn("final_is_sheltered", output)
        self.assertIn("cold_weather_events", output)
        self.assertIn("cold_status_events", output)
        self.assertIn("shelter_events", output)

    def test_cave_logs_use_shelter_language_but_trajectory_keeps_exploit(self) -> None:
        config = SimConfig(
            width=32,
            height=32,
            max_days=1,
            seed=11,
            enable_initial_discoverables=True,
        )
        sim = Simulation(config)
        cave: Discoverable = sim.world.discoverables["cave_001"]
        sim.agent.position = Position(cave.x, cave.y)
        sim.agent.cold_stress = 0.9
        sim.agent.thirst = 0.1
        sim.agent.hunger = 0.1
        sim.agent.fatigue = 0.1

        sim.step()

        messages = [event.message for event in sim.events]
        self.assertIn("seeking shelter at cave_001", messages)
        self.assertIn("sheltered at cave_001", messages)
        self.assertNotIn("exploit cave_001 success", messages)
        self.assertEqual(
            sim.recorded_trajectories[0].steps[0].events,
            ["exploit:cave_001:success"],
        )

    def test_perceive_fallback_remains_for_legacy_tests(self) -> None:
        config = SimConfig(width=16, height=16, max_days=1, seed=3)
        sim = Simulation(config)
        clock = clock_from_tick(sim.tick, sim.config)

        observation = perceive(sim.world, sim.agent.position, clock, sim.config)

        self.assertFalse(observation.is_raining)
        self.assertEqual(observation.is_night, clock.is_night)
