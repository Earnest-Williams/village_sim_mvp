from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from village_sim.agent.needs import update_needs
from village_sim.agent.perception import perceive
from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.time import clock_from_tick
from village_sim.core.types import Position
from village_sim.sim.engine import Simulation
from village_sim.sim.replay import write_run_report
from village_sim.world.discoverables import Discoverable
from village_sim.world.weather import make_weather_state


def _cave_position(sim: Simulation) -> Position:
    cave: Discoverable = sim.world.discoverables["cave_001"]
    return Position(cave.x, cave.y)


class TestWeatherColdAwareness(unittest.TestCase):
    def test_weather_derivation_is_deterministic(self) -> None:
        config = SimConfig()

        day = make_weather_state(is_raining=False, is_night=False, config=config)
        self.assertFalse(day.feels_cold)
        self.assertEqual(day.temperature_c, 18.0)
        self.assertEqual(day.cold_reason, "none")

        night = make_weather_state(is_raining=False, is_night=True, config=config)
        self.assertTrue(night.feels_cold)
        self.assertEqual(night.temperature_c, 7.0)
        self.assertEqual(night.cold_reason, "night")

        rainy_day = make_weather_state(is_raining=True, is_night=False, config=config)
        self.assertEqual(rainy_day.temperature_c, 15.0)
        self.assertFalse(rainy_day.feels_cold)
        self.assertEqual(rainy_day.cold_reason, "none")

        cold_rain_config = SimConfig(cold_temperature_threshold_c=16.0)
        cold_rain = make_weather_state(
            is_raining=True,
            is_night=False,
            config=cold_rain_config,
        )
        self.assertTrue(cold_rain.feels_cold)
        self.assertEqual(cold_rain.cold_reason, "rain")

        night_rain = make_weather_state(is_raining=True, is_night=True, config=config)
        self.assertTrue(night_rain.feels_cold)
        self.assertEqual(night_rain.temperature_c, 4.0)
        self.assertEqual(night_rain.cold_reason, "night_rain")

    def test_perception_includes_weather_cold_and_shelter(self) -> None:
        config = SimConfig(enable_initial_discoverables=True)
        sim = Simulation(config)
        sim.agent.position = _cave_position(sim)
        clock = clock_from_tick(0, config)
        weather = make_weather_state(
            is_raining=True,
            is_night=clock.is_night,
            config=config,
        )

        observation = perceive(
            sim.world,
            sim.agent.position,
            clock,
            config,
            weather,
            sim._agent_is_sheltered(),
        )

        self.assertTrue(observation.is_raining)
        self.assertEqual(observation.temperature_c, 4.0)
        self.assertTrue(observation.feels_cold)
        self.assertTrue(observation.is_sheltered)

    def test_need_updates_use_cold_exposure_and_shelter(self) -> None:
        config = SimConfig()
        agent = AgentState(agent_id=1, position=Position(0, 0))
        before = agent.cold_stress

        update_needs(
            agent,
            config,
            is_cold_exposed=True,
            is_sheltered=False,
        )
        self.assertGreater(agent.cold_stress, before)

        before_shelter = agent.cold_stress
        update_needs(agent, config, is_cold_exposed=True, is_sheltered=True)
        self.assertLess(agent.cold_stress, before_shelter)

        agent.cold_stress = 0.1
        update_needs(
            agent,
            config,
            is_night=False,
            is_raining=False,
            is_cold_exposed=False,
            is_sheltered=False,
        )
        self.assertLess(agent.cold_stress, 0.1)

    def test_logs_include_weather_status_and_shelter_events(self) -> None:
        config = SimConfig(enable_initial_discoverables=True)
        sim = Simulation(config)
        self.assertTrue(
            any(
                event.kind == "weather" and "cold" in event.message
                for event in sim.events
            )
        )

        sim.agent.cold_stress = 0.299
        sim.step()
        self.assertTrue(
            any(
                event.kind == "status" and event.message == "agent is cold"
                for event in sim.events
            )
        )

        sim.agent.position = _cave_position(sim)
        sim.agent.cold_stress = 0.9
        sim.agent.thirst = 0.1
        sim.agent.hunger = 0.1
        sim.step()
        messages = [event.message for event in sim.events if event.kind == "action"]
        self.assertIn("seeking shelter at cave_001", messages)
        self.assertIn("sheltered at cave_001", messages)

    def test_snapshots_and_replay_include_cold_fields(self) -> None:
        config = SimConfig(enable_initial_discoverables=True)
        sim = Simulation(config)
        sim.step()
        snapshot = sim.snapshot(include_ascii=False)

        self.assertTrue(hasattr(snapshot, "is_raining"))
        self.assertTrue(hasattr(snapshot, "temperature_c"))
        self.assertTrue(hasattr(snapshot, "feels_cold"))
        self.assertTrue(hasattr(snapshot, "cold_reason"))
        self.assertTrue(hasattr(snapshot.agents[0], "feels_cold"))
        self.assertTrue(hasattr(snapshot.agents[0], "is_sheltered"))

        sim.snapshots.append(snapshot)
        result = sim.result()
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_path = Path(temp_dir) / "replay.json"
            write_run_report(replay_path, config, result, sim.events, sim.snapshots)
            payload: dict[str, Any] = json.loads(replay_path.read_text())

        replay_snapshot = payload["snapshots"][0]
        replay_agent = replay_snapshot["agents"][0]
        self.assertIn("temperature_c", replay_snapshot)
        self.assertIn("feels_cold", replay_snapshot)
        self.assertIn("cold_reason", replay_snapshot)
        self.assertIn("feels_cold", replay_agent)
        self.assertIn("is_sheltered", replay_agent)

    def test_goap_shelter_behavior_logs_shelter_action(self) -> None:
        config = SimConfig(
            enable_initial_discoverables=True,
            enable_goap_control=True,
        )
        sim = Simulation(config)
        sim.agent.position = _cave_position(sim)
        sim.agent.cold_stress = 0.95
        sim.agent.thirst = 0.1
        sim.agent.hunger = 0.1
        sim.step()

        shelter_messages = [
            event.message
            for event in sim.events
            if event.kind == "action" and "shelter" in event.message
        ]
        self.assertTrue(shelter_messages)


if __name__ == "__main__":
    unittest.main()
