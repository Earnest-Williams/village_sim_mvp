"""Deterministic weather and cold exposure state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from village_sim.core.config import SimConfig

ColdReason = Literal["none", "night", "rain", "night_rain", "day"]


@dataclass(frozen=True, slots=True)
class WeatherState:
    """Typed weather facts visible to the simulation and agent."""

    is_raining: bool
    temperature_c: float
    feels_cold: bool
    cold_reason: ColdReason


def make_weather_state(
    *,
    is_raining: bool,
    is_night: bool,
    config: SimConfig,
) -> WeatherState:
    """Derive deterministic weather facts from rain, daylight, and config."""

    base_temperature_c: float = (
        config.night_temperature_c if is_night else config.day_temperature_c
    )
    rain_penalty_c: float = config.rain_temperature_penalty_c if is_raining else 0.0
    temperature_c: float = base_temperature_c - rain_penalty_c
    feels_cold: bool = temperature_c <= config.cold_temperature_threshold_c
    cold_reason: ColdReason = "none"
    if feels_cold:
        if is_night and is_raining:
            cold_reason = "night_rain"
        elif is_night:
            cold_reason = "night"
        elif is_raining:
            cold_reason = "rain"
        else:
            cold_reason = "day"
    return WeatherState(
        is_raining=is_raining,
        temperature_c=temperature_c,
        feels_cold=feels_cold,
        cold_reason=cold_reason,
    )
