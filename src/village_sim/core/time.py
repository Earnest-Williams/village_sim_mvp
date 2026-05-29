"""Discrete simulation time helpers."""

from __future__ import annotations

from dataclasses import dataclass

from village_sim.core.config import SimConfig


@dataclass(frozen=True, slots=True)
class SimClock:
    """Derived clock values for the current simulation tick."""

    tick: int
    day: int
    tick_of_day: int
    is_daylight: bool
    is_night: bool


def clock_from_tick(tick: int, config: SimConfig) -> SimClock:
    tick_of_day: int = tick % config.ticks_per_day
    day: int = tick // config.ticks_per_day
    is_daylight: bool = config.day_start_tick <= tick_of_day < config.night_start_tick
    return SimClock(
        tick=tick,
        day=day,
        tick_of_day=tick_of_day,
        is_daylight=is_daylight,
        is_night=not is_daylight,
    )
