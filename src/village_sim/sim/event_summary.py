"""Stable cold/weather/shelter event summary helpers."""

from __future__ import annotations

from collections.abc import Sequence

from village_sim.sim.events import TickEvent
from village_sim.world.weather import (
    SHELTERED_ACTION_PREFIX,
    SEEKING_SHELTER_ACTION_PREFIX,
    STATUS_AGENT_COLD,
    STATUS_AGENT_SEVERELY_COLD,
    WEATHER_COLD_DAY,
    WEATHER_COLD_NIGHT,
    WEATHER_COLD_NIGHT_RAIN,
    WEATHER_COLD_RAIN,
)

_COLD_WEATHER_MESSAGES: frozenset[str] = frozenset(
    {
        WEATHER_COLD_NIGHT,
        WEATHER_COLD_RAIN,
        WEATHER_COLD_NIGHT_RAIN,
        WEATHER_COLD_DAY,
    }
)
_COLD_STATUS_MESSAGES: frozenset[str] = frozenset(
    {
        STATUS_AGENT_COLD,
        STATUS_AGENT_SEVERELY_COLD,
    }
)


def count_cold_weather_events(events: Sequence[TickEvent]) -> int:
    """Count exact cold-weather transition events in a tick event sequence."""

    return sum(
        1
        for event in events
        if event.kind == "weather" and event.message in _COLD_WEATHER_MESSAGES
    )


def count_cold_status_events(events: Sequence[TickEvent]) -> int:
    """Count exact cold-status transition events in a tick event sequence."""

    return sum(
        1
        for event in events
        if event.kind == "status" and event.message in _COLD_STATUS_MESSAGES
    )


def count_shelter_events(events: Sequence[TickEvent]) -> int:
    """Count stable shelter action events in a tick event sequence."""

    return sum(
        1
        for event in events
        if event.kind == "action"
        and (
            event.message.startswith(SEEKING_SHELTER_ACTION_PREFIX)
            or event.message.startswith(SHELTERED_ACTION_PREFIX)
        )
    )
