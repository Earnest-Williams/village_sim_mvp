"""Simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Immutable configuration for deterministic simulation runs."""

    width: int = 32
    height: int = 32
    max_days: int = 10
    ticks_per_day: int = 144
    seed: int = 1

    day_start_tick: int = 36
    night_start_tick: int = 108

    vision_radius_day: int = 12
    vision_radius_night: int = 3

    thirst_gain_per_tick: float = 0.0045
    hunger_gain_per_tick: float = 0.0025
    fatigue_gain_awake: float = 0.006
    fatigue_recovery_sleeping: float = 0.030

    drink_amount_per_tick: float = 0.24
    eat_amount_per_tick: float = 0.22

    rain_chance_per_tick: float = 0.020
    rain_amount: float = 0.030
    evaporation_per_tick: float = 0.0015
    downhill_flow_fraction: float = 0.18
    min_flow_water: float = 0.020

    food_regrowth_per_tick: float = 0.0007
    max_food_per_cell: float = 1.0

    water_memory_decay_per_day: float = 0.015
    food_memory_decay_per_day: float = 0.120

    def max_ticks(self) -> int:
        return self.max_days * self.ticks_per_day

    def validate(self) -> None:
        if self.width < 8:
            raise ValueError("width must be at least 8")
        if self.height < 8:
            raise ValueError("height must be at least 8")
        if self.max_days < 1:
            raise ValueError("max_days must be at least 1")
        if self.ticks_per_day < 24:
            raise ValueError("ticks_per_day must be at least 24")
        if not 0 <= self.day_start_tick < self.ticks_per_day:
            raise ValueError("day_start_tick must be inside the day")
        if not 0 <= self.night_start_tick < self.ticks_per_day:
            raise ValueError("night_start_tick must be inside the day")
