"""Simulation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SimResult:
    """Summary result for one simulation run."""

    seed: int
    days_elapsed: float
    survived: bool
    death_reason: str | None
    final_health: float
    final_thirst: float
    final_hunger: float
    final_fatigue: float
    final_cold_stress: float
    water_discoveries: int
    food_discoveries: int
    distance_walked: int
    remembered_water_sites: int
    remembered_food_sites: int

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)
