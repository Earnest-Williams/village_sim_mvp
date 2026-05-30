"""Simulation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from village_sim.world.weather import ColdStatus


@dataclass(slots=True)
class LearningStats:
    """Compact counters that make resource-memory learning measurable."""

    learned_water_sites: int = 0
    learned_food_sites: int = 0
    memory_selected_water: int = 0
    memory_selected_food: int = 0
    memory_search_water: int = 0
    memory_search_food: int = 0
    memory_reinforced_water: int = 0
    memory_reinforced_food: int = 0
    memory_failed_water: int = 0
    memory_failed_food: int = 0
    explore_for_water_ticks: int = 0
    explore_for_food_ticks: int = 0
    visible_resource_target_ticks: int = 0
    remembered_resource_target_ticks: int = 0
    search_near_memory_ticks: int = 0

    @property
    def memory_directed_resource_ticks(self) -> int:
        return self.remembered_resource_target_ticks + self.search_near_memory_ticks

    @property
    def exploration_resource_ticks(self) -> int:
        return self.explore_for_water_ticks + self.explore_for_food_ticks

    @property
    def memory_use_ratio(self) -> float:
        total: int = (
            self.memory_directed_resource_ticks + self.exploration_resource_ticks
        )
        return float(self.memory_directed_resource_ticks) / float(max(1, total))

    def to_msgpack_obj(self) -> dict[str, Any]:
        data: dict[str, Any] = asdict(self)
        data["memory_directed_resource_ticks"] = self.memory_directed_resource_ticks
        data["exploration_resource_ticks"] = self.exploration_resource_ticks
        data["memory_use_ratio"] = self.memory_use_ratio
        return data


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
    final_temperature_c: float
    final_feels_cold: bool
    final_is_sheltered: bool
    final_cold_status: ColdStatus
    cold_weather_events: int
    cold_status_events: int
    shelter_events: int
    water_discoveries: int
    food_discoveries: int
    distance_walked: int
    remembered_water_sites: int
    remembered_food_sites: int
    learning: LearningStats = field(default_factory=LearningStats)
    best_water_memory_x: int = -1
    best_water_memory_y: int = -1
    best_water_memory_confidence: float = 0.0
    best_water_memory_successful_uses: int = 0
    best_water_memory_failed_uses: int = 0
    best_food_memory_x: int = -1
    best_food_memory_y: int = -1
    best_food_memory_confidence: float = 0.0
    best_food_memory_successful_uses: int = 0
    best_food_memory_failed_uses: int = 0
    action_library_size: int = 0
    synthesized_actions_added: int = 0
    goap_plan_executions: int = 0
    successful_goap_plan_executions: int = 0

    def to_msgpack_obj(self) -> dict[str, Any]:
        data: dict[str, Any] = asdict(self)
        data["learning"] = self.learning.to_msgpack_obj()
        return data
