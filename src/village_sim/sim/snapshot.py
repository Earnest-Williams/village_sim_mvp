"""Serializable snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    agent_id: int
    x: int
    y: int
    thirst: float
    hunger: float
    fatigue: float
    cold_stress: float
    health: float
    alive: bool
    goal: str
    action: str
    feels_cold: bool = False
    is_sheltered: bool = False
    cold_status: str = "ok"
    decision_source: str = "none"
    decision_target_kind: str = "none"
    decision_target_x: int = -1
    decision_target_y: int = -1
    decision_memory_confidence: float = 0.0
    memory_use_ratio: float = 0.0

    def to_msgpack_obj(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    tick: int
    day: int
    tick_of_day: int
    is_daylight: bool
    agents: list[AgentSnapshot]
    ascii_map: str | None = None
    is_raining: bool = False
    temperature_c: float = 18.0
    feels_cold: bool = False
    cold_reason: str = "none"

    def to_msgpack_obj(self) -> dict[str, Any]:
        return asdict(self)
