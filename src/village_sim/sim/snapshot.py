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
    health: float
    alive: bool
    goal: str
    action: str

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    tick: int
    day: int
    tick_of_day: int
    is_daylight: bool
    agents: list[AgentSnapshot]
    ascii_map: str | None = None

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)
