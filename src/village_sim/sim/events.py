"""Simulation event records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TickEvent:
    """Compact event emitted by the simulation."""

    tick: int
    day: int
    actor: str
    kind: str
    message: str
    x: int
    y: int

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)
