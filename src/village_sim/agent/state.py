"""Agent state."""

from __future__ import annotations

from dataclasses import dataclass, field

from village_sim.agent.decision import DecisionTrace
from village_sim.core.types import (
    ActionKind,
    DeathReason,
    GoalKind,
    Position,
    ResourceKind,
)


@dataclass(frozen=True, slots=True)
class MemoryMarker:
    """Read-only resource memory marker exposed to renderers."""

    position: Position
    kind: ResourceKind
    confidence: float


@dataclass(slots=True)
class AgentState:
    """Mutable state for a single survival agent."""

    agent_id: int
    position: Position
    thirst: float = 0.28
    hunger: float = 0.35
    fatigue: float = 0.20
    cold_stress: float = 0.0
    health: float = 1.0
    awake_ticks: int = 0
    alive: bool = True
    death_reason: DeathReason | None = None
    current_goal: GoalKind = GoalKind.EXPLORE
    current_action: ActionKind = ActionKind.IDLE
    target: Position | None = None
    path: list[Position] = field(default_factory=list)
    distance_walked: int = 0
    water_discoveries: int = 0
    food_discoveries: int = 0
    visited_counts: list[int] = field(default_factory=list)
    decision_trace: DecisionTrace = field(default_factory=DecisionTrace)
    memory_markers: list[MemoryMarker] = field(default_factory=list)

    def ensure_visit_buffer(self, world_size: int) -> None:
        if len(self.visited_counts) != world_size:
            self.visited_counts = [0 for _ in range(world_size)]

    def clamp_needs(self) -> None:
        self.thirst = min(1.0, max(0.0, self.thirst))
        self.hunger = min(1.0, max(0.0, self.hunger))
        self.fatigue = min(1.0, max(0.0, self.fatigue))
        self.cold_stress = min(1.0, max(0.0, self.cold_stress))
        self.health = min(1.0, max(0.0, self.health))
