"""Agent state and struct-of-arrays buffers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

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
    """Mutable compatibility state for a single survival agent."""

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
    path: list[Position] = field(default_factory=lambda: list[Position]())
    distance_walked: int = 0
    water_discoveries: int = 0
    food_discoveries: int = 0
    visited_counts: list[int] = field(default_factory=lambda: list[int]())
    decision_trace: DecisionTrace = field(default_factory=DecisionTrace)
    memory_markers: list[MemoryMarker] = field(
        default_factory=lambda: list[MemoryMarker]()
    )

    def ensure_visit_buffer(self, world_size: int) -> None:
        if len(self.visited_counts) != world_size:
            self.visited_counts = [0 for _ in range(world_size)]

    def clamp_needs(self) -> None:
        self.thirst = min(1.0, max(0.0, self.thirst))
        self.hunger = min(1.0, max(0.0, self.hunger))
        self.fatigue = min(1.0, max(0.0, self.fatigue))
        self.cold_stress = min(1.0, max(0.0, self.cold_stress))
        self.health = min(1.0, max(0.0, self.health))


@dataclass(slots=True)
class AgentArrays:
    """Struct-of-arrays source of truth for hot agent simulation fields."""

    count: int
    alive: NDArray[np.bool_]
    x: NDArray[np.int32]
    y: NDArray[np.int32]
    thirst: NDArray[np.float64]
    hunger: NDArray[np.float64]
    fatigue: NDArray[np.float64]
    cold_stress: NDArray[np.float64]
    health: NDArray[np.float64]
    awake_ticks: NDArray[np.int32]
    current_goal: NDArray[np.int16]
    current_action: NDArray[np.int16]


ACTION_TO_ID: dict[ActionKind, int] = {
    action: index for index, action in enumerate(ActionKind)
}
ID_TO_ACTION: dict[int, ActionKind] = {
    index: action for action, index in ACTION_TO_ID.items()
}
GOAL_TO_ID: dict[GoalKind, int] = {goal: index for index, goal in enumerate(GoalKind)}
ID_TO_GOAL: dict[int, GoalKind] = {index: goal for goal, index in GOAL_TO_ID.items()}


def make_agent_arrays(capacity: int) -> AgentArrays:
    """Allocate zero-filled SoA buffers for the requested agent capacity."""

    if capacity <= 0:
        raise ValueError("agent array capacity must be positive")
    return AgentArrays(
        count=capacity,
        alive=np.zeros(capacity, dtype=np.bool_),
        x=np.zeros(capacity, dtype=np.int32),
        y=np.zeros(capacity, dtype=np.int32),
        thirst=np.zeros(capacity, dtype=np.float64),
        hunger=np.zeros(capacity, dtype=np.float64),
        fatigue=np.zeros(capacity, dtype=np.float64),
        cold_stress=np.zeros(capacity, dtype=np.float64),
        health=np.ones(capacity, dtype=np.float64),
        awake_ticks=np.zeros(capacity, dtype=np.int32),
        current_goal=np.zeros(capacity, dtype=np.int16),
        current_action=np.zeros(capacity, dtype=np.int16),
    )


def agent_arrays_from_states(agents: list[AgentState]) -> AgentArrays:
    """Create SoA buffers from compatibility dataclasses."""

    arrays: AgentArrays = make_agent_arrays(len(agents))
    for index, agent in enumerate(agents):
        sync_agent_to_arrays(arrays, agent, index)
    return arrays


def sync_agent_to_arrays(arrays: AgentArrays, agent: AgentState, index: int) -> None:
    """Copy one compatibility AgentState into SoA buffers."""

    _validate_index(arrays, index)
    arrays.alive[index] = agent.alive
    arrays.x[index] = agent.position.x
    arrays.y[index] = agent.position.y
    arrays.thirst[index] = agent.thirst
    arrays.hunger[index] = agent.hunger
    arrays.fatigue[index] = agent.fatigue
    arrays.cold_stress[index] = agent.cold_stress
    arrays.health[index] = agent.health
    arrays.awake_ticks[index] = agent.awake_ticks
    arrays.current_goal[index] = GOAL_TO_ID[agent.current_goal]
    arrays.current_action[index] = ACTION_TO_ID[agent.current_action]


def sync_agent_from_arrays(arrays: AgentArrays, agent: AgentState, index: int) -> None:
    """Copy one SoA row back to a compatibility AgentState."""

    _validate_index(arrays, index)
    agent.position = Position(x=int(arrays.x[index]), y=int(arrays.y[index]))
    agent.thirst = float(arrays.thirst[index])
    agent.hunger = float(arrays.hunger[index])
    agent.fatigue = float(arrays.fatigue[index])
    agent.cold_stress = float(arrays.cold_stress[index])
    agent.health = float(arrays.health[index])
    agent.awake_ticks = int(arrays.awake_ticks[index])
    agent.alive = bool(arrays.alive[index])
    agent.current_goal = ID_TO_GOAL[int(arrays.current_goal[index])]
    agent.current_action = ID_TO_ACTION[int(arrays.current_action[index])]


def validate_arrays_match_dataclasses(
    agent_arrays: AgentArrays,
    agent_state_list: list[AgentState],
) -> None:
    """Strict debug assertion that compatibility state matches SoA buffers."""

    if agent_arrays.count != len(agent_state_list):
        raise AssertionError("agent array count does not match dataclass count")
    for index, agent in enumerate(agent_state_list):
        if bool(agent_arrays.alive[index]) != agent.alive:
            raise AssertionError("agent alive mismatch")
        if int(agent_arrays.x[index]) != agent.position.x:
            raise AssertionError("agent x mismatch")
        if int(agent_arrays.y[index]) != agent.position.y:
            raise AssertionError("agent y mismatch")
        _assert_float_matches(float(agent_arrays.thirst[index]), agent.thirst, "thirst")
        _assert_float_matches(float(agent_arrays.hunger[index]), agent.hunger, "hunger")
        _assert_float_matches(
            float(agent_arrays.fatigue[index]), agent.fatigue, "fatigue"
        )
        _assert_float_matches(
            float(agent_arrays.cold_stress[index]), agent.cold_stress, "cold_stress"
        )
        _assert_float_matches(float(agent_arrays.health[index]), agent.health, "health")
        if int(agent_arrays.awake_ticks[index]) != agent.awake_ticks:
            raise AssertionError("agent awake_ticks mismatch")
        if ID_TO_GOAL[int(agent_arrays.current_goal[index])] is not agent.current_goal:
            raise AssertionError("agent current_goal mismatch")
        if (
            ID_TO_ACTION[int(agent_arrays.current_action[index])]
            is not agent.current_action
        ):
            raise AssertionError("agent current_action mismatch")


def _validate_index(arrays: AgentArrays, index: int) -> None:
    if not 0 <= index < arrays.count:
        raise IndexError("agent array index out of range")


def _assert_float_matches(actual: float, expected: float, name: str) -> None:
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"agent {name} mismatch")
