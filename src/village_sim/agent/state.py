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

MAX_AGENTS = 500


@dataclass(frozen=True, slots=True)
class MemoryMarker:
    """Read-only resource memory marker exposed to renderers."""

    position: Position
    kind: ResourceKind
    confidence: float


@dataclass(slots=True)
class AgentState:
    """Cold-path compatibility view for one survival agent.

    Hot simulation code must use :class:`AgentArrays` as the source of truth.
    This adapter remains for renderers, GOAP compatibility, and legacy tests.
    """

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
    active: NDArray[np.bool_]
    x: NDArray[np.int32]
    y: NDArray[np.int32]
    thirst: NDArray[np.float32]
    hunger: NDArray[np.float32]
    fatigue: NDArray[np.float32]
    cold_stress: NDArray[np.float32]
    health: NDArray[np.float32]
    awake_ticks: NDArray[np.int32]
    current_goal: NDArray[np.int16]
    current_action: NDArray[np.int16]
    action_queue_kind: NDArray[np.int32]
    action_queue_duration: NDArray[np.int32]

    @property
    def alive(self) -> NDArray[np.bool_]:
        """Compatibility alias for the active mask."""

        return self.active


ACTION_TO_ID: dict[ActionKind, int] = {
    action: index for index, action in enumerate(ActionKind)
}
ID_TO_ACTION: dict[int, ActionKind] = {
    index: action for action, index in ACTION_TO_ID.items()
}
GOAL_TO_ID: dict[GoalKind, int] = {goal: index for index, goal in enumerate(GoalKind)}
ID_TO_GOAL: dict[int, GoalKind] = {index: goal for goal, index in GOAL_TO_ID.items()}


def make_agent_arrays(capacity: int = MAX_AGENTS) -> AgentArrays:
    """Allocate bounded SoA buffers for the requested agent capacity."""

    if capacity <= 0:
        raise ValueError("agent array capacity must be positive")
    return AgentArrays(
        count=capacity,
        active=np.zeros(capacity, dtype=np.bool_),
        x=np.zeros(capacity, dtype=np.int32),
        y=np.zeros(capacity, dtype=np.int32),
        thirst=np.zeros(capacity, dtype=np.float32),
        hunger=np.zeros(capacity, dtype=np.float32),
        fatigue=np.zeros(capacity, dtype=np.float32),
        cold_stress=np.zeros(capacity, dtype=np.float32),
        health=np.ones(capacity, dtype=np.float32),
        awake_ticks=np.zeros(capacity, dtype=np.int32),
        current_goal=np.zeros(capacity, dtype=np.int16),
        current_action=np.zeros(capacity, dtype=np.int16),
        action_queue_kind=np.zeros(capacity, dtype=np.int32),
        action_queue_duration=np.zeros(capacity, dtype=np.int32),
    )


def agent_arrays_from_states(
    agents: list[AgentState], capacity: int = MAX_AGENTS
) -> AgentArrays:
    """Create preallocated SoA buffers from compatibility dataclasses."""

    if len(agents) > capacity:
        raise ValueError("agent state count exceeds requested array capacity")
    arrays: AgentArrays = make_agent_arrays(capacity)
    for index, agent in enumerate(agents):
        sync_agent_to_arrays(arrays, agent, index)
    return arrays


def sync_agent_to_arrays(arrays: AgentArrays, agent: AgentState, index: int) -> None:
    """Copy one compatibility AgentState into SoA buffers."""

    _validate_index(arrays, index)
    arrays.active[index] = agent.alive
    arrays.x[index] = agent.position.x
    arrays.y[index] = agent.position.y
    arrays.thirst[index] = np.float32(agent.thirst)
    arrays.hunger[index] = np.float32(agent.hunger)
    arrays.fatigue[index] = np.float32(agent.fatigue)
    arrays.cold_stress[index] = np.float32(agent.cold_stress)
    arrays.health[index] = np.float32(agent.health)
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
    agent.alive = bool(arrays.active[index])
    agent.current_goal = ID_TO_GOAL.get(
        int(arrays.current_goal[index]), GoalKind.EXPLORE
    )
    agent.current_action = ID_TO_ACTION.get(
        int(arrays.current_action[index]), ActionKind.IDLE
    )


def validate_arrays_match_dataclasses(
    agent_arrays: AgentArrays,
    agent_state_list: list[AgentState],
) -> None:
    """Strict debug assertion that compatibility state matches SoA buffers."""

    if agent_arrays.count < len(agent_state_list):
        raise AssertionError("agent array count is smaller than dataclass count")
    for index, agent in enumerate(agent_state_list):
        if bool(agent_arrays.active[index]) != agent.alive:
            raise AssertionError("agent active mask mismatch")
        if int(agent_arrays.x[index]) != agent.position.x:
            raise AssertionError("agent x mismatch")
        if int(agent_arrays.y[index]) != agent.position.y:
            raise AssertionError("agent y mismatch")
        if abs(float(agent_arrays.thirst[index]) - agent.thirst) > 1.0e-6:
            raise AssertionError("agent thirst mismatch")
        if abs(float(agent_arrays.hunger[index]) - agent.hunger) > 1.0e-6:
            raise AssertionError("agent hunger mismatch")
        if abs(float(agent_arrays.fatigue[index]) - agent.fatigue) > 1.0e-6:
            raise AssertionError("agent fatigue mismatch")
        if abs(float(agent_arrays.cold_stress[index]) - agent.cold_stress) > 1.0e-6:
            raise AssertionError("agent cold stress mismatch")
        if abs(float(agent_arrays.health[index]) - agent.health) > 1.0e-6:
            raise AssertionError("agent health mismatch")
        if int(agent_arrays.awake_ticks[index]) != agent.awake_ticks:
            raise AssertionError("agent awake ticks mismatch")
        if ID_TO_GOAL.get(int(agent_arrays.current_goal[index])) != agent.current_goal:
            raise AssertionError("agent goal mismatch")
        if (
            ID_TO_ACTION.get(int(agent_arrays.current_action[index]))
            != agent.current_action
        ):
            raise AssertionError("agent action mismatch")


def _validate_index(arrays: AgentArrays, index: int) -> None:
    if index < 0 or index >= arrays.count:
        raise IndexError("agent array index out of range")
