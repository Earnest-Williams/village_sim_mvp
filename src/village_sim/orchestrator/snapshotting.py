"""Helpers for converting live simulation state into trajectory snapshots."""

from __future__ import annotations

from village_sim.agent.memory import DiscoverableAgentMemory
from village_sim.agent.perception import Observation
from village_sim.agent.state import AgentState
from village_sim.core.time import SimClock
from village_sim.orchestrator.symbolic import extract_symbolic_state
from village_sim.orchestrator.trajectory import NeedState, StateSnapshot


def make_need_state(agent: AgentState) -> NeedState:
    """Capture the mutable agent need fields as a trajectory value object."""
    return NeedState(
        hunger=agent.hunger,
        thirst=agent.thirst,
        fatigue=agent.fatigue,
        health=agent.health,
    )


def make_state_snapshot(
    *,
    tick: int,
    agent: AgentState,
    observation: Observation,
    discoverable_memory: DiscoverableAgentMemory,
    clock: SimClock,
) -> StateSnapshot:
    """Capture live state for trajectory recording and induction."""
    return StateSnapshot(
        tick=tick,
        agent_id=agent.agent_id,
        x=agent.position.x,
        y=agent.position.y,
        needs=make_need_state(agent),
        symbolic=extract_symbolic_state(
            agent,
            observation,
            discoverable_memory,
            clock,
        ),
    )
