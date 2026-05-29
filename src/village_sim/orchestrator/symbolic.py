"""Symbolic state extraction (§7, §8).

Raw simulation state is converted into a flat dictionary of symbolic facts
for use by the orchestrator and GOAP planner.
"""

from __future__ import annotations

from village_sim.agent.memory import DiscoverableAgentMemory
from village_sim.agent.perception import Observation
from village_sim.agent.state import AgentState
from village_sim.core.time import SimClock
from village_sim.world.discoverables import DiscoverableKind

FactValue = bool | int | float | str
SymbolicState = dict[str, FactValue]


# ── Need bucketing (§8) ───────────────────────────────────────────────────────


def bucket_need(value: float) -> str:
    """Convert a raw need float to a symbolic bucket label."""
    if value >= 0.80:
        return "critical"
    if value >= 0.60:
        return "high"
    if value >= 0.30:
        return "medium"
    return "low"


# ── Full state predicate extractor (§7) ───────────────────────────────────────


def extract_symbolic_state(
    agent: AgentState,
    observation: Observation,
    disc_memory: DiscoverableAgentMemory,
    clock: SimClock,
) -> SymbolicState:
    """Convert raw simulation state into a flat symbolic fact dictionary."""

    state: SymbolicState = {
        "hunger_bucket": bucket_need(agent.hunger),
        "thirst_bucket": bucket_need(agent.thirst),
        "fatigue_bucket": bucket_need(agent.fatigue),
        "health_low": agent.health < 0.30,
        "is_daylight": clock.is_daylight,
    }

    # Closest discoverable in current observation (§7 example predicates)
    if observation.discoverables:
        closest = min(
            observation.discoverables,
            key=lambda d: (d.x - agent.position.x) ** 2
            + (d.y - agent.position.y) ** 2,
        )
        at_disc = (
            abs(closest.x - agent.position.x) <= 1
            and abs(closest.y - agent.position.y) <= 1
        )
        state["at_discoverable"] = at_disc
        state["visible_discoverable"] = True
        state["target_id"] = closest.discoverable_id
        state["target_type"] = str(closest.kind)
        state["target_has_resource"] = closest.amount > 0.0
    else:
        state["at_discoverable"] = False
        state["visible_discoverable"] = False
        state["target_id"] = "none"
        state["target_type"] = "none"
        state["target_has_resource"] = False

    # Memory-derived predicates
    known_water = any(
        m.kind is DiscoverableKind.FRESHWATER_SPRING
        for m in disc_memory.discoverables.values()
    )
    known_food = any(
        m.kind is DiscoverableKind.BERRY_BUSH
        for m in disc_memory.discoverables.values()
    )
    state["known_water"] = known_water
    state["known_food"] = known_food

    return state
