"""Symbolic state extraction (§7, §8).

Raw simulation state is converted into a flat dictionary of symbolic facts
for use by the orchestrator and GOAP planner.
"""

from __future__ import annotations

from village_sim.agent.memory import DiscoverableAgentMemory
from village_sim.agent.perception import Observation
from village_sim.agent.state import AgentState
from village_sim.core.time import SimClock
from village_sim.world.discoverables import DiscoverableKind, DiscoverableMemory

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


def bucket_distance(distance: int) -> str:
    """Convert a Chebyshev target distance to a symbolic bucket label."""
    if distance <= 1:
        return "at"
    if distance <= 5:
        return "near"
    if distance <= 16:
        return "medium"
    return "far"


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
        "cold_stress_bucket": bucket_need(agent.cold_stress),
        "health_low": agent.health < 0.30,
        "is_daylight": clock.is_daylight,
        "is_night": clock.is_night,
    }

    # Closest discoverable in current observation (§7 example predicates)
    if observation.discoverables:
        closest = min(
            observation.discoverables,
            key=lambda d: (d.x - agent.position.x) ** 2 + (d.y - agent.position.y) ** 2,
        )
        distance = max(
            abs(closest.x - agent.position.x),
            abs(closest.y - agent.position.y),
        )
        at_disc = distance <= 1
        state["at_discoverable"] = at_disc
        state["visible_discoverable"] = True
        state["target_id"] = closest.discoverable_id
        state["target_type"] = str(closest.kind)
        state["target_has_resource"] = closest.amount > 0.0
        state["has_target_location"] = True
        state["target_known_x"] = closest.x
        state["target_known_y"] = closest.y
        state["distance_to_target_bucket"] = bucket_distance(distance)
        state["at_known_target"] = at_disc
    else:
        _apply_memory_target_facts(state, agent, disc_memory)

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
    known_shelter = any(
        m.kind is DiscoverableKind.CAVE for m in disc_memory.discoverables.values()
    )
    state["known_food"] = known_food
    state["known_shelter"] = known_shelter
    state["at_shelter"] = (
        state.get("at_known_target") is True and state.get("target_type") == "cave"
    )

    return state


def _apply_memory_target_facts(
    state: SymbolicState,
    agent: AgentState,
    disc_memory: DiscoverableAgentMemory,
) -> None:
    target = _choose_memory_target(agent, disc_memory)
    if target is None:
        state["at_discoverable"] = False
        state["visible_discoverable"] = False
        state["target_id"] = "none"
        state["target_type"] = "none"
        state["target_has_resource"] = False
        state["has_target_location"] = False
        state["target_known_x"] = -1
        state["target_known_y"] = -1
        state["distance_to_target_bucket"] = "far"
        state["at_known_target"] = False
        return

    distance = max(abs(target.x - agent.position.x), abs(target.y - agent.position.y))
    at_target = distance <= 1
    state["at_discoverable"] = False
    state["visible_discoverable"] = False
    state["target_id"] = target.discoverable_id
    state["target_type"] = str(target.kind)
    state["target_has_resource"] = target.last_known_amount > 0.0
    state["has_target_location"] = True
    state["target_known_x"] = target.x
    state["target_known_y"] = target.y
    state["distance_to_target_bucket"] = bucket_distance(distance)
    state["at_known_target"] = at_target


def _choose_memory_target(
    agent: AgentState,
    disc_memory: DiscoverableAgentMemory,
) -> DiscoverableMemory | None:
    preferred_kinds: list[DiscoverableKind] = []
    if agent.thirst >= agent.hunger:
        preferred_kinds.extend(
            [DiscoverableKind.FRESHWATER_SPRING, DiscoverableKind.BERRY_BUSH]
        )
    else:
        preferred_kinds.extend(
            [DiscoverableKind.BERRY_BUSH, DiscoverableKind.FRESHWATER_SPRING]
        )
    if agent.cold_stress >= 0.60:
        preferred_kinds.insert(0, DiscoverableKind.CAVE)
    else:
        preferred_kinds.append(DiscoverableKind.CAVE)

    for kind in preferred_kinds:
        target = _best_memory_of_kind(disc_memory, kind)
        if target is not None:
            return target
    return None


def _best_memory_of_kind(
    disc_memory: DiscoverableAgentMemory,
    kind: DiscoverableKind,
) -> DiscoverableMemory | None:
    matches = [m for m in disc_memory.discoverables.values() if m.kind is kind]
    if not matches:
        return None
    matches.sort(key=lambda m: (-m.confidence, -m.last_seen_tick, m.discoverable_id))
    return matches[0]
