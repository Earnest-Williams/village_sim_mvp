"""Discoverable world entities (§22–29).

Discoverables are stable world objects with unique IDs that agents can
perceive, remember, interact with, and deplete/refresh.  They are stored
separately from terrain arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


# ── Discoverable kind (§23) ───────────────────────────────────────────────────


class DiscoverableKind(StrEnum):
    FRESHWATER_SPRING = "freshwater_spring"
    BERRY_BUSH = "berry_bush"


# ── Core entity (§23) ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class Discoverable:
    """A stable world entity that agents can find and exploit."""

    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int

    visible_name: str
    discovered: bool  # True once any agent has perceived it

    amount: float  # current resource stock
    max_amount: float
    regrowth_per_day: float  # added to amount each simulated day

    satisfies_need: str  # "hunger" | "thirst"
    need_delta: float  # negative means need decreases (good)
    interaction_ticks: int  # ticks consumed by one exploitation


# ── Canonical seed instances (§24, §25) ───────────────────────────────────────


def make_spring_001() -> Discoverable:
    """Create the canonical freshwater spring instance."""
    return Discoverable(
        discoverable_id="spring_001",
        kind=DiscoverableKind.FRESHWATER_SPRING,
        x=12,
        y=12,
        visible_name="freshwater spring",
        discovered=False,
        amount=9999.0,
        max_amount=9999.0,
        regrowth_per_day=0.0,
        satisfies_need="thirst",
        need_delta=-0.65,
        interaction_ticks=3,
    )


def make_berry_bush_001() -> Discoverable:
    """Create the canonical berry bush instance."""
    return Discoverable(
        discoverable_id="berry_bush_001",
        kind=DiscoverableKind.BERRY_BUSH,
        x=20,
        y=18,
        visible_name="berry bush",
        discovered=False,
        amount=4.0,
        max_amount=4.0,
        regrowth_per_day=0.5,
        satisfies_need="hunger",
        need_delta=-0.35,
        interaction_ticks=6,
    )


# ── Minimal test-only world (§30) ─────────────────────────────────────────────


@dataclass(slots=True)
class DiscoverableWorld:
    """Minimal world fixture for unit tests (no terrain arrays)."""

    width: int
    height: int
    discoverables: dict[str, Discoverable]


def make_discoverable_test_world() -> DiscoverableWorld:
    """Return a deterministic 64×64 world with spring_001 and berry_bush_001."""
    return DiscoverableWorld(
        width=64,
        height=64,
        discoverables={
            "spring_001": make_spring_001(),
            "berry_bush_001": make_berry_bush_001(),
        },
    )


# ── Protocol: any world with a discoverables dict (§26) ──────────────────────


class HasDiscoverables(Protocol):
    """Structural type: any world-like object with a discoverables dict."""

    discoverables: dict[str, Discoverable]


# ── Perception (§26) ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class DiscoverableObservation:
    """A discoverable visible in the current observation."""

    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int
    amount: float


def nearby_discoverables(
    world: HasDiscoverables,
    x: int,
    y: int,
    radius: int,
) -> list[Discoverable]:
    """Return all discoverables within Euclidean *radius* of (x, y)."""
    found: list[Discoverable] = []
    for item in world.discoverables.values():
        dx: int = item.x - x
        dy: int = item.y - y
        if dx * dx + dy * dy <= radius * radius:
            found.append(item)
    return found


def perceive_discoverables(
    world: HasDiscoverables,
    agent_x: int,
    agent_y: int,
    vision_radius: int,
) -> list[DiscoverableObservation]:
    """Return observations for all discoverables within vision radius."""
    observations: list[DiscoverableObservation] = []
    for item in nearby_discoverables(world, agent_x, agent_y, vision_radius):
        observations.append(
            DiscoverableObservation(
                discoverable_id=item.discoverable_id,
                kind=item.kind,
                x=item.x,
                y=item.y,
                amount=item.amount,
            )
        )
    return observations


# ── Memory (§27) ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DiscoverableMemory:
    """A remembered discoverable with staleness and confidence tracking."""

    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int
    last_seen_tick: int
    last_known_amount: float
    confidence: float


@dataclass(slots=True)
class DiscoverableAgentMemory:
    """ID-indexed discoverable knowledge for one agent."""

    discoverables: dict[str, DiscoverableMemory] = field(default_factory=dict)


def update_discoverable_memory(
    memory: DiscoverableAgentMemory,
    observations: list[DiscoverableObservation],
    tick: int,
) -> None:
    """Upsert discoverable memories from the current tick's observations.

    First sighting sets confidence=1.0; subsequent sightings refresh it.
    """
    for item in observations:
        memory.discoverables[item.discoverable_id] = DiscoverableMemory(
            discoverable_id=item.discoverable_id,
            kind=item.kind,
            x=item.x,
            y=item.y,
            last_seen_tick=tick,
            last_known_amount=item.amount,
            confidence=1.0,
        )


# ── Interaction (§28) ────────────────────────────────────────────────────────


@dataclass(slots=True)
class AgentNeeds:
    """Minimal needs container for discoverable interaction tests."""

    hunger: float
    thirst: float
    fatigue: float
    health: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def exploit_discoverable(
    agent_needs: AgentNeeds,
    item: Discoverable,
) -> bool:
    """Apply one interaction with a discoverable to the agent's needs.

    Returns True on success, False when the resource is depleted or the
    satisfies_need field is unrecognised.
    """
    if item.amount <= 0.0:
        return False

    if item.satisfies_need == "thirst":
        agent_needs.thirst = _clamp(agent_needs.thirst + item.need_delta)
    elif item.satisfies_need == "hunger":
        agent_needs.hunger = _clamp(agent_needs.hunger + item.need_delta)
    else:
        return False

    # Infinite sources (e.g. spring) never deplete.
    if item.max_amount < 9999.0:
        item.amount = max(0.0, item.amount - 1.0)

    return True


# ── Regrowth (§29) ───────────────────────────────────────────────────────────


def update_discoverables_daily(world: HasDiscoverables) -> None:
    """Replenish depletable discoverables once per simulated day."""
    for item in world.discoverables.values():
        if item.regrowth_per_day <= 0.0:
            continue
        item.amount = min(item.max_amount, item.amount + item.regrowth_per_day)
