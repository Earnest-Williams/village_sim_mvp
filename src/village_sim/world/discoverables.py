"""Discoverable world entities (§22–29).

Discoverables are stable world objects with unique IDs that agents can
perceive, remember, interact with, and deplete/refresh. They are stored
separately from terrain arrays and indexed by flat row-major cells for
constant-time spatial lookup at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from village_sim.core.types import Position

# ── Constants ─────────────────────────────────────────────────────────────────

_INFINITE_AMOUNT_THRESHOLD: float = 9999.0
"""Sources with max_amount at or above this value are treated as inexhaustible."""

# ── Discoverable kind (§23) ───────────────────────────────────────────────────


class DiscoverableKind(StrEnum):
    FRESHWATER_SPRING = "freshwater_spring"
    BERRY_BUSH = "berry_bush"
    CAVE = "cave"


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

    satisfies_need: str  # "hunger" | "thirst" | "cold_stress"
    need_delta: float  # negative means need decreases (good)
    interaction_ticks: int  # ticks consumed by one exploitation


# ── Spatial index (§26) ───────────────────────────────────────────────────────


@dataclass(slots=True)
class DiscoverableSpatialIndex:
    """Flat row-major spatial index for discoverable lookup."""

    width: int
    height: int
    cells: list[list[Discoverable]]

    def cell_index(self, x: int, y: int) -> int:
        return y * self.width + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def at(self, x: int, y: int) -> list[Discoverable]:
        if not self.in_bounds(x, y):
            return []
        return self.cells[self.cell_index(x, y)]


def build_discoverable_spatial_index(
    width: int,
    height: int,
    discoverables: dict[str, Discoverable],
) -> DiscoverableSpatialIndex:
    """Build a flat per-tile index from stable world-generation discoverables."""

    cells: list[list[Discoverable]] = [
        list[Discoverable]() for _ in range(width * height)
    ]
    for item in discoverables.values():
        if 0 <= item.x < width and 0 <= item.y < height:
            cells[item.y * width + item.x].append(item)
    return DiscoverableSpatialIndex(width=width, height=height, cells=cells)


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
        amount=_INFINITE_AMOUNT_THRESHOLD,
        max_amount=_INFINITE_AMOUNT_THRESHOLD,
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


def make_cave_001() -> Discoverable:
    """Create the canonical natural cave shelter instance."""
    return Discoverable(
        discoverable_id="cave_001",
        kind=DiscoverableKind.CAVE,
        x=8,
        y=24,
        visible_name="cave",
        discovered=False,
        amount=_INFINITE_AMOUNT_THRESHOLD,
        max_amount=_INFINITE_AMOUNT_THRESHOLD,
        regrowth_per_day=0.0,
        satisfies_need="cold_stress",
        need_delta=-0.50,
        interaction_ticks=4,
    )


# ── Minimal test-only world (§30) ─────────────────────────────────────────────


@dataclass(slots=True)
class DiscoverableWorld:
    """Minimal world fixture for unit tests (no terrain arrays)."""

    width: int
    height: int
    discoverables: dict[str, Discoverable]
    discoverable_index: DiscoverableSpatialIndex = field(init=False)

    def __post_init__(self) -> None:
        self.discoverable_index = build_discoverable_spatial_index(
            self.width,
            self.height,
            self.discoverables,
        )


def make_initial_discoverables() -> dict[str, Discoverable]:
    """Return fresh canonical discoverables for optional live-world seeding."""
    return {
        "spring_001": make_spring_001(),
        "berry_bush_001": make_berry_bush_001(),
        "cave_001": make_cave_001(),
    }


def make_discoverable_test_world() -> DiscoverableWorld:
    """Return a deterministic 64×64 world with canonical discoverables."""
    return DiscoverableWorld(
        width=64,
        height=64,
        discoverables=make_initial_discoverables(),
    )


# ── Protocol: any world with indexed discoverables (§26) ──────────────────────


class HasDiscoverables(Protocol):
    """Structural type: any world-like object with indexed discoverables."""

    width: int
    height: int
    discoverables: dict[str, Discoverable]
    discoverable_index: DiscoverableSpatialIndex


# ── Perception (§26) ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class DiscoverableObservation:
    """A discoverable visible in the current observation."""

    discoverable_id: str
    kind: DiscoverableKind
    x: int
    y: int
    amount: float


def refresh_discoverable_spatial_index(world: HasDiscoverables) -> None:
    """Rebuild the spatial index after external discoverable position changes."""

    world.discoverable_index = build_discoverable_spatial_index(
        world.width,
        world.height,
        world.discoverables,
    )


def nearby_discoverables(
    world: HasDiscoverables,
    x: int,
    y: int,
    radius: int,
) -> list[Discoverable]:
    """Return all discoverables within Euclidean *radius* of (x, y)."""

    found: list[Discoverable] = []
    if radius < 0:
        return found

    index: DiscoverableSpatialIndex = world.discoverable_index
    radius_squared: int = radius * radius
    min_y: int = max(0, y - radius)
    max_y: int = min(index.height - 1, y + radius)
    min_x: int = max(0, x - radius)
    max_x: int = min(index.width - 1, x + radius)

    for cy in range(min_y, max_y + 1):
        dy: int = cy - y
        dy_squared: int = dy * dy
        row_offset: int = cy * index.width
        for cx in range(min_x, max_x + 1):
            dx: int = cx - x
            if dx * dx + dy_squared > radius_squared:
                continue
            for item in index.cells[row_offset + cx]:
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

    discoverables: dict[str, DiscoverableMemory] = field(
        default_factory=lambda: dict[str, DiscoverableMemory]()
    )


def update_discoverable_memory(
    memory: DiscoverableAgentMemory,
    observations: list[DiscoverableObservation],
    tick: int,
) -> list[str]:
    """Upsert discoverable memories and return newly seen discoverable IDs.

    First sighting sets confidence=1.0; subsequent sightings refresh it.
    """

    newly_discovered: list[str] = []
    for item in observations:
        if item.discoverable_id not in memory.discoverables:
            newly_discovered.append(item.discoverable_id)
        memory.discoverables[item.discoverable_id] = DiscoverableMemory(
            discoverable_id=item.discoverable_id,
            kind=item.kind,
            x=item.x,
            y=item.y,
            last_seen_tick=tick,
            last_known_amount=item.amount,
            confidence=1.0,
        )
    return newly_discovered


# ── Interaction (§28) ────────────────────────────────────────────────────────


class HasNeeds(Protocol):
    """Structural type for objects with mutable survival need fields."""

    hunger: float
    thirst: float
    fatigue: float
    cold_stress: float
    health: float


@dataclass(slots=True)
class AgentNeeds:
    """Minimal needs container for discoverable interaction tests."""

    hunger: float
    thirst: float
    fatigue: float
    health: float
    cold_stress: float = 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def discoverable_at_or_adjacent(
    world: HasDiscoverables,
    x: int,
    y: int,
) -> Discoverable | None:
    """Return the closest discoverable within Chebyshev distance one."""

    index: DiscoverableSpatialIndex = world.discoverable_index
    best: Discoverable | None = None
    best_distance: int = 2
    min_y: int = max(0, y - 1)
    max_y: int = min(index.height - 1, y + 1)
    min_x: int = max(0, x - 1)
    max_x: int = min(index.width - 1, x + 1)

    for cy in range(min_y, max_y + 1):
        row_offset: int = cy * index.width
        for cx in range(min_x, max_x + 1):
            distance: int = max(abs(cx - x), abs(cy - y))
            if distance > best_distance:
                continue
            for item in index.cells[row_offset + cx]:
                if distance < best_distance:
                    best = item
                    best_distance = distance
    return best


def exploit_discoverable(
    agent_needs: HasNeeds,
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
    elif item.satisfies_need == "cold_stress":
        agent_needs.cold_stress = _clamp(agent_needs.cold_stress + item.need_delta)
    else:
        return False

    # Infinite sources (e.g. spring) never deplete.
    if item.max_amount < _INFINITE_AMOUNT_THRESHOLD:
        item.amount = max(0.0, item.amount - 1.0)

    return True


def exploit_nearby_discoverable(
    world: HasDiscoverables,
    agent_needs: HasNeeds,
    position: Position,
) -> str | None:
    """Exploit an adjacent discoverable and return its ID on success."""

    item = discoverable_at_or_adjacent(world, position.x, position.y)
    if item is None:
        return None
    if not exploit_discoverable(agent_needs, item):
        return None
    return item.discoverable_id


# ── Regrowth (§29) ───────────────────────────────────────────────────────────


def update_discoverables_daily(world: HasDiscoverables) -> None:
    """Replenish depletable discoverables once per simulated day."""

    for item in world.discoverables.values():
        if item.regrowth_per_day <= 0.0:
            continue
        item.amount = min(item.max_amount, item.amount + item.regrowth_per_day)
