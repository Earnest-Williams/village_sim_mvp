"""Fixed-capacity agent memory model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, overload

import polars as pl

from village_sim.core.config import SimConfig
from village_sim.core.types import Position, ResourceKind, ResourceSighting

# Re-export discoverable memory types so orchestrator/symbolic.py can import
# them from agent.memory without circular dependencies.
from village_sim.world.discoverables import (
    DiscoverableAgentMemory,
    DiscoverableMemory,
    update_discoverable_memory,
)

__all__ = [
    "AgentMemory",
    "DiscoverableAgentMemory",
    "DiscoverableMemory",
    "ResourceMemory",
    "update_discoverable_memory",
]

MEMORY_AGENT_ID = "agent_id"
MEMORY_KIND = "kind"
MEMORY_X = "x"
MEMORY_Y = "y"
MEMORY_LAST_SEEN_TICK = "last_seen_tick"
MEMORY_LAST_AMOUNT = "last_amount"
MEMORY_CONFIDENCE = "confidence"
MEMORY_SUCCESSFUL_USES = "successful_uses"
MEMORY_FAILED_USES = "failed_uses"
MEMORY_SEARCH_RADIUS = "search_radius"
DEFAULT_MEMORY_CAPACITY = 512

MEMORY_SCHEMA: dict[str, Any] = {
    MEMORY_AGENT_ID: pl.Int64,
    MEMORY_KIND: pl.String,
    MEMORY_X: pl.Int64,
    MEMORY_Y: pl.Int64,
    MEMORY_LAST_SEEN_TICK: pl.Int64,
    MEMORY_LAST_AMOUNT: pl.Float64,
    MEMORY_CONFIDENCE: pl.Float64,
    MEMORY_SUCCESSFUL_USES: pl.Int64,
    MEMORY_FAILED_USES: pl.Int64,
    MEMORY_SEARCH_RADIUS: pl.Int64,
}


@dataclass(slots=True)
class ResourceMemory:
    """A remembered resource location with confidence and staleness."""

    position: Position
    kind: ResourceKind
    last_seen_tick: int
    last_amount: float
    confidence: float
    successful_uses: int = 0
    failed_uses: int = 0
    search_radius: int = 3

    def age_ticks(self, tick: int) -> int:
        return max(0, tick - self.last_seen_tick)

    def decayed_confidence(self, tick: int, config: SimConfig) -> float:
        age_days: float = self.age_ticks(tick) / float(config.ticks_per_day)
        if self.kind is ResourceKind.WATER:
            decay: float = age_days * config.water_memory_decay_per_day
        else:
            decay = age_days * config.food_memory_decay_per_day
        reliability_bonus: float = min(0.25, self.successful_uses * 0.04)
        failure_penalty: float = min(0.45, self.failed_uses * 0.08)
        confidence: float = (
            self.confidence + reliability_bonus - failure_penalty - decay
        )
        return max(0.0, min(1.0, confidence))


class ResourceMemoryView(Sequence[ResourceMemory]):
    """Compatibility sequence backed by AgentMemory's Python list."""

    def __init__(self, owner: AgentMemory) -> None:
        self._owner = owner

    def __len__(self) -> int:
        return self._owner.memory_count()

    @overload
    def __getitem__(self, index: int) -> ResourceMemory: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ResourceMemory]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ResourceMemory | Sequence[ResourceMemory]:
        return self._owner.memory_at(index)

    def __iter__(self) -> Iterator[ResourceMemory]:
        return self._owner.iter_memories()

    def append(self, memory: ResourceMemory) -> None:
        self._owner.append_resource_memory(memory)


@dataclass(slots=True)
class AgentMemory:
    """All learned facts for one agent, stored in a fixed-capacity list."""

    agent_id: int = 1
    capacity: int = DEFAULT_MEMORY_CAPACITY
    _memories: list[ResourceMemory] = field(
        default_factory=lambda: list[ResourceMemory]()
    )
    _lookup: dict[tuple[ResourceKind, int, int], int] = field(
        default_factory=lambda: dict[tuple[ResourceKind, int, int], int]()
    )

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("memory capacity must be positive")
        if len(self._memories) > self.capacity:
            raise ValueError("initial memories cannot exceed memory capacity")
        self._rebuild_lookup()

    def memory_count(self) -> int:
        return len(self._memories)

    def memory_at(
        self, index: int | slice
    ) -> ResourceMemory | Sequence[ResourceMemory]:
        return self._memories[index]

    def iter_memories(self) -> Iterator[ResourceMemory]:
        return iter(self._memories)

    @property
    def resource_memories(self) -> ResourceMemoryView:
        return ResourceMemoryView(self)

    def export_to_dataframe(self) -> pl.DataFrame:
        """Build a report-time Polars view of resource memory."""

        rows: list[dict[str, int | float | str]] = []
        for memory in self._memories:
            rows.append(
                {
                    MEMORY_AGENT_ID: self.agent_id,
                    MEMORY_KIND: memory.kind.value,
                    MEMORY_X: memory.position.x,
                    MEMORY_Y: memory.position.y,
                    MEMORY_LAST_SEEN_TICK: memory.last_seen_tick,
                    MEMORY_LAST_AMOUNT: memory.last_amount,
                    MEMORY_CONFIDENCE: memory.confidence,
                    MEMORY_SUCCESSFUL_USES: memory.successful_uses,
                    MEMORY_FAILED_USES: memory.failed_uses,
                    MEMORY_SEARCH_RADIUS: memory.search_radius,
                }
            )
        return pl.DataFrame(rows, schema=MEMORY_SCHEMA, orient="row")

    def observe(self, sighting: ResourceSighting, tick: int) -> bool:
        """Record a sighting. Return True when this was a new location."""

        return self.observe_resource(
            kind=sighting.kind,
            position=sighting.position,
            amount=sighting.amount,
            tick=tick,
        )

    def observe_resource(
        self, kind: ResourceKind, position: Position, amount: float, tick: int
    ) -> bool:
        """Record a resource sighting without allocating a ResourceSighting object."""

        key: tuple[ResourceKind, int, int] = _key(kind, position)
        index: int | None = self._lookup.get(key)
        if index is not None:
            memory: ResourceMemory = self._memories[index]
            memory.last_seen_tick = tick
            memory.last_amount = amount
            memory.confidence = min(1.0, max(memory.confidence, 0.50) + 0.12)
            if amount > 0.0:
                memory.failed_uses = max(0, memory.failed_uses - 1)
            return False

        self.append_resource_memory(
            ResourceMemory(
                position=position,
                kind=kind,
                last_seen_tick=tick,
                last_amount=amount,
                confidence=0.70,
            )
        )
        return True

    def mark_success(
        self, kind: ResourceKind, position: Position, tick: int, amount: float
    ) -> None:
        index: int | None = self._lookup.get(_key(kind, position))
        if index is None:
            self.append_resource_memory(
                ResourceMemory(
                    position=position,
                    kind=kind,
                    last_seen_tick=tick,
                    last_amount=amount,
                    confidence=0.80,
                    successful_uses=1,
                )
            )
            return
        memory: ResourceMemory = self._memories[index]
        memory.last_seen_tick = tick
        memory.last_amount = amount
        memory.confidence = min(1.0, memory.confidence + 0.16)
        memory.successful_uses += 1
        memory.failed_uses = max(0, memory.failed_uses - 1)

    def mark_failure(self, kind: ResourceKind, position: Position, tick: int) -> None:
        index: int | None = self._lookup.get(_key(kind, position))
        if index is None:
            return
        memory: ResourceMemory = self._memories[index]
        memory.last_seen_tick = tick
        memory.last_amount = 0.0
        memory.confidence = max(0.0, memory.confidence - 0.24)
        memory.failed_uses += 1
        memory.search_radius = min(10, memory.search_radius + 1)

    def best_memory(
        self,
        kind: ResourceKind,
        current: Position,
        tick: int,
        config: SimConfig,
    ) -> ResourceMemory | None:
        best: ResourceMemory | None = None
        best_score: float = -1_000_000.0
        for memory in self._memories:
            if memory.kind is not kind:
                continue
            decayed_confidence: float = memory.decayed_confidence(tick, config)
            if decayed_confidence <= 0.08:
                continue
            amount_bonus: float = min(0.25, max(0.0, memory.last_amount * 0.12))
            distance_penalty: float = current.distance_to(memory.position) * 0.025
            score: float = decayed_confidence + amount_bonus - distance_penalty
            if _is_better_memory(score, memory, best_score, best):
                best = memory
                best_score = score
        return best

    def append_resource_memory(self, memory: ResourceMemory) -> None:
        key: tuple[ResourceKind, int, int] = _key(memory.kind, memory.position)
        existing_index: int | None = self._lookup.get(key)
        if existing_index is not None:
            self._memories[existing_index] = memory
            return
        if len(self._memories) >= self.capacity:
            removed: ResourceMemory = self._memories.pop(0)
            del self._lookup[_key(removed.kind, removed.position)]
            self._rebuild_lookup()
        self._lookup[key] = len(self._memories)
        self._memories.append(memory)

    def _rebuild_lookup(self) -> None:
        self._lookup.clear()
        for index, memory in enumerate(self._memories):
            key: tuple[ResourceKind, int, int] = _key(memory.kind, memory.position)
            if key in self._lookup:
                raise ValueError("initial memories cannot contain duplicate locations")
            self._lookup[key] = index


def _key(kind: ResourceKind, position: Position) -> tuple[ResourceKind, int, int]:
    return (kind, position.x, position.y)


def _is_better_memory(
    score: float,
    memory: ResourceMemory,
    best_score: float,
    best: ResourceMemory | None,
) -> bool:
    if best is None:
        return True
    if score > best_score:
        return True
    if score < best_score:
        return False
    if memory.position.x < best.position.x:
        return True
    if memory.position.x > best.position.x:
        return False
    return memory.position.y < best.position.y
