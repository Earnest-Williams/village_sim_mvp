"""Agent memory model."""

from __future__ import annotations

from dataclasses import dataclass, field

from village_sim.core.config import SimConfig
from village_sim.core.types import Position, ResourceKind, ResourceSighting


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
        return max(0.0, min(1.0, self.confidence + reliability_bonus - failure_penalty - decay))


@dataclass(slots=True)
class AgentMemory:
    """All learned facts for one agent."""

    resource_memories: list[ResourceMemory] = field(default_factory=list)
    _lookup: dict[tuple[ResourceKind, int, int], int] = field(default_factory=dict)

    def observe(self, sighting: ResourceSighting, tick: int) -> bool:
        """Record a sighting. Return True when this was a new location."""

        key: tuple[ResourceKind, int, int] = _key(sighting.kind, sighting.position)
        existing_index: int | None = self._lookup.get(key)
        if existing_index is not None:
            memory: ResourceMemory = self.resource_memories[existing_index]
            memory.last_seen_tick = tick
            memory.last_amount = sighting.amount
            memory.confidence = min(1.0, max(memory.confidence, 0.50) + 0.12)
            if sighting.amount > 0.0:
                memory.failed_uses = max(0, memory.failed_uses - 1)
            return False

        self._lookup[key] = len(self.resource_memories)
        self.resource_memories.append(
            ResourceMemory(
                position=sighting.position,
                kind=sighting.kind,
                last_seen_tick=tick,
                last_amount=sighting.amount,
                confidence=0.70,
            )
        )
        return True

    def mark_success(self, kind: ResourceKind, position: Position, tick: int, amount: float) -> None:
        memory: ResourceMemory | None = self._find(kind, position)
        if memory is None:
            self._lookup[_key(kind, position)] = len(self.resource_memories)
            self.resource_memories.append(
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
        memory.last_seen_tick = tick
        memory.last_amount = amount
        memory.confidence = min(1.0, memory.confidence + 0.16)
        memory.successful_uses += 1
        memory.failed_uses = max(0, memory.failed_uses - 1)

    def mark_failure(self, kind: ResourceKind, position: Position, tick: int) -> None:
        memory: ResourceMemory | None = self._find(kind, position)
        if memory is None:
            return
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
        for memory in self.resource_memories:
            if memory.kind is not kind:
                continue
            confidence: float = memory.decayed_confidence(tick, config)
            if confidence <= 0.08:
                continue
            distance: float = current.distance_to(memory.position)
            amount_bonus: float = min(0.25, memory.last_amount * 0.12)
            score: float = confidence + amount_bonus - distance * 0.025
            if score > best_score:
                best_score = score
                best = memory
        return best

    def _find(self, kind: ResourceKind, position: Position) -> ResourceMemory | None:
        index: int | None = self._lookup.get(_key(kind, position))
        if index is None:
            return None
        return self.resource_memories[index]


def _key(kind: ResourceKind, position: Position) -> tuple[ResourceKind, int, int]:
    return (kind, position.x, position.y)
