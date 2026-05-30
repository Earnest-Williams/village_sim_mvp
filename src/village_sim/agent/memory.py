"""Centralized Polars-backed agent resource memory."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import overload

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
    "GlobalMemory",
    "ResourceMemory",
    "update_discoverable_memory",
]

MEMORY_AGENT_ID = "agent_id"
MEMORY_KIND = "kind"
MEMORY_X = "x"
MEMORY_Y = "y"
MEMORY_CONFIDENCE = "confidence"
MEMORY_LAST_SEEN = "last_seen"
MEMORY_LAST_SEEN_TICK = "last_seen_tick"
MEMORY_LAST_AMOUNT = "last_amount"
MEMORY_SUCCESSFUL_USES = "successful_uses"
MEMORY_FAILED_USES = "failed_uses"
MEMORY_SEARCH_RADIUS = "search_radius"
MEMORY_ORDER = "_order"
DEFAULT_MEMORY_CAPACITY = 512
_KIND_TO_ID: dict[ResourceKind, int] = {
    ResourceKind.WATER: 1,
    ResourceKind.FOOD: 2,
}
_ID_TO_KIND: dict[int, ResourceKind] = {
    value: key for key, value in _KIND_TO_ID.items()
}

_GLOBAL_MEMORY_SCHEMA_MAP = {
    MEMORY_AGENT_ID: pl.Int32,
    MEMORY_KIND: pl.Int8,
    MEMORY_X: pl.Int32,
    MEMORY_Y: pl.Int32,
    MEMORY_CONFIDENCE: pl.Float32,
    MEMORY_LAST_SEEN: pl.Int64,
}
GLOBAL_MEMORY_SCHEMA: pl.Schema = pl.Schema(_GLOBAL_MEMORY_SCHEMA_MAP)

_MEMORY_SCHEMA_MAP = {
    **_GLOBAL_MEMORY_SCHEMA_MAP,
    MEMORY_LAST_AMOUNT: pl.Float32,
    MEMORY_SUCCESSFUL_USES: pl.Int32,
    MEMORY_FAILED_USES: pl.Int32,
    MEMORY_SEARCH_RADIUS: pl.Int32,
    MEMORY_ORDER: pl.Int64,
}
MEMORY_SCHEMA: pl.Schema = pl.Schema(_MEMORY_SCHEMA_MAP)

EXPORT_MEMORY_SCHEMA: pl.Schema = pl.Schema(
    {
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
)


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


@dataclass(slots=True)
class GlobalMemory:
    """Global resource memory table for all agents.

    Mutations are staged into a compact pending DataFrame and applied through
    one Polars concat/deduplicate pass when ``flush_pending`` is called.
    """

    capacity_per_agent: int = DEFAULT_MEMORY_CAPACITY
    frame: pl.DataFrame = field(default_factory=lambda: _empty_memory_frame())
    _pending_dict: dict[
        tuple[int, int, int, int], tuple[ResourceMemory, int]
    ] = field(default_factory=dict)
    _next_order: int = 0

    def __post_init__(self) -> None:
        if self.capacity_per_agent <= 0:
            raise ValueError("memory capacity must be positive")
        self.frame = self.frame.cast(MEMORY_SCHEMA)
        if not self.frame.is_empty():
            max_order: int = _require_int(self.frame.get_column(MEMORY_ORDER).max())
            self._next_order = max_order + 1

    def memory_count(self, agent_id: int) -> int:
        return self.frame.filter(pl.col(MEMORY_AGENT_ID) == agent_id).height

    def queue_observation(
        self,
        *,
        agent_id: int,
        kind: ResourceKind,
        position: Position,
        amount: float,
        tick: int,
    ) -> bool:
        existing: ResourceMemory | None = self._get_staged_memory(
            agent_id, kind, position
        )
        if existing is None:
            memory = ResourceMemory(
                position=position,
                kind=kind,
                last_seen_tick=tick,
                last_amount=amount,
                confidence=0.70,
            )
            is_new = True
        else:
            memory = ResourceMemory(
                position=position,
                kind=kind,
                last_seen_tick=tick,
                last_amount=amount,
                confidence=min(1.0, max(existing.confidence, 0.50) + 0.12),
                successful_uses=existing.successful_uses,
                failed_uses=(
                    max(0, existing.failed_uses - 1)
                    if amount > 0.0
                    else existing.failed_uses
                ),
                search_radius=existing.search_radius,
            )
            is_new = False
        self.queue_memory(agent_id, memory)
        return is_new

    def queue_memory(self, agent_id: int, memory: ResourceMemory) -> None:
        key = _memory_key(agent_id, memory.kind, memory.position)
        pending = self._pending_dict.get(key)
        if pending is not None:
            order = pending[1]
        else:
            order = self._existing_order(agent_id, memory.kind, memory.position)
            if order < 0:
                order = self._next_order
                self._next_order += 1
        self._pending_dict[key] = (memory, order)

    def flush_pending(self) -> None:
        if not self._pending_dict:
            return
        rows = [
            {
                MEMORY_AGENT_ID: agent_id,
                MEMORY_KIND: kind_id,
                MEMORY_X: x,
                MEMORY_Y: y,
                MEMORY_CONFIDENCE: memory.confidence,
                MEMORY_LAST_SEEN: memory.last_seen_tick,
                MEMORY_LAST_AMOUNT: memory.last_amount,
                MEMORY_SUCCESSFUL_USES: memory.successful_uses,
                MEMORY_FAILED_USES: memory.failed_uses,
                MEMORY_SEARCH_RADIUS: memory.search_radius,
                MEMORY_ORDER: order,
            }
            for (agent_id, kind_id, x, y), (memory, order) in self._pending_dict.items()
        ]
        pending_df = pl.DataFrame(rows, schema=MEMORY_SCHEMA)
        combined = pl.concat([self.frame, pending_df], how="vertical")
        unique = combined.unique(
            subset=[MEMORY_AGENT_ID, MEMORY_KIND, MEMORY_X, MEMORY_Y],
            keep="last",
            maintain_order=True,
        )
        self.frame = self._enforce_capacity(unique)
        self._pending_dict.clear()

    def get_memory(
        self, agent_id: int, kind: ResourceKind, position: Position
    ) -> ResourceMemory | None:
        matches = self.frame.filter(
            (pl.col(MEMORY_AGENT_ID) == agent_id)
            & (pl.col(MEMORY_KIND) == _kind_id(kind))
            & (pl.col(MEMORY_X) == position.x)
            & (pl.col(MEMORY_Y) == position.y)
        )
        if matches.is_empty():
            return None
        return _row_to_memory(matches.row(0, named=True))

    def best_memory(
        self,
        *,
        agent_id: int,
        kind: ResourceKind,
        current: Position,
        tick: int,
        config: SimConfig,
    ) -> ResourceMemory | None:
        kind_id = _kind_id(kind)
        if kind is ResourceKind.WATER:
            decay_per_day = config.water_memory_decay_per_day
        else:
            decay_per_day = config.food_memory_decay_per_day
        age_days = (pl.lit(tick, dtype=pl.Int64) - pl.col(MEMORY_LAST_SEEN)).clip(
            0, None
        ).cast(pl.Float32) / float(config.ticks_per_day)
        decayed = (
            pl.col(MEMORY_CONFIDENCE)
            + (pl.col(MEMORY_SUCCESSFUL_USES).cast(pl.Float32) * 0.04).clip(0.0, 0.25)
            - (pl.col(MEMORY_FAILED_USES).cast(pl.Float32) * 0.08).clip(0.0, 0.45)
            - (age_days * decay_per_day)
        ).clip(0.0, 1.0)
        distance = (
            (pl.col(MEMORY_X) - current.x).pow(2)
            + (pl.col(MEMORY_Y) - current.y).pow(2)
        ).sqrt()
        scored = (
            self.frame.filter(
                (pl.col(MEMORY_AGENT_ID) == agent_id) & (pl.col(MEMORY_KIND) == kind_id)
            )
            .with_columns(
                decayed.alias("_decayed_confidence"),
                (
                    decayed
                    + (pl.col(MEMORY_LAST_AMOUNT).clip(0.0, None) * 0.12).clip(
                        0.0, 0.25
                    )
                    - (distance * 0.025)
                ).alias("_score"),
            )
            .filter(pl.col("_decayed_confidence") > 0.08)
            .sort(
                ["_score", MEMORY_X, MEMORY_Y],
                descending=[True, False, False],
            )
        )
        if scored.is_empty():
            return None
        return _row_to_memory(scored.row(0, named=True))

    def memories_for_agent(self, agent_id: int) -> pl.DataFrame:
        return self.frame.filter(pl.col(MEMORY_AGENT_ID) == agent_id).sort(MEMORY_ORDER)

    def export_agent(self, agent_id: int) -> pl.DataFrame:
        frame = self.memories_for_agent(agent_id)
        if frame.is_empty():
            return pl.DataFrame(schema=EXPORT_MEMORY_SCHEMA)
        exported = frame.with_columns(
            pl.col(MEMORY_KIND)
            .replace_strict({1: "water", 2: "food"}, return_dtype=pl.String)
            .alias(MEMORY_KIND),
            pl.col(MEMORY_LAST_SEEN).alias(MEMORY_LAST_SEEN_TICK),
        ).select(EXPORT_MEMORY_SCHEMA.keys())
        return exported.cast(EXPORT_MEMORY_SCHEMA)

    def _get_staged_memory(
        self, agent_id: int, kind: ResourceKind, position: Position
    ) -> ResourceMemory | None:
        pending = self._pending_dict.get(_memory_key(agent_id, kind, position))
        if pending is not None:
            return pending[0]
        return self.get_memory(agent_id, kind, position)

    def _existing_order(
        self, agent_id: int, kind: ResourceKind, position: Position
    ) -> int:
        pending = self._pending_dict.get(_memory_key(agent_id, kind, position))
        if pending is not None:
            return pending[1]
        matches = self.frame.filter(
            (pl.col(MEMORY_AGENT_ID) == agent_id)
            & (pl.col(MEMORY_KIND) == _kind_id(kind))
            & (pl.col(MEMORY_X) == position.x)
            & (pl.col(MEMORY_Y) == position.y)
        )
        if matches.is_empty():
            return -1
        return _require_int(matches.get_column(MEMORY_ORDER)[-1])

    def _enforce_capacity(self, frame: pl.DataFrame) -> pl.DataFrame:
        ranked = frame.sort([MEMORY_AGENT_ID, MEMORY_ORDER]).with_columns(
            pl.int_range(pl.len()).over(MEMORY_AGENT_ID).alias("_rank"),
            pl.len().over(MEMORY_AGENT_ID).cast(pl.Int64).alias("_agent_count"),
        )
        overflow = pl.col("_agent_count") - self.capacity_per_agent
        first_kept = pl.when(overflow > 0).then(overflow).otherwise(0)
        limited = ranked.filter(pl.col("_rank") >= first_kept)
        return limited.drop(["_rank", "_agent_count"]).sort(
            [MEMORY_AGENT_ID, MEMORY_ORDER]
        )


class ResourceMemoryView(Sequence[ResourceMemory]):
    """Compatibility sequence backed by AgentMemory's Polars table."""

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
    """Compatibility facade over the global Polars resource memory table."""

    agent_id: int = 1
    capacity: int = DEFAULT_MEMORY_CAPACITY
    global_memory: GlobalMemory | None = None

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("memory capacity must be positive")
        if self.global_memory is None:
            self.global_memory = GlobalMemory(capacity_per_agent=self.capacity)
        elif self.global_memory.capacity_per_agent != self.capacity:
            self.global_memory.capacity_per_agent = self.capacity

    def memory_count(self) -> int:
        return self._global.memory_count(self.agent_id)

    def memory_at(
        self, index: int | slice
    ) -> ResourceMemory | Sequence[ResourceMemory]:
        frame = self._global.memories_for_agent(self.agent_id)
        if isinstance(index, slice):
            rows = frame.slice(
                0 if index.start is None else index.start,
                None if index.stop is None else index.stop - (index.start or 0),
            ).iter_rows(named=True)
            return [_row_to_memory(row) for row in rows]
        if index < 0:
            index = frame.height + index
        if index < 0 or index >= frame.height:
            raise IndexError("resource memory index out of range")
        return _row_to_memory(frame.row(index, named=True))

    def iter_memories(self) -> Iterator[ResourceMemory]:
        for row in self._global.memories_for_agent(self.agent_id).iter_rows(named=True):
            yield _row_to_memory(row)

    @property
    def resource_memories(self) -> ResourceMemoryView:
        return ResourceMemoryView(self)

    @property
    def dataframe(self) -> pl.DataFrame:
        return self._global.memories_for_agent(self.agent_id)

    def export_to_dataframe(self) -> pl.DataFrame:
        """Build a report-time Polars view of resource memory."""

        return self._global.export_agent(self.agent_id)

    def flush_pending(self) -> None:
        self._global.flush_pending()

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

        is_new = self._global.queue_observation(
            agent_id=self.agent_id,
            kind=kind,
            position=position,
            amount=amount,
            tick=tick,
        )
        self._global.flush_pending()
        return is_new

    def queue_resource_observation(
        self, kind: ResourceKind, position: Position, amount: float, tick: int
    ) -> bool:
        return self._global.queue_observation(
            agent_id=self.agent_id,
            kind=kind,
            position=position,
            amount=amount,
            tick=tick,
        )

    def mark_success(
        self, kind: ResourceKind, position: Position, tick: int, amount: float
    ) -> None:
        existing = self._global.get_memory(self.agent_id, kind, position)
        if existing is None:
            memory = ResourceMemory(
                position=position,
                kind=kind,
                last_seen_tick=tick,
                last_amount=amount,
                confidence=0.80,
                successful_uses=1,
            )
        else:
            memory = ResourceMemory(
                position=position,
                kind=kind,
                last_seen_tick=tick,
                last_amount=amount,
                confidence=min(1.0, existing.confidence + 0.16),
                successful_uses=existing.successful_uses + 1,
                failed_uses=max(0, existing.failed_uses - 1),
                search_radius=existing.search_radius,
            )
        self._global.queue_memory(self.agent_id, memory)
        self._global.flush_pending()

    def mark_failure(self, kind: ResourceKind, position: Position, tick: int) -> None:
        existing = self._global.get_memory(self.agent_id, kind, position)
        if existing is None:
            return
        memory = ResourceMemory(
            position=position,
            kind=kind,
            last_seen_tick=tick,
            last_amount=0.0,
            confidence=max(0.0, existing.confidence - 0.24),
            successful_uses=existing.successful_uses,
            failed_uses=existing.failed_uses + 1,
            search_radius=min(10, existing.search_radius + 1),
        )
        self._global.queue_memory(self.agent_id, memory)
        self._global.flush_pending()

    def best_memory(
        self,
        kind: ResourceKind,
        current: Position,
        tick: int,
        config: SimConfig,
    ) -> ResourceMemory | None:
        return self._global.best_memory(
            agent_id=self.agent_id,
            kind=kind,
            current=current,
            tick=tick,
            config=config,
        )

    def append_resource_memory(self, memory: ResourceMemory) -> None:
        self._global.queue_memory(self.agent_id, memory)
        self._global.flush_pending()

    def get_memory(
        self, kind: ResourceKind, position: Position
    ) -> ResourceMemory | None:
        return self._global.get_memory(self.agent_id, kind, position)

    @property
    def _global(self) -> GlobalMemory:
        if self.global_memory is None:
            raise RuntimeError("agent memory was not initialized")
        return self.global_memory


def _empty_memory_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=MEMORY_SCHEMA)


def _kind_id(kind: ResourceKind) -> int:
    return _KIND_TO_ID[kind]


def _memory_key(
    agent_id: int, kind: ResourceKind, position: Position
) -> tuple[int, int, int, int]:
    return (agent_id, _kind_id(kind), position.x, position.y)


def _memory_rows(
    agent_id: int, memories: Sequence[ResourceMemory], orders: Sequence[int]
) -> pl.DataFrame:
    if len(memories) != len(orders):
        raise ValueError("memory and order batches must have equal length")
    rows: list[dict[str, int | float]] = [
        {
            MEMORY_AGENT_ID: agent_id,
            MEMORY_KIND: _kind_id(memory.kind),
            MEMORY_X: memory.position.x,
            MEMORY_Y: memory.position.y,
            MEMORY_CONFIDENCE: memory.confidence,
            MEMORY_LAST_SEEN: memory.last_seen_tick,
            MEMORY_LAST_AMOUNT: memory.last_amount,
            MEMORY_SUCCESSFUL_USES: memory.successful_uses,
            MEMORY_FAILED_USES: memory.failed_uses,
            MEMORY_SEARCH_RADIUS: memory.search_radius,
            MEMORY_ORDER: order,
        }
        for memory, order in zip(memories, orders, strict=True)
    ]
    return pl.DataFrame(rows, schema=MEMORY_SCHEMA, orient="row")


def _row_to_memory(row: dict[str, object]) -> ResourceMemory:
    kind_id = _require_int(row[MEMORY_KIND])
    return ResourceMemory(
        position=Position(
            x=_require_int(row[MEMORY_X]),
            y=_require_int(row[MEMORY_Y]),
        ),
        kind=_ID_TO_KIND[kind_id],
        last_seen_tick=_require_int(row[MEMORY_LAST_SEEN]),
        last_amount=_require_float(row[MEMORY_LAST_AMOUNT]),
        confidence=_require_float(row[MEMORY_CONFIDENCE]),
        successful_uses=_require_int(row[MEMORY_SUCCESSFUL_USES]),
        failed_uses=_require_int(row[MEMORY_FAILED_USES]),
        search_radius=_require_int(row[MEMORY_SEARCH_RADIUS]),
    )


def _require_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("memory scalar must be numeric")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError("memory scalar must be numeric")


def _require_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("memory scalar must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError("memory scalar must be numeric")
