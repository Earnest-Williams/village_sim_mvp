"""Columnar agent memory model."""

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
        return max(
            0.0, min(1.0, self.confidence + reliability_bonus - failure_penalty - decay)
        )


class ResourceMemoryView(Sequence[ResourceMemory]):
    """Compatibility sequence backed by AgentMemory's Polars DataFrame."""

    def __init__(self, owner: AgentMemory) -> None:
        self._owner = owner

    def __len__(self) -> int:
        return self._owner.frame.height

    @overload
    def __getitem__(self, index: int) -> ResourceMemory: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ResourceMemory]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ResourceMemory | Sequence[ResourceMemory]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[item] for item in range(start, stop, step)]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("resource memory index out of range")
        row: pl.DataFrame = self._owner.frame.slice(index, 1)
        return _resource_memory_from_frame(row)

    def __iter__(self) -> Iterator[ResourceMemory]:
        for index in range(len(self)):
            item = self[index]
            if isinstance(item, ResourceMemory):
                yield item

    def append(self, memory: ResourceMemory) -> None:
        self._owner._append_memory(memory)


@dataclass(slots=True)
class AgentMemory:
    """All learned facts for one agent, stored in a Polars DataFrame."""

    agent_id: int = 1
    frame: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=MEMORY_SCHEMA)
    )
    _lookup: dict[tuple[ResourceKind, int, int], bool] = field(default_factory=dict)

    @property
    def resource_memories(self) -> ResourceMemoryView:
        return ResourceMemoryView(self)

    def observe(self, sighting: ResourceSighting, tick: int) -> bool:
        """Record a sighting. Return True when this was a new location."""

        key: tuple[ResourceKind, int, int] = _key(sighting.kind, sighting.position)
        if key in self._lookup:
            amount_positive: bool = sighting.amount > 0.0
            failure_expr: pl.Expr = pl.col(MEMORY_FAILED_USES)
            if amount_positive:
                failure_expr = (pl.col(MEMORY_FAILED_USES) - 1).clip(0, None)
            self.frame = self.frame.with_columns(
                pl.when(
                    _memory_key_expr(self.agent_id, sighting.kind, sighting.position)
                )
                .then(tick)
                .otherwise(pl.col(MEMORY_LAST_SEEN_TICK))
                .alias(MEMORY_LAST_SEEN_TICK),
                pl.when(
                    _memory_key_expr(self.agent_id, sighting.kind, sighting.position)
                )
                .then(sighting.amount)
                .otherwise(pl.col(MEMORY_LAST_AMOUNT))
                .alias(MEMORY_LAST_AMOUNT),
                pl.when(
                    _memory_key_expr(self.agent_id, sighting.kind, sighting.position)
                )
                .then(
                    (
                        pl.max_horizontal(pl.col(MEMORY_CONFIDENCE), pl.lit(0.50))
                        + 0.12
                    ).clip(0.0, 1.0)
                )
                .otherwise(pl.col(MEMORY_CONFIDENCE))
                .alias(MEMORY_CONFIDENCE),
                pl.when(
                    _memory_key_expr(self.agent_id, sighting.kind, sighting.position)
                )
                .then(failure_expr)
                .otherwise(pl.col(MEMORY_FAILED_USES))
                .alias(MEMORY_FAILED_USES),
            )
            return False

        self._append_memory(
            ResourceMemory(
                position=sighting.position,
                kind=sighting.kind,
                last_seen_tick=tick,
                last_amount=sighting.amount,
                confidence=0.70,
            )
        )
        return True

    def mark_success(
        self, kind: ResourceKind, position: Position, tick: int, amount: float
    ) -> None:
        if _key(kind, position) not in self._lookup:
            self._append_memory(
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
        key_expr: pl.Expr = _memory_key_expr(self.agent_id, kind, position)
        self.frame = self.frame.with_columns(
            pl.when(key_expr)
            .then(tick)
            .otherwise(pl.col(MEMORY_LAST_SEEN_TICK))
            .alias(MEMORY_LAST_SEEN_TICK),
            pl.when(key_expr)
            .then(amount)
            .otherwise(pl.col(MEMORY_LAST_AMOUNT))
            .alias(MEMORY_LAST_AMOUNT),
            pl.when(key_expr)
            .then((pl.col(MEMORY_CONFIDENCE) + 0.16).clip(0.0, 1.0))
            .otherwise(pl.col(MEMORY_CONFIDENCE))
            .alias(MEMORY_CONFIDENCE),
            pl.when(key_expr)
            .then(pl.col(MEMORY_SUCCESSFUL_USES) + 1)
            .otherwise(pl.col(MEMORY_SUCCESSFUL_USES))
            .alias(MEMORY_SUCCESSFUL_USES),
            pl.when(key_expr)
            .then((pl.col(MEMORY_FAILED_USES) - 1).clip(0, None))
            .otherwise(pl.col(MEMORY_FAILED_USES))
            .alias(MEMORY_FAILED_USES),
        )

    def mark_failure(self, kind: ResourceKind, position: Position, tick: int) -> None:
        if _key(kind, position) not in self._lookup:
            return
        key_expr: pl.Expr = _memory_key_expr(self.agent_id, kind, position)
        self.frame = self.frame.with_columns(
            pl.when(key_expr)
            .then(tick)
            .otherwise(pl.col(MEMORY_LAST_SEEN_TICK))
            .alias(MEMORY_LAST_SEEN_TICK),
            pl.when(key_expr)
            .then(0.0)
            .otherwise(pl.col(MEMORY_LAST_AMOUNT))
            .alias(MEMORY_LAST_AMOUNT),
            pl.when(key_expr)
            .then((pl.col(MEMORY_CONFIDENCE) - 0.24).clip(0.0, 1.0))
            .otherwise(pl.col(MEMORY_CONFIDENCE))
            .alias(MEMORY_CONFIDENCE),
            pl.when(key_expr)
            .then(pl.col(MEMORY_FAILED_USES) + 1)
            .otherwise(pl.col(MEMORY_FAILED_USES))
            .alias(MEMORY_FAILED_USES),
            pl.when(key_expr)
            .then((pl.col(MEMORY_SEARCH_RADIUS) + 1).clip(None, 10))
            .otherwise(pl.col(MEMORY_SEARCH_RADIUS))
            .alias(MEMORY_SEARCH_RADIUS),
        )

    def best_memory(
        self,
        kind: ResourceKind,
        current: Position,
        tick: int,
        config: SimConfig,
    ) -> ResourceMemory | None:
        filtered: pl.DataFrame = self.frame.filter(
            (pl.col(MEMORY_AGENT_ID) == self.agent_id)
            & (pl.col(MEMORY_KIND) == kind.value)
        )
        if filtered.is_empty():
            return None

        decay_per_day: float = config.water_memory_decay_per_day
        if kind is ResourceKind.FOOD:
            decay_per_day = config.food_memory_decay_per_day
        age_days: pl.Expr = (tick - pl.col(MEMORY_LAST_SEEN_TICK)).clip(
            0, None
        ) / float(config.ticks_per_day)
        confidence: pl.Expr = (
            pl.col(MEMORY_CONFIDENCE)
            + (pl.col(MEMORY_SUCCESSFUL_USES) * 0.04).clip(0.0, 0.25)
            - (pl.col(MEMORY_FAILED_USES) * 0.08).clip(0.0, 0.45)
            - age_days * decay_per_day
        ).clip(0.0, 1.0)
        dx: pl.Expr = (pl.col(MEMORY_X) - current.x).cast(pl.Float64)
        dy: pl.Expr = (pl.col(MEMORY_Y) - current.y).cast(pl.Float64)
        distance: pl.Expr = (dx.pow(2) + dy.pow(2)).sqrt()
        scored: pl.DataFrame = (
            filtered.with_columns(
                confidence.alias("decayed_confidence"),
                (pl.col(MEMORY_LAST_AMOUNT) * 0.12)
                .clip(0.0, 0.25)
                .alias("amount_bonus"),
                distance.alias("distance"),
            )
            .with_columns(
                (
                    pl.col("decayed_confidence")
                    + pl.col("amount_bonus")
                    - pl.col("distance") * 0.025
                ).alias("score")
            )
            .filter(pl.col("decayed_confidence") > 0.08)
            .sort(["score", MEMORY_X, MEMORY_Y], descending=[True, False, False])
            .limit(1)
        )
        if scored.is_empty():
            return None
        return _resource_memory_from_frame(scored)

    def _append_memory(self, memory: ResourceMemory) -> None:
        row: pl.DataFrame = pl.DataFrame(
            [
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
            ],
            schema=MEMORY_SCHEMA,
            orient="row",
        )
        self.frame.extend(row)
        self._lookup[_key(memory.kind, memory.position)] = True


def _key(kind: ResourceKind, position: Position) -> tuple[ResourceKind, int, int]:
    return (kind, position.x, position.y)


def _memory_key_expr(agent_id: int, kind: ResourceKind, position: Position) -> pl.Expr:
    return (
        (pl.col(MEMORY_AGENT_ID) == agent_id)
        & (pl.col(MEMORY_KIND) == kind.value)
        & (pl.col(MEMORY_X) == position.x)
        & (pl.col(MEMORY_Y) == position.y)
    )


def _resource_memory_from_frame(frame: pl.DataFrame) -> ResourceMemory:
    if frame.height != 1:
        raise ValueError("resource memory frame must contain exactly one row")
    values: dict[str, list[Any]] = frame.to_dict(as_series=False)
    return ResourceMemory(
        position=Position(
            _required_int(values, MEMORY_X),
            _required_int(values, MEMORY_Y),
        ),
        kind=ResourceKind(_required_str(values, MEMORY_KIND)),
        last_seen_tick=_required_int(values, MEMORY_LAST_SEEN_TICK),
        last_amount=_required_float(values, MEMORY_LAST_AMOUNT),
        confidence=_required_float(values, MEMORY_CONFIDENCE),
        successful_uses=_required_int(values, MEMORY_SUCCESSFUL_USES),
        failed_uses=_required_int(values, MEMORY_FAILED_USES),
        search_radius=_required_int(values, MEMORY_SEARCH_RADIUS),
    )


def _required_value(values: dict[str, list[Any]], key: str) -> Any:
    column: list[Any] | None = values.get(key)
    if column is None or len(column) != 1:
        raise ValueError(f"memory frame missing scalar column: {key}")
    value: Any = column[0]
    if value is None:
        raise ValueError(f"memory frame column cannot be null: {key}")
    return value


def _required_int(values: dict[str, list[Any]], key: str) -> int:
    value: Any = _required_value(values, key)
    if not isinstance(value, int):
        raise TypeError(f"memory frame column must be int: {key}")
    return value


def _required_float(values: dict[str, list[Any]], key: str) -> float:
    value: Any = _required_value(values, key)
    if not isinstance(value, float):
        raise TypeError(f"memory frame column must be float: {key}")
    return value


def _required_str(values: dict[str, list[Any]], key: str) -> str:
    value: Any = _required_value(values, key)
    if not isinstance(value, str):
        raise TypeError(f"memory frame column must be str: {key}")
    return value
