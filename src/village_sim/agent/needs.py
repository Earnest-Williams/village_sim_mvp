"""Vectorized need and health updates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.types import ActionKind, DeathReason, Position

AGENT_ID = "agent_id"
X = "x"
Y = "y"
THIRST = "thirst"
HUNGER = "hunger"
FATIGUE = "fatigue"
COLD_STRESS = "cold_stress"
HEALTH = "health"
AWAKE_TICKS = "awake_ticks"
ALIVE = "alive"
DEATH_REASON = "death_reason"
CURRENT_GOAL = "current_goal"
CURRENT_ACTION = "current_action"
TARGET_X = "target_x"
TARGET_Y = "target_y"
DISTANCE_WALKED = "distance_walked"
WATER_DISCOVERIES = "water_discoveries"
FOOD_DISCOVERIES = "food_discoveries"

AGENT_SCHEMA: dict[str, Any] = {
    AGENT_ID: pl.Int64,
    X: pl.Int64,
    Y: pl.Int64,
    THIRST: pl.Float64,
    HUNGER: pl.Float64,
    FATIGUE: pl.Float64,
    COLD_STRESS: pl.Float64,
    HEALTH: pl.Float64,
    AWAKE_TICKS: pl.Int64,
    ALIVE: pl.Boolean,
    DEATH_REASON: pl.String,
    CURRENT_GOAL: pl.String,
    CURRENT_ACTION: pl.String,
    TARGET_X: pl.Int64,
    TARGET_Y: pl.Int64,
    DISTANCE_WALKED: pl.Int64,
    WATER_DISCOVERIES: pl.Int64,
    FOOD_DISCOVERIES: pl.Int64,
}


def agent_frame_from_states(agents: Iterable[AgentState]) -> pl.DataFrame:
    """Build the centralized columnar source of truth for agent state."""

    rows: list[dict[str, int | float | bool | str | None]] = []
    for agent in agents:
        target_x: int | None = None if agent.target is None else agent.target.x
        target_y: int | None = None if agent.target is None else agent.target.y
        death_reason: str | None = None
        if agent.death_reason is not None:
            death_reason = agent.death_reason.value
        rows.append(
            {
                AGENT_ID: agent.agent_id,
                X: agent.position.x,
                Y: agent.position.y,
                THIRST: agent.thirst,
                HUNGER: agent.hunger,
                FATIGUE: agent.fatigue,
                COLD_STRESS: agent.cold_stress,
                HEALTH: agent.health,
                AWAKE_TICKS: agent.awake_ticks,
                ALIVE: agent.alive,
                DEATH_REASON: death_reason,
                CURRENT_GOAL: agent.current_goal.value,
                CURRENT_ACTION: agent.current_action.value,
                TARGET_X: target_x,
                TARGET_Y: target_y,
                DISTANCE_WALKED: agent.distance_walked,
                WATER_DISCOVERIES: agent.water_discoveries,
                FOOD_DISCOVERIES: agent.food_discoveries,
            }
        )
    return pl.DataFrame(rows, schema=AGENT_SCHEMA, orient="row")


def sync_agent_to_frame(frame: pl.DataFrame, agent: AgentState) -> pl.DataFrame:
    """Replace one agent row in the centralized DataFrame from compatibility state."""

    replacement: pl.DataFrame = agent_frame_from_states([agent])
    if frame.is_empty():
        return replacement
    if frame.height == 1:
        existing_id: int = int(frame.get_column(AGENT_ID).item())
        if existing_id == agent.agent_id:
            return replacement
    survivors: pl.DataFrame = frame.filter(pl.col(AGENT_ID) != agent.agent_id)
    return pl.concat([survivors, replacement], how="vertical").sort(AGENT_ID)


def sync_agent_from_frame(frame: pl.DataFrame, agent: AgentState) -> None:
    """Refresh the compatibility AgentState cache from the centralized DataFrame."""

    row: pl.DataFrame = frame
    if frame.height != 1 or int(frame.get_column(AGENT_ID).item()) != agent.agent_id:
        row = frame.filter(pl.col(AGENT_ID) == agent.agent_id)
    if row.height != 1:
        raise ValueError("agent frame must contain exactly one row for agent_id")
    values: dict[str, list[Any]] = row.to_dict(as_series=False)
    agent.position = Position(
        x=_required_int(values, X),
        y=_required_int(values, Y),
    )
    agent.thirst = _required_float(values, THIRST)
    agent.hunger = _required_float(values, HUNGER)
    agent.fatigue = _required_float(values, FATIGUE)
    agent.cold_stress = _required_float(values, COLD_STRESS)
    agent.health = _required_float(values, HEALTH)
    agent.awake_ticks = _required_int(values, AWAKE_TICKS)
    agent.alive = _required_bool(values, ALIVE)
    death_reason: str | None = _nullable_str(values, DEATH_REASON)
    agent.death_reason = None if death_reason is None else DeathReason(death_reason)
    agent.current_action = ActionKind(_required_str(values, CURRENT_ACTION))
    target_x: int | None = _nullable_int(values, TARGET_X)
    target_y: int | None = _nullable_int(values, TARGET_Y)
    agent.target = None
    if target_x is not None and target_y is not None:
        agent.target = Position(target_x, target_y)
    agent.distance_walked = _required_int(values, DISTANCE_WALKED)
    agent.water_discoveries = _required_int(values, WATER_DISCOVERIES)
    agent.food_discoveries = _required_int(values, FOOD_DISCOVERIES)


def filter_hungry_agents(frame: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Return agents whose hunger exceeds threshold using a Polars predicate."""

    return frame.filter(pl.col(HUNGER) > threshold)


def filter_thirsty_agents(frame: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Return agents whose thirst exceeds threshold using a Polars predicate."""

    return frame.filter(pl.col(THIRST) > threshold)


def update_needs_frame(
    frame: pl.DataFrame,
    config: SimConfig,
    *,
    is_night: bool = False,
    is_raining: bool = False,
    is_sheltered: bool = False,
    is_cold_exposed: bool | None = None,
) -> pl.DataFrame:
    """Advance biological needs for all live agents with Polars expressions."""

    if frame.height == 1:
        return _update_single_agent_needs_frame(
            frame,
            config,
            is_night=is_night,
            is_raining=is_raining,
            is_sheltered=is_sheltered,
            is_cold_exposed=is_cold_exposed,
        )

    cold_exposed: bool = is_night or is_raining
    if is_cold_exposed is not None:
        cold_exposed = is_cold_exposed

    cold_delta: float = -config.cold_recovery_daylight
    if is_sheltered:
        cold_delta = -config.cold_recovery_shelter
    elif cold_exposed:
        cold_delta = 0.0
        if is_night or not is_raining:
            cold_delta += config.cold_gain_night
        if is_raining:
            cold_delta += config.cold_gain_rain

    awake_expr: pl.Expr = pl.col(CURRENT_ACTION) != ActionKind.SLEEP.value
    live_expr: pl.Expr = pl.col(ALIVE)
    thirst_damage: pl.Expr = pl.when(pl.col(THIRST) >= 0.96).then(0.025).otherwise(0.0)
    hunger_damage: pl.Expr = pl.when(pl.col(HUNGER) >= 0.96).then(0.012).otherwise(0.0)
    fatigue_damage: pl.Expr = (
        pl.when(pl.col(FATIGUE) >= 0.98).then(0.012).otherwise(0.0)
    )
    cold_damage: pl.Expr = (
        pl.when(pl.col(COLD_STRESS) >= config.cold_health_threshold)
        .then(config.cold_health_damage)
        .otherwise(0.0)
    )
    newly_dead: pl.Expr = pl.col(HEALTH) <= 0.0
    max_need: pl.Expr = pl.max_horizontal(
        pl.col(THIRST), pl.col(HUNGER), pl.col(FATIGUE), pl.col(COLD_STRESS)
    )
    death_reason: pl.Expr = (
        pl.when(pl.col(THIRST) == max_need)
        .then(pl.lit(DeathReason.THIRST.value))
        .when(pl.col(HUNGER) == max_need)
        .then(pl.lit(DeathReason.HUNGER.value))
        .when(pl.col(FATIGUE) == max_need)
        .then(pl.lit(DeathReason.EXHAUSTION.value))
        .otherwise(pl.lit(DeathReason.COLD.value))
    )

    return (
        frame.lazy()
        .with_columns(
            pl.when(live_expr)
            .then(pl.col(THIRST) + config.thirst_gain_per_tick)
            .otherwise(pl.col(THIRST))
            .clip(0.0, 1.0)
            .alias(THIRST),
            pl.when(live_expr)
            .then(pl.col(HUNGER) + config.hunger_gain_per_tick)
            .otherwise(pl.col(HUNGER))
            .clip(0.0, 1.0)
            .alias(HUNGER),
            pl.when(live_expr & awake_expr)
            .then(pl.col(FATIGUE) + config.fatigue_gain_awake)
            .when(live_expr)
            .then(pl.col(FATIGUE) - config.fatigue_recovery_sleeping)
            .otherwise(pl.col(FATIGUE))
            .clip(0.0, 1.0)
            .alias(FATIGUE),
            pl.when(live_expr & awake_expr)
            .then(pl.col(AWAKE_TICKS) + 1)
            .otherwise(pl.col(AWAKE_TICKS))
            .alias(AWAKE_TICKS),
            pl.when(live_expr)
            .then(pl.col(COLD_STRESS) + cold_delta)
            .otherwise(pl.col(COLD_STRESS))
            .clip(0.0, 1.0)
            .alias(COLD_STRESS),
        )
        .with_columns(
            pl.when(pl.col(ALIVE))
            .then(
                pl.col(HEALTH)
                - thirst_damage
                - hunger_damage
                - fatigue_damage
                - cold_damage
            )
            .otherwise(pl.col(HEALTH))
            .clip(0.0, 1.0)
            .alias(HEALTH)
        )
        .with_columns(
            pl.when(newly_dead).then(False).otherwise(pl.col(ALIVE)).alias(ALIVE),
            pl.when(newly_dead)
            .then(death_reason)
            .otherwise(pl.col(DEATH_REASON))
            .alias(DEATH_REASON),
        )
        .collect()
    )


def _update_single_agent_needs_frame(
    frame: pl.DataFrame,
    config: SimConfig,
    *,
    is_night: bool,
    is_raining: bool,
    is_sheltered: bool,
    is_cold_exposed: bool | None,
) -> pl.DataFrame:
    values: dict[str, list[Any]] = frame.to_dict(as_series=False)
    if not _required_bool(values, ALIVE):
        return frame

    thirst: float = min(
        1.0,
        max(0.0, _required_float(values, THIRST) + config.thirst_gain_per_tick),
    )
    hunger: float = min(
        1.0,
        max(0.0, _required_float(values, HUNGER) + config.hunger_gain_per_tick),
    )
    fatigue: float = _required_float(values, FATIGUE)
    awake_ticks: int = _required_int(values, AWAKE_TICKS)
    if _required_str(values, CURRENT_ACTION) == ActionKind.SLEEP.value:
        fatigue -= config.fatigue_recovery_sleeping
    else:
        fatigue += config.fatigue_gain_awake
        awake_ticks += 1
    fatigue = min(1.0, max(0.0, fatigue))

    cold_exposed: bool = is_night or is_raining
    if is_cold_exposed is not None:
        cold_exposed = is_cold_exposed
    cold_stress: float = _required_float(values, COLD_STRESS)
    if is_sheltered:
        cold_stress -= config.cold_recovery_shelter
    elif cold_exposed:
        if is_night or not is_raining:
            cold_stress += config.cold_gain_night
        if is_raining:
            cold_stress += config.cold_gain_rain
    else:
        cold_stress -= config.cold_recovery_daylight
    cold_stress = min(1.0, max(0.0, cold_stress))

    health: float = _required_float(values, HEALTH)
    if thirst >= 0.96:
        health -= 0.025
    if hunger >= 0.96:
        health -= 0.012
    if fatigue >= 0.98:
        health -= 0.012
    if cold_stress >= config.cold_health_threshold:
        health -= config.cold_health_damage
    health = min(1.0, max(0.0, health))

    alive: bool = health > 0.0
    death_reason: str | None = _nullable_str(values, DEATH_REASON)
    if not alive:
        severe_needs: dict[str, float] = {
            DeathReason.THIRST.value: thirst,
            DeathReason.HUNGER.value: hunger,
            DeathReason.EXHAUSTION.value: fatigue,
            DeathReason.COLD.value: cold_stress,
        }
        death_reason = max(severe_needs, key=lambda reason: severe_needs[reason])

    row: dict[str, int | float | bool | str | None] = {
        key: column[0] for key, column in values.items()
    }
    row[THIRST] = thirst
    row[HUNGER] = hunger
    row[FATIGUE] = fatigue
    row[COLD_STRESS] = cold_stress
    row[HEALTH] = health
    row[AWAKE_TICKS] = awake_ticks
    row[ALIVE] = alive
    row[DEATH_REASON] = death_reason
    return pl.DataFrame([row], schema=AGENT_SCHEMA, orient="row")


def update_needs(
    agent: AgentState,
    config: SimConfig,
    *,
    is_night: bool = False,
    is_raining: bool = False,
    is_sheltered: bool = False,
    is_cold_exposed: bool | None = None,
) -> None:
    """Advance one compatibility AgentState via the Polars batch path."""

    frame: pl.DataFrame = agent_frame_from_states([agent])
    updated: pl.DataFrame = update_needs_frame(
        frame,
        config,
        is_night=is_night,
        is_raining=is_raining,
        is_sheltered=is_sheltered,
        is_cold_exposed=is_cold_exposed,
    )
    sync_agent_from_frame(updated, agent)


def _required_value(values: dict[str, list[Any]], key: str) -> Any:
    column: list[Any] | None = values.get(key)
    if column is None or len(column) != 1:
        raise ValueError(f"agent frame missing scalar column: {key}")
    value: Any = column[0]
    if value is None:
        raise ValueError(f"agent frame column cannot be null: {key}")
    return value


def _nullable_value(values: dict[str, list[Any]], key: str) -> Any | None:
    column: list[Any] | None = values.get(key)
    if column is None or len(column) != 1:
        raise ValueError(f"agent frame missing scalar column: {key}")
    return column[0]


def _required_int(values: dict[str, list[Any]], key: str) -> int:
    value: Any = _required_value(values, key)
    if not isinstance(value, int):
        raise TypeError(f"agent frame column must be int: {key}")
    return value


def _nullable_int(values: dict[str, list[Any]], key: str) -> int | None:
    value: Any | None = _nullable_value(values, key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"agent frame column must be int or null: {key}")
    return value


def _required_float(values: dict[str, list[Any]], key: str) -> float:
    value: Any = _required_value(values, key)
    if not isinstance(value, float):
        raise TypeError(f"agent frame column must be float: {key}")
    return value


def _required_bool(values: dict[str, list[Any]], key: str) -> bool:
    value: Any = _required_value(values, key)
    if not isinstance(value, bool):
        raise TypeError(f"agent frame column must be bool: {key}")
    return value


def _required_str(values: dict[str, list[Any]], key: str) -> str:
    value: Any = _required_value(values, key)
    if not isinstance(value, str):
        raise TypeError(f"agent frame column must be str: {key}")
    return value


def _nullable_str(values: dict[str, list[Any]], key: str) -> str | None:
    value: Any | None = _nullable_value(values, key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"agent frame column must be str or null: {key}")
    return value
