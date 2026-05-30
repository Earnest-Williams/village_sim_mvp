"""Agent need and health updates."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import Any, TypeAlias

import numpy as np
import polars as pl
from numba import njit
from numpy.typing import NDArray

from village_sim.agent.state import ACTION_TO_ID, AgentArrays, AgentState
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


AgentControlArrays: TypeAlias = tuple[
    NDArray[np.bool_],
    NDArray[np.int32],
    NDArray[np.int16],
]
AgentNeedArrays: TypeAlias = tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]

RATE_THIRST_GAIN = 0
RATE_HUNGER_GAIN = 1
RATE_FATIGUE_GAIN_AWAKE = 2
RATE_FATIGUE_RECOVERY_SLEEPING = 3
RATE_COLD_GAIN_NIGHT = 4
RATE_COLD_GAIN_RAIN = 5
RATE_COLD_RECOVERY_DAYLIGHT = 6
RATE_COLD_RECOVERY_SHELTER = 7
RATE_COLD_HEALTH_THRESHOLD = 8
RATE_COLD_HEALTH_DAMAGE = 9

FLAG_SLEEP_ACTION_ID = 0
FLAG_IS_NIGHT = 1
FLAG_IS_RAINING = 2
FLAG_IS_SHELTERED = 3
FLAG_IS_COLD_EXPOSED = 4

AGENT_SCHEMA: dict[str, Any] = {
    AGENT_ID: pl.Int64,
    X: pl.Int32,
    Y: pl.Int32,
    THIRST: pl.Float64,
    HUNGER: pl.Float64,
    FATIGUE: pl.Float64,
    COLD_STRESS: pl.Float64,
    HEALTH: pl.Float64,
    AWAKE_TICKS: pl.Int32,
    ALIVE: pl.Boolean,
    DEATH_REASON: pl.String,
    CURRENT_GOAL: pl.String,
    CURRENT_ACTION: pl.String,
    TARGET_X: pl.Int32,
    TARGET_Y: pl.Int32,
    DISTANCE_WALKED: pl.Int64,
    WATER_DISCOVERIES: pl.Int64,
    FOOD_DISCOVERIES: pl.Int64,
}


def agent_frame_from_states(agents: Iterable[AgentState]) -> pl.DataFrame:
    """Build a report-time columnar view of agent state."""

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
    """Replace one agent row in a report-time DataFrame."""

    replacement: pl.DataFrame = agent_frame_from_states([agent])
    if frame.is_empty():
        return replacement
    survivors: pl.DataFrame = frame.filter(pl.col(AGENT_ID) != agent.agent_id)
    return pl.concat([survivors, replacement], how="vertical").sort(AGENT_ID)


def sync_agent_from_frame(frame: pl.DataFrame, agent: AgentState) -> None:
    """Refresh a compatibility AgentState from a report-time DataFrame."""

    row: pl.DataFrame = frame.filter(pl.col(AGENT_ID) == agent.agent_id)
    if row.height != 1:
        raise ValueError("agent frame must contain exactly one row for agent_id")
    values: dict[str, list[Any]] = row.to_dict(as_series=False)
    agent.position = Position(x=_required_int(values, X), y=_required_int(values, Y))
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
    """Advance biological needs for report-time compatibility DataFrames."""

    rows: list[dict[str, int | float | bool | str | None]] = []
    for values in frame.iter_rows(named=True):
        agent = AgentState(
            agent_id=int(values[AGENT_ID]),
            position=Position(int(values[X]), int(values[Y])),
            thirst=float(values[THIRST]),
            hunger=float(values[HUNGER]),
            fatigue=float(values[FATIGUE]),
            cold_stress=float(values[COLD_STRESS]),
            health=float(values[HEALTH]),
            awake_ticks=int(values[AWAKE_TICKS]),
            alive=bool(values[ALIVE]),
            death_reason=_death_reason_from_value(values[DEATH_REASON]),
            current_action=ActionKind(str(values[CURRENT_ACTION])),
        )
        update_needs(
            agent,
            config,
            is_night=is_night,
            is_raining=is_raining,
            is_sheltered=is_sheltered,
            is_cold_exposed=is_cold_exposed,
        )
        agent_frame = agent_frame_from_states([agent])
        rows.extend(agent_frame.iter_rows(named=True))
    return pl.DataFrame(rows, schema=AGENT_SCHEMA, orient="row")


@njit(cache=True)
def update_needs_batch(
    control_arrays: AgentControlArrays,
    need_arrays: AgentNeedArrays,
    flags: NDArray[np.int32],
    rates: NDArray[np.float64],
) -> None:
    """Advance all live agents' continuous needs in a Numba kernel."""

    alive, awake_ticks, current_action = control_arrays
    thirst, hunger, fatigue, cold_stress, health = need_arrays
    sleep_action_id: int = int(flags[FLAG_SLEEP_ACTION_ID])
    is_night: bool = bool(flags[FLAG_IS_NIGHT])
    is_raining: bool = bool(flags[FLAG_IS_RAINING])
    is_sheltered: bool = bool(flags[FLAG_IS_SHELTERED])
    cold_exposed: bool = bool(flags[FLAG_IS_COLD_EXPOSED])
    for index in range(alive.shape[0]):
        if not alive[index]:
            continue
        thirst[index] = _clip01(thirst[index] + rates[RATE_THIRST_GAIN])
        hunger[index] = _clip01(hunger[index] + rates[RATE_HUNGER_GAIN])
        if int(current_action[index]) == sleep_action_id:
            fatigue[index] = _clip01(
                fatigue[index] - rates[RATE_FATIGUE_RECOVERY_SLEEPING]
            )
        else:
            fatigue[index] = _clip01(fatigue[index] + rates[RATE_FATIGUE_GAIN_AWAKE])
            awake_ticks[index] += 1

        if is_sheltered:
            cold_stress[index] = _clip01(
                cold_stress[index] - rates[RATE_COLD_RECOVERY_SHELTER]
            )
        elif cold_exposed:
            cold_delta: float = 0.0
            if is_night or not is_raining:
                cold_delta += rates[RATE_COLD_GAIN_NIGHT]
            if is_raining:
                cold_delta += rates[RATE_COLD_GAIN_RAIN]
            cold_stress[index] = _clip01(cold_stress[index] + cold_delta)
        else:
            cold_stress[index] = _clip01(
                cold_stress[index] - rates[RATE_COLD_RECOVERY_DAYLIGHT]
            )

        damage: float = 0.0
        if thirst[index] >= 0.96:
            damage += 0.025
        if hunger[index] >= 0.96:
            damage += 0.012
        if fatigue[index] >= 0.98:
            damage += 0.012
        if cold_stress[index] >= rates[RATE_COLD_HEALTH_THRESHOLD]:
            damage += rates[RATE_COLD_HEALTH_DAMAGE]
        health[index] = _clip01(health[index] - damage)
        if health[index] <= 0.0:
            alive[index] = False


@njit(cache=True)
def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def update_needs(
    agent: AgentState,
    config: SimConfig,
    *,
    is_night: bool = False,
    is_raining: bool = False,
    is_sheltered: bool = False,
    is_cold_exposed: bool | None = None,
) -> None:
    """Advance one compatibility AgentState with standard Python floats."""

    if not agent.alive:
        return
    agent.thirst = min(1.0, max(0.0, agent.thirst + config.thirst_gain_per_tick))
    agent.hunger = min(1.0, max(0.0, agent.hunger + config.hunger_gain_per_tick))
    if agent.current_action is ActionKind.SLEEP:
        agent.fatigue -= config.fatigue_recovery_sleeping
    else:
        agent.fatigue += config.fatigue_gain_awake
        agent.awake_ticks += 1
    agent.fatigue = min(1.0, max(0.0, agent.fatigue))

    cold_exposed: bool = is_night or is_raining
    if is_cold_exposed is not None:
        cold_exposed = is_cold_exposed
    if is_sheltered:
        agent.cold_stress -= config.cold_recovery_shelter
    elif cold_exposed:
        if is_night or not is_raining:
            agent.cold_stress += config.cold_gain_night
        if is_raining:
            agent.cold_stress += config.cold_gain_rain
    else:
        agent.cold_stress -= config.cold_recovery_daylight
    agent.cold_stress = min(1.0, max(0.0, agent.cold_stress))

    if agent.thirst >= 0.96:
        agent.health -= 0.025
    if agent.hunger >= 0.96:
        agent.health -= 0.012
    if agent.fatigue >= 0.98:
        agent.health -= 0.012
    if agent.cold_stress >= config.cold_health_threshold:
        agent.health -= config.cold_health_damage
    agent.health = min(1.0, max(0.0, agent.health))

    if agent.health <= 0.0:
        agent.alive = False
        agent.death_reason = _most_severe_death_reason(agent)


def update_needs_arrays(
    agent_arrays: AgentArrays,
    config: SimConfig,
    *,
    is_night: bool,
    is_raining: bool,
    is_sheltered: bool,
    is_cold_exposed: bool | None,
) -> None:
    """Advance all active agent arrays with NumPy vectorized operations."""

    active_mask: NDArray[np.bool_] = agent_arrays.active
    if not bool(np.any(active_mask)):
        return

    thirst_gain: np.float32 = np.float32(config.thirst_gain_per_tick)
    hunger_gain: np.float32 = np.float32(config.hunger_gain_per_tick)
    agent_arrays.thirst[active_mask] = np.clip(
        agent_arrays.thirst[active_mask] + thirst_gain,
        np.float32(0.0),
        np.float32(1.0),
    )
    agent_arrays.hunger[active_mask] = np.clip(
        agent_arrays.hunger[active_mask] + hunger_gain,
        np.float32(0.0),
        np.float32(1.0),
    )

    sleep_mask: NDArray[np.bool_] = active_mask & (
        agent_arrays.current_action == ACTION_TO_ID[ActionKind.SLEEP]
    )
    awake_mask: NDArray[np.bool_] = active_mask & ~sleep_mask
    agent_arrays.fatigue[sleep_mask] = np.clip(
        agent_arrays.fatigue[sleep_mask] - np.float32(config.fatigue_recovery_sleeping),
        np.float32(0.0),
        np.float32(1.0),
    )
    agent_arrays.fatigue[awake_mask] = np.clip(
        agent_arrays.fatigue[awake_mask] + np.float32(config.fatigue_gain_awake),
        np.float32(0.0),
        np.float32(1.0),
    )
    agent_arrays.awake_ticks[awake_mask] += np.int32(1)

    cold_exposed: bool = is_night or is_raining
    if is_cold_exposed is not None:
        cold_exposed = is_cold_exposed
    if is_sheltered:
        agent_arrays.cold_stress[active_mask] = np.clip(
            agent_arrays.cold_stress[active_mask]
            - np.float32(config.cold_recovery_shelter),
            np.float32(0.0),
            np.float32(1.0),
        )
    elif cold_exposed:
        cold_delta: np.float32 = np.float32(0.0)
        if is_night or not is_raining:
            cold_delta = np.float32(cold_delta + np.float32(config.cold_gain_night))
        if is_raining:
            cold_delta = np.float32(cold_delta + np.float32(config.cold_gain_rain))
        agent_arrays.cold_stress[active_mask] = np.clip(
            agent_arrays.cold_stress[active_mask] + cold_delta,
            np.float32(0.0),
            np.float32(1.0),
        )
    else:
        agent_arrays.cold_stress[active_mask] = np.clip(
            agent_arrays.cold_stress[active_mask]
            - np.float32(config.cold_recovery_daylight),
            np.float32(0.0),
            np.float32(1.0),
        )

    damage: NDArray[np.float32] = np.zeros(agent_arrays.count, dtype=np.float32)
    damage += np.where(
        active_mask & (agent_arrays.thirst >= np.float32(0.96)),
        np.float32(0.025),
        np.float32(0.0),
    )
    damage += np.where(
        active_mask & (agent_arrays.hunger >= np.float32(0.96)),
        np.float32(0.012),
        np.float32(0.0),
    )
    damage += np.where(
        active_mask & (agent_arrays.fatigue >= np.float32(0.98)),
        np.float32(0.012),
        np.float32(0.0),
    )
    damage += np.where(
        active_mask
        & (agent_arrays.cold_stress >= np.float32(config.cold_health_threshold)),
        np.float32(config.cold_health_damage),
        np.float32(0.0),
    )
    agent_arrays.health[active_mask] = np.clip(
        agent_arrays.health[active_mask] - damage[active_mask],
        np.float32(0.0),
        np.float32(1.0),
    )
    agent_arrays.active[active_mask & (agent_arrays.health <= np.float32(0.0))] = False


@cache
def _needs_kernel_rates(config: SimConfig) -> NDArray[np.float64]:
    return np.asarray(
        (
            config.thirst_gain_per_tick,
            config.hunger_gain_per_tick,
            config.fatigue_gain_awake,
            config.fatigue_recovery_sleeping,
            config.cold_gain_night,
            config.cold_gain_rain,
            config.cold_recovery_daylight,
            config.cold_recovery_shelter,
            config.cold_health_threshold,
            config.cold_health_damage,
        ),
        dtype=np.float64,
    )


def _most_severe_death_reason(agent: AgentState) -> DeathReason:
    severe_needs: dict[DeathReason, float] = {
        DeathReason.THIRST: agent.thirst,
        DeathReason.HUNGER: agent.hunger,
        DeathReason.EXHAUSTION: agent.fatigue,
        DeathReason.COLD: agent.cold_stress,
    }
    return max(severe_needs, key=lambda reason: severe_needs[reason])


def _death_reason_from_value(value: object) -> DeathReason | None:
    if value is None:
        return None
    return DeathReason(str(value))


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
    if isinstance(value, int):
        return value
    raise TypeError(f"agent frame column must be int: {key}")


def _nullable_int(values: dict[str, list[Any]], key: str) -> int | None:
    value: Any | None = _nullable_value(values, key)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise TypeError(f"agent frame column must be int or null: {key}")


def _required_float(values: dict[str, list[Any]], key: str) -> float:
    value: Any = _required_value(values, key)
    if isinstance(value, float):
        return value
    raise TypeError(f"agent frame column must be float: {key}")


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
