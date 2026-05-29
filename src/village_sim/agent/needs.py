"""Need and health updates."""

from __future__ import annotations

from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.types import ActionKind, DeathReason


def update_needs(
    agent: AgentState,
    config: SimConfig,
    *,
    is_night: bool = False,
    is_raining: bool = False,
    is_sheltered: bool = False,
) -> None:
    """Advance biological needs by one tick."""

    if not agent.alive:
        return

    agent.thirst += config.thirst_gain_per_tick
    agent.hunger += config.hunger_gain_per_tick

    if agent.current_action is ActionKind.SLEEP:
        agent.fatigue -= config.fatigue_recovery_sleeping
    else:
        agent.fatigue += config.fatigue_gain_awake
        agent.awake_ticks += 1

    if is_sheltered:
        agent.cold_stress -= config.cold_recovery_shelter
    else:
        if is_night:
            agent.cold_stress += config.cold_gain_night
        if is_raining:
            agent.cold_stress += config.cold_gain_rain
        if not is_night and not is_raining:
            agent.cold_stress -= config.cold_recovery_daylight

    agent.clamp_needs()

    if agent.thirst >= 0.96:
        agent.health -= 0.025
    if agent.hunger >= 0.96:
        agent.health -= 0.012
    if agent.fatigue >= 0.98:
        agent.health -= 0.012
    if agent.cold_stress >= config.cold_health_threshold:
        agent.health -= config.cold_health_damage

    agent.clamp_needs()

    if agent.health <= 0.0:
        agent.alive = False
        severe_needs: dict[DeathReason, float] = {
            DeathReason.THIRST: agent.thirst,
            DeathReason.HUNGER: agent.hunger,
            DeathReason.EXHAUSTION: agent.fatigue,
            DeathReason.COLD: agent.cold_stress,
        }
        agent.death_reason = max(severe_needs, key=lambda reason: severe_needs[reason])
