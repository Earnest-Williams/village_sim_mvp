"""Need and health updates."""

from __future__ import annotations

from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.types import ActionKind, DeathReason


def update_needs(agent: AgentState, config: SimConfig) -> None:
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

    if agent.thirst >= 0.96:
        agent.health -= 0.025
    if agent.hunger >= 0.96:
        agent.health -= 0.012
    if agent.fatigue >= 0.98:
        agent.health -= 0.012

    agent.clamp_needs()

    if agent.health <= 0.0:
        agent.alive = False
        if agent.thirst >= agent.hunger and agent.thirst >= agent.fatigue:
            agent.death_reason = DeathReason.THIRST
        elif agent.hunger >= agent.fatigue:
            agent.death_reason = DeathReason.HUNGER
        else:
            agent.death_reason = DeathReason.EXHAUSTION
