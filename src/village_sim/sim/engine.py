"""Simulation engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from village_sim.agent.memory import AgentMemory
from village_sim.agent.needs import update_needs
from village_sim.agent.perception import Observation, perceive
from village_sim.agent.policy import choose_and_execute_action
from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.time import SimClock, clock_from_tick
from village_sim.core.types import ResourceKind
from village_sim.sim.events import TickEvent
from village_sim.sim.metrics import SimResult
from village_sim.sim.snapshot import AgentSnapshot, WorldSnapshot
from village_sim.view.ascii_view import render_ascii_map
from village_sim.world.grid import index_of
from village_sim.world.world import World, choose_spawn_position, generate_world


@dataclass(slots=True)
class Simulation:
    """Headless simulation runtime.

    The engine owns truth. Renderers and exporters consume snapshots.
    """

    config: SimConfig
    rng: random.Random = field(init=False)
    world: World = field(init=False)
    agent: AgentState = field(init=False)
    memory: AgentMemory = field(init=False)
    tick: int = 0
    events: list[TickEvent] = field(default_factory=list)
    snapshots: list[WorldSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.config.validate()
        self.rng = random.Random(self.config.seed)
        self.world = generate_world(self.config, self.rng)
        spawn_position = choose_spawn_position(self.world, self.rng)
        self.agent = AgentState(agent_id=1, position=spawn_position)
        self.agent.ensure_visit_buffer(self.world.width * self.world.height)
        self.memory = AgentMemory()
        self._log("spawn", "agent spawned")

    def step(self) -> None:
        if self.tick >= self.config.max_ticks():
            return

        clock: SimClock = clock_from_tick(self.tick, self.config)
        raining: bool = self.world.step_environment(self.rng, self.config)
        if raining and self.tick % 12 == 0:
            self._log("weather", "rain fell")

        if self.agent.alive:
            self._step_agent(clock)

        self.tick += 1

    def run(self, snapshot_every: int = 0) -> SimResult:
        while self.tick < self.config.max_ticks() and self.agent.alive:
            self.step()
            if snapshot_every > 0 and self.tick % snapshot_every == 0:
                self.snapshots.append(self.snapshot(include_ascii=True))
        return self.result()

    def result(self) -> SimResult:
        days_elapsed: float = self.tick / float(self.config.ticks_per_day)
        death_reason: str | None = None
        if self.agent.death_reason is not None:
            death_reason = self.agent.death_reason.value
        remembered_water_sites: int = 0
        remembered_food_sites: int = 0
        for memory in self.memory.resource_memories:
            if memory.kind is ResourceKind.WATER:
                remembered_water_sites += 1
            elif memory.kind is ResourceKind.FOOD:
                remembered_food_sites += 1
        return SimResult(
            seed=self.config.seed,
            days_elapsed=days_elapsed,
            survived=self.agent.alive,
            death_reason=death_reason,
            final_health=self.agent.health,
            final_thirst=self.agent.thirst,
            final_hunger=self.agent.hunger,
            final_fatigue=self.agent.fatigue,
            water_discoveries=self.agent.water_discoveries,
            food_discoveries=self.agent.food_discoveries,
            distance_walked=self.agent.distance_walked,
            remembered_water_sites=remembered_water_sites,
            remembered_food_sites=remembered_food_sites,
        )

    def snapshot(self, include_ascii: bool = False) -> WorldSnapshot:
        clock: SimClock = clock_from_tick(self.tick, self.config)
        agent_snapshot = AgentSnapshot(
            agent_id=self.agent.agent_id,
            x=self.agent.position.x,
            y=self.agent.position.y,
            thirst=self.agent.thirst,
            hunger=self.agent.hunger,
            fatigue=self.agent.fatigue,
            health=self.agent.health,
            alive=self.agent.alive,
            goal=self.agent.current_goal.value,
            action=self.agent.current_action.value,
        )
        ascii_map: str | None = render_ascii_map(self.world, self.agent) if include_ascii else None
        return WorldSnapshot(
            tick=self.tick,
            day=clock.day,
            tick_of_day=clock.tick_of_day,
            is_daylight=clock.is_daylight,
            agents=[agent_snapshot],
            ascii_map=ascii_map,
        )

    def _step_agent(self, clock: SimClock) -> None:
        assert self.agent.alive
        self.agent.ensure_visit_buffer(self.world.width * self.world.height)
        position_index: int = index_of(self.world.width, self.agent.position)
        self.agent.visited_counts[position_index] += 1

        observation: Observation = perceive(self.world, self.agent.position, clock, self.config)
        for sighting in observation.all_sightings():
            is_new: bool = self.memory.observe(sighting, self.tick)
            if is_new:
                if sighting.kind is ResourceKind.WATER:
                    self.agent.water_discoveries += 1
                    self._log("memory", "discovered water")
                elif sighting.kind is ResourceKind.FOOD:
                    self.agent.food_discoveries += 1
                    self._log("memory", "discovered food")

        action_message: str = choose_and_execute_action(
            self.agent,
            self.memory,
            observation,
            self.world,
            clock,
            self.rng,
            self.config,
        )
        if action_message in {"drank water", "ate food", "slept"} or self.tick % 48 == 0:
            self._log("action", action_message)

        update_needs(self.agent, self.config)
        if not self.agent.alive:
            reason: str = "unknown"
            if self.agent.death_reason is not None:
                reason = self.agent.death_reason.value
            self._log("death", f"agent died from {reason}")

    def _log(self, kind: str, message: str) -> None:
        clock: SimClock = clock_from_tick(self.tick, self.config)
        self.events.append(
            TickEvent(
                tick=self.tick,
                day=clock.day,
                actor=f"agent:{self.agent.agent_id}" if hasattr(self, "agent") else "world",
                kind=kind,
                message=message,
                x=self.agent.position.x if hasattr(self, "agent") else -1,
                y=self.agent.position.y if hasattr(self, "agent") else -1,
            )
        )
