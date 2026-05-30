"""Simulation engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from village_sim.agent.decision import DecisionSource
from village_sim.agent.memory import (
    AgentMemory,
    DiscoverableAgentMemory,
    ResourceMemory,
)
from village_sim.agent.needs import update_needs
from village_sim.agent.perception import Observation, perceive
from village_sim.agent.policy import choose_and_execute_action
from village_sim.agent.state import AgentState, MemoryMarker
from village_sim.core.config import SimConfig
from village_sim.core.time import SimClock, clock_from_tick
from village_sim.core.types import ActionKind, Position, PrimitiveAction, ResourceKind
from village_sim.goap.executor import ExecutionResult, PlanExecutor
from village_sim.goap.planner import PlanStep, plan
from village_sim.orchestrator.action_model import (
    ActionLibrary,
    ActionLifecycle,
    update_confidence_after_execution,
)
from village_sim.orchestrator.orchestrator import Orchestrator
from village_sim.orchestrator.snapshotting import make_state_snapshot
from village_sim.orchestrator.symbolic import FactValue, extract_symbolic_state
from village_sim.orchestrator.trajectory import Trajectory, TrajectoryRecorder
from village_sim.sim.event_summary import (
    count_cold_status_events,
    count_cold_weather_events,
    count_shelter_events,
)
from village_sim.sim.events import TickEvent
from village_sim.sim.metrics import LearningStats, SimResult
from village_sim.sim.snapshot import AgentSnapshot, WorldSnapshot
from village_sim.view.ascii_view import render_ascii_map
from village_sim.world.discoverables import (
    Discoverable,
    DiscoverableKind,
    discoverable_at_or_adjacent,
    exploit_discoverable,
    make_initial_discoverables,
    update_discoverable_memory,
)
from village_sim.world.grid import index_of
from village_sim.world.weather import (
    COLD_REASON_DAY,
    COLD_REASON_NIGHT,
    COLD_REASON_NIGHT_RAIN,
    COLD_REASON_RAIN,
    COLD_STATUS_COLD,
    COLD_STATUS_OK,
    COLD_STATUS_SEVERE,
    SHELTERED_ACTION_PREFIX,
    SEEKING_SHELTER_ACTION_PREFIX,
    STATUS_AGENT_COLD,
    STATUS_AGENT_SEVERELY_COLD,
    WEATHER_COLD_DAY,
    WEATHER_COLD_NIGHT,
    WEATHER_COLD_NIGHT_RAIN,
    WEATHER_COLD_RAIN,
    ColdStatus,
    WeatherState,
    make_weather_state,
)
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
    discoverable_memory: DiscoverableAgentMemory = field(init=False)
    orchestrator: Orchestrator = field(init=False)
    action_library: ActionLibrary = field(init=False)
    recorded_trajectories: list[Trajectory] = field(
        default_factory=lambda: list[Trajectory]()
    )
    tick: int = 0
    events: list[TickEvent] = field(default_factory=lambda: list[TickEvent]())
    snapshots: list[WorldSnapshot] = field(
        default_factory=lambda: list[WorldSnapshot]()
    )
    current_weather: WeatherState = field(init=False)
    _last_feels_cold: bool = field(default=False, init=False)
    _last_cold_status_bucket: ColdStatus = field(default=COLD_STATUS_OK, init=False)
    learning: LearningStats = field(default_factory=LearningStats, init=False)
    synthesized_actions_added: int = field(default=0, init=False)
    goap_plan_executions: int = field(default=0, init=False)
    successful_goap_plan_executions: int = field(default=0, init=False)
    _last_memory_decision_key: tuple[str, str, int, int] | None = field(
        default=None, init=False
    )
    _last_memory_decision_tick: int = field(default=-1_000_000, init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        self.rng = random.Random(self.config.seed)
        initial_discoverables: dict[str, Discoverable] | None = None
        if self.config.enable_initial_discoverables:
            initial_discoverables = make_initial_discoverables()
        self.world = generate_world(
            self.config,
            self.rng,
            discoverables=initial_discoverables,
        )
        spawn_position = choose_spawn_position(self.world, self.rng)
        self.agent = AgentState(agent_id=1, position=spawn_position)
        self.agent.ensure_visit_buffer(self.world.width * self.world.height)
        self.memory = AgentMemory()
        self.discoverable_memory = DiscoverableAgentMemory()
        self.orchestrator = Orchestrator()
        self.action_library = ActionLibrary()
        initial_clock: SimClock = clock_from_tick(self.tick, self.config)
        self.current_weather = make_weather_state(
            is_raining=False,
            is_night=initial_clock.is_night,
            config=self.config,
        )
        self._log("spawn", "agent spawned")
        self._log_weather_transition(self.current_weather)

    def step(self) -> None:
        if self.tick >= self.config.max_ticks():
            return

        clock: SimClock = clock_from_tick(self.tick, self.config)
        raining: bool = self.world.step_environment(
            self.rng, self.config, clock.tick_of_day
        )
        weather: WeatherState = make_weather_state(
            is_raining=raining,
            is_night=clock.is_night,
            config=self.config,
        )
        self.current_weather = weather
        if raining and self.tick % 12 == 0:
            self._log("weather", "rain fell")
        self._log_weather_transition(weather)

        if self.agent.alive:
            self._step_agent(clock, weather)

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
        best_water: ResourceMemory | None = self.memory.best_memory(
            ResourceKind.WATER, self.agent.position, self.tick, self.config
        )
        best_food: ResourceMemory | None = self.memory.best_memory(
            ResourceKind.FOOD, self.agent.position, self.tick, self.config
        )
        return SimResult(
            seed=self.config.seed,
            days_elapsed=days_elapsed,
            survived=self.agent.alive,
            death_reason=death_reason,
            final_health=self.agent.health,
            final_thirst=self.agent.thirst,
            final_hunger=self.agent.hunger,
            final_fatigue=self.agent.fatigue,
            final_cold_stress=self.agent.cold_stress,
            final_temperature_c=self.current_weather.temperature_c,
            final_feels_cold=self.current_weather.feels_cold,
            final_is_sheltered=self.agent_is_sheltered(),
            final_cold_status=self._cold_status_bucket(self.agent.cold_stress),
            cold_weather_events=count_cold_weather_events(self.events),
            cold_status_events=count_cold_status_events(self.events),
            shelter_events=count_shelter_events(self.events),
            water_discoveries=self.agent.water_discoveries,
            food_discoveries=self.agent.food_discoveries,
            distance_walked=self.agent.distance_walked,
            remembered_water_sites=remembered_water_sites,
            remembered_food_sites=remembered_food_sites,
            learning=self.learning,
            best_water_memory_x=-1 if best_water is None else best_water.position.x,
            best_water_memory_y=-1 if best_water is None else best_water.position.y,
            best_water_memory_confidence=(
                0.0
                if best_water is None
                else best_water.decayed_confidence(self.tick, self.config)
            ),
            best_water_memory_successful_uses=(
                0 if best_water is None else best_water.successful_uses
            ),
            best_water_memory_failed_uses=(
                0 if best_water is None else best_water.failed_uses
            ),
            best_food_memory_x=-1 if best_food is None else best_food.position.x,
            best_food_memory_y=-1 if best_food is None else best_food.position.y,
            best_food_memory_confidence=(
                0.0
                if best_food is None
                else best_food.decayed_confidence(self.tick, self.config)
            ),
            best_food_memory_successful_uses=(
                0 if best_food is None else best_food.successful_uses
            ),
            best_food_memory_failed_uses=(
                0 if best_food is None else best_food.failed_uses
            ),
            action_library_size=len(self.action_library.all_actions()),
            synthesized_actions_added=self.synthesized_actions_added,
            goap_plan_executions=self.goap_plan_executions,
            successful_goap_plan_executions=self.successful_goap_plan_executions,
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
            cold_stress=self.agent.cold_stress,
            health=self.agent.health,
            alive=self.agent.alive,
            goal=self.agent.current_goal.value,
            action=self.agent.current_action.value,
            feels_cold=self.current_weather.feels_cold,
            is_sheltered=self.agent_is_sheltered(),
            cold_status=self._cold_status_bucket(self.agent.cold_stress),
            decision_source=self.agent.decision_trace.source.value,
            decision_target_kind=self.agent.decision_trace.target_kind,
            decision_target_x=self.agent.decision_trace.target_x,
            decision_target_y=self.agent.decision_trace.target_y,
            decision_memory_confidence=self.agent.decision_trace.memory_confidence,
            memory_use_ratio=self.learning.memory_use_ratio,
        )
        ascii_map: str | None = (
            render_ascii_map(self.world, self.agent) if include_ascii else None
        )
        return WorldSnapshot(
            tick=self.tick,
            day=clock.day,
            tick_of_day=clock.tick_of_day,
            is_daylight=clock.is_daylight,
            is_raining=self.current_weather.is_raining,
            temperature_c=self.current_weather.temperature_c,
            feels_cold=self.current_weather.feels_cold,
            cold_reason=self.current_weather.cold_reason,
            agents=[agent_snapshot],
            ascii_map=ascii_map,
        )

    def _step_agent(self, clock: SimClock, weather: WeatherState) -> None:
        self.agent.ensure_visit_buffer(self.world.width * self.world.height)
        position_index: int = index_of(self.world.width, self.agent.position)
        self.agent.visited_counts[position_index] += 1

        is_sheltered: bool = self.agent_is_sheltered()
        observation: Observation = perceive(
            self.world,
            self.agent.position,
            clock,
            self.config,
            weather,
            is_sheltered,
        )
        for sighting in observation.all_sightings():
            is_new: bool = self.memory.observe(sighting, self.tick)
            memory_record: ResourceMemory | None = self._memory_at(
                sighting.kind, sighting.position
            )
            if is_new:
                if sighting.kind is ResourceKind.WATER:
                    self.agent.water_discoveries += 1
                    self.learning.learned_water_sites += 1
                    self._log_at(
                        "learning",
                        self._learned_message(ResourceKind.WATER, sighting.position),
                        sighting.position,
                    )
                elif sighting.kind is ResourceKind.FOOD:
                    self.agent.food_discoveries += 1
                    self.learning.learned_food_sites += 1
                    self._log_at(
                        "learning",
                        self._learned_message(ResourceKind.FOOD, sighting.position),
                        sighting.position,
                    )
            elif memory_record is not None:
                confidence: str = self._format_confidence(
                    memory_record.decayed_confidence(self.tick, self.config)
                )
                self._log_at(
                    "learning",
                    self._updated_memory_message(
                        sighting.kind, sighting.position, confidence
                    ),
                    sighting.position,
                )
        self._sync_memory_markers()

        new_discoverable_ids: list[str] = update_discoverable_memory(
            self.discoverable_memory,
            observation.discoverables,
            self.tick,
        )
        for discoverable_id in new_discoverable_ids:
            item = self.world.discoverables.get(discoverable_id)
            if item is not None:
                item.discovered = True
            self._log("memory", f"discovered {discoverable_id}")

        if self.config.enable_goap_control:
            goal = self._urgent_goap_goal()
            if goal is not None:
                results = self.execute_goap_plan(goal)
                if results and all(result.success for result in results):
                    if not self.agent.alive:
                        self._log_agent_death()
                    return

        interaction_ticks = self._try_record_discoverable_exploitation(
            clock, observation
        )
        if interaction_ticks > 0:
            update_needs(
                self.agent,
                self.config,
                is_night=clock.is_night,
                is_raining=weather.is_raining,
                is_sheltered=self.agent_is_sheltered(),
                is_cold_exposed=weather.feels_cold,
            )
            self._log_cold_status_transition()
            self._advance_interaction_ticks(interaction_ticks)
            if not self.agent.alive:
                self._log_agent_death()
            return

        before_memory_state: dict[tuple[str, int, int], tuple[int, int]] = (
            self._memory_use_state()
        )
        action_message: str = choose_and_execute_action(
            self.agent,
            self.memory,
            observation,
            self.world,
            clock,
            self.rng,
            self.config,
        )
        self._record_decision_trace()
        self._record_memory_use_deltas(before_memory_state)
        self._sync_memory_markers()
        if (
            action_message in {"drank water", "ate food", "slept"}
            or self.tick % 48 == 0
        ):
            self._log("action", action_message)

        update_needs(
            self.agent,
            self.config,
            is_night=clock.is_night,
            is_raining=weather.is_raining,
            is_sheltered=self.agent_is_sheltered(),
            is_cold_exposed=weather.feels_cold,
        )
        self._log_cold_status_transition()
        if not self.agent.alive:
            self._log_agent_death()

    def _try_record_discoverable_exploitation(
        self,
        clock: SimClock,
        observation: Observation,
    ) -> int:
        item = discoverable_at_or_adjacent(
            self.world,
            self.agent.position.x,
            self.agent.position.y,
        )
        if item is None:
            return 0
        if not self._should_exploit_discoverable(item):
            return 0

        before_snapshot = make_state_snapshot(
            tick=self.tick,
            agent=self.agent,
            observation=observation,
            discoverable_memory=self.discoverable_memory,
            clock=clock,
        )
        if item.kind is DiscoverableKind.CAVE:
            self._log(
                "action", f"{SEEKING_SHELTER_ACTION_PREFIX}{item.discoverable_id}"
            )
        success: bool = exploit_discoverable(self.agent, item)
        after_tick: int = self.tick + item.interaction_ticks
        # The exploit trajectory after-snapshot captures the immediate
        # post-exploit state before multi-tick environmental advancement. The
        # after_tick records modeled action duration for trajectory cost, while
        # weather/daylight context remains the current tick;
        # _advance_interaction_ticks() applies later environmental changes.
        after_weather: WeatherState = make_weather_state(
            is_raining=self.current_weather.is_raining,
            is_night=clock.is_night,
            config=self.config,
        )
        after_observation: Observation = perceive(
            self.world,
            self.agent.position,
            clock,
            self.config,
            after_weather,
            self.agent_is_sheltered(),
        )
        update_discoverable_memory(
            self.discoverable_memory,
            after_observation.discoverables,
            after_tick,
        )
        after_snapshot = make_state_snapshot(
            tick=after_tick,
            agent=self.agent,
            observation=after_observation,
            discoverable_memory=self.discoverable_memory,
            clock=clock,
        )
        task_name: str = item.satisfies_need
        primitive_action: PrimitiveAction = PrimitiveAction.WAIT
        if item.satisfies_need == "thirst":
            primitive_action = PrimitiveAction.DRINK
            self.agent.current_action = ActionKind.DRINK
        elif item.satisfies_need == "hunger":
            primitive_action = PrimitiveAction.EAT
            self.agent.current_action = ActionKind.EAT
        elif item.satisfies_need == "cold_stress":
            self.agent.current_action = ActionKind.IDLE

        recorder = TrajectoryRecorder(
            trajectory_id=f"traj_live_{self.agent.agent_id}_{self.tick}_{item.discoverable_id}",
            policy_id=f"policy_exploit_{item.kind.value}_v1",
            task_name=task_name,
        )
        event_name = "success" if success else "failed"
        recorder.record(
            before=before_snapshot,
            action=primitive_action,
            after=after_snapshot,
            reward=1.0 if success else -1.0,
            events=[f"exploit:{item.discoverable_id}:{event_name}"],
        )
        trajectory = recorder.finish()
        self.recorded_trajectories.append(trajectory)
        self.orchestrator.record(trajectory)
        for action in self.orchestrator.synthesize_all():
            self.action_library.add(action)

        if item.kind is DiscoverableKind.CAVE:
            if success:
                self._log("action", f"{SHELTERED_ACTION_PREFIX}{item.discoverable_id}")
        else:
            self._log("action", f"exploit {item.discoverable_id} {event_name}")
        return item.interaction_ticks

    def _advance_interaction_ticks(self, interaction_ticks: int) -> None:
        extra_ticks: int = max(0, interaction_ticks - 1)
        while extra_ticks > 0 and self.tick < self.config.max_ticks() - 1:
            self.tick += 1
            busy_clock: SimClock = clock_from_tick(self.tick, self.config)
            raining: bool = self.world.step_environment(
                self.rng, self.config, busy_clock.tick_of_day
            )
            weather: WeatherState = make_weather_state(
                is_raining=raining,
                is_night=busy_clock.is_night,
                config=self.config,
            )
            self.current_weather = weather
            if raining and self.tick % 12 == 0:
                self._log("weather", "rain fell")
            self._log_weather_transition(weather)
            update_needs(
                self.agent,
                self.config,
                is_night=busy_clock.is_night,
                is_raining=weather.is_raining,
                is_sheltered=self.agent_is_sheltered(),
                is_cold_exposed=weather.feels_cold,
            )
            self._log_cold_status_transition()
            extra_ticks -= 1
            if not self.agent.alive:
                return

    def _log_weather_transition(self, weather: WeatherState) -> None:
        if weather.feels_cold and not self._last_feels_cold:
            if weather.cold_reason == COLD_REASON_NIGHT:
                self._log("weather", WEATHER_COLD_NIGHT)
            elif weather.cold_reason == COLD_REASON_RAIN:
                self._log("weather", WEATHER_COLD_RAIN)
            elif weather.cold_reason == COLD_REASON_NIGHT_RAIN:
                self._log("weather", WEATHER_COLD_NIGHT_RAIN)
            elif weather.cold_reason == COLD_REASON_DAY:
                self._log("weather", WEATHER_COLD_DAY)
        self._last_feels_cold = weather.feels_cold

    def _log_cold_status_transition(self) -> None:
        bucket: ColdStatus = self._cold_status_bucket(self.agent.cold_stress)
        if bucket == self._last_cold_status_bucket:
            return
        if bucket == COLD_STATUS_COLD:
            self._log("status", STATUS_AGENT_COLD)
        elif bucket == COLD_STATUS_SEVERE:
            self._log("status", STATUS_AGENT_SEVERELY_COLD)
        self._last_cold_status_bucket = bucket

    @staticmethod
    def _cold_status_bucket(cold_stress: float) -> ColdStatus:
        if cold_stress >= 0.60:
            return COLD_STATUS_SEVERE
        if cold_stress >= 0.30:
            return COLD_STATUS_COLD
        return COLD_STATUS_OK

    def record_discoverable_exploitation(
        self,
        clock: SimClock,
        observation: Observation,
    ) -> int:
        """Record and execute one adjacent discoverable exploitation attempt."""
        return self._try_record_discoverable_exploitation(clock, observation)

    def advance_interaction_ticks(self, interaction_ticks: int) -> None:
        """Advance time while an explicit multi-tick interaction is in progress."""
        self._advance_interaction_ticks(interaction_ticks)

    def agent_is_sheltered(self) -> bool:
        """Return whether the agent is currently sheltered by cave adjacency."""

        return self._agent_is_sheltered()

    def _agent_is_sheltered(self) -> bool:
        item = discoverable_at_or_adjacent(
            self.world,
            self.agent.position.x,
            self.agent.position.y,
        )
        return item is not None and item.kind is DiscoverableKind.CAVE

    def _should_exploit_discoverable(self, item: Discoverable) -> bool:
        if item.amount <= 0.0:
            return False
        if item.satisfies_need == "thirst":
            return self.agent.thirst >= 0.60
        if item.satisfies_need == "hunger":
            return self.agent.hunger >= 0.60
        if item.satisfies_need == "cold_stress":
            return self.agent.cold_stress >= 0.60
        return False

    def current_goap_plan(self, goal: dict[str, FactValue]) -> list[PlanStep]:
        """Return a bounded multi-step GOAP plan from the live symbolic state."""
        clock: SimClock = clock_from_tick(self.tick, self.config)
        observation: Observation = perceive(
            self.world,
            self.agent.position,
            clock,
            self.config,
            self.current_weather,
            self.agent_is_sheltered(),
        )
        symbolic = extract_symbolic_state(
            self.agent,
            observation,
            self.discoverable_memory,
            clock,
        )
        return plan(
            symbolic,
            goal,
            self.action_library.all_actions(),
            agent_lifecycle_floor=ActionLifecycle.CANDIDATE,
        )

    def execute_goap_plan(
        self,
        goal: dict[str, FactValue],
        max_steps: int = 4,
    ) -> list[ExecutionResult]:
        """Plan and execute a synthesized GOAP chain in the live simulation."""
        steps = self.current_goap_plan(goal)[:max_steps]
        if not steps:
            return []

        self.goap_plan_executions += 1
        self.agent.decision_trace.source = DecisionSource.GOAP
        self.agent.decision_trace.reason = "executing GOAP plan"
        start_tick = self.tick
        executor = PlanExecutor(self)
        results: list[ExecutionResult] = []
        for step in steps:
            result = executor.execute_step(step)
            results.append(result)
            update_confidence_after_execution(
                step.action,
                success=result.success,
                death=result.death,
                timeout=result.timeout,
            )
            if result.trajectory is not None:
                if result.trajectory not in self.recorded_trajectories:
                    self.recorded_trajectories.append(result.trajectory)
                    self.orchestrator.record(result.trajectory)
                synthesized_actions = self.orchestrator.synthesize_all()
                if synthesized_actions:
                    before_count = len(self.action_library.all_actions())
                    for action in synthesized_actions:
                        self.action_library.add(action)
                    after_count = len(self.action_library.all_actions())
                    if after_count > before_count:
                        self.synthesized_actions_added += after_count - before_count
            if not result.success or result.death or result.timeout:
                break
        if results and all(result.success for result in results):
            self.successful_goap_plan_executions += 1
        if self.tick > start_tick:
            self.tick -= 1
        return results

    def _urgent_goap_goal(self) -> dict[str, FactValue] | None:
        urgent_needs: tuple[tuple[str, float], ...] = (
            ("thirst_bucket", self.agent.thirst),
            ("hunger_bucket", self.agent.hunger),
            ("cold_stress_bucket", self.agent.cold_stress),
        )
        goal_name, goal_value = max(urgent_needs, key=lambda pair: pair[1])
        if goal_value < 0.60:
            return None
        return {goal_name: "low"}

    def _log_agent_death(self) -> None:
        reason: str = "unknown"
        if self.agent.death_reason is not None:
            reason = self.agent.death_reason.value
        self._log("death", f"agent died from {reason}")

    def _memory_at(
        self, kind: ResourceKind, position: Position
    ) -> ResourceMemory | None:
        for memory in self.memory.resource_memories:
            if memory.kind is kind and memory.position == position:
                return memory
        return None

    def _sync_memory_markers(self) -> None:
        markers: list[MemoryMarker] = []
        for memory in self.memory.resource_memories:
            markers.append(
                MemoryMarker(
                    position=memory.position,
                    kind=memory.kind,
                    confidence=memory.decayed_confidence(self.tick, self.config),
                )
            )
        self.agent.memory_markers = markers

    def _memory_use_state(self) -> dict[tuple[str, int, int], tuple[int, int]]:
        state: dict[tuple[str, int, int], tuple[int, int]] = {}
        for memory in self.memory.resource_memories:
            state[(memory.kind.value, memory.position.x, memory.position.y)] = (
                memory.successful_uses,
                memory.failed_uses,
            )
        return state

    def _record_decision_trace(self) -> None:
        trace = self.agent.decision_trace
        if trace.source is DecisionSource.EXPLORE:
            if trace.target_kind == ResourceKind.WATER.value:
                self.learning.explore_for_water_ticks += 1
            elif trace.target_kind == ResourceKind.FOOD.value:
                self.learning.explore_for_food_ticks += 1
            return
        if trace.source is DecisionSource.VISIBLE_RESOURCE:
            self.learning.visible_resource_target_ticks += 1
            return
        if trace.source is DecisionSource.REMEMBERED_RESOURCE:
            self.learning.remembered_resource_target_ticks += 1
            self._record_memory_selection(is_search=False)
            return
        if trace.source is DecisionSource.SEARCH_NEAR_MEMORY:
            self.learning.search_near_memory_ticks += 1
            self._record_memory_selection(is_search=True)

    def _record_memory_selection(self, *, is_search: bool) -> None:
        trace = self.agent.decision_trace
        key: tuple[str, str, int, int] = (
            trace.source.value,
            trace.target_kind,
            trace.target_x,
            trace.target_y,
        )
        should_log: bool = (
            self._last_memory_decision_key != key
            or self.tick - self._last_memory_decision_tick >= 12
        )
        if trace.target_kind == ResourceKind.WATER.value:
            if is_search:
                self.learning.memory_search_water += 1
            else:
                self.learning.memory_selected_water += 1
        elif trace.target_kind == ResourceKind.FOOD.value:
            if is_search:
                self.learning.memory_search_food += 1
            else:
                self.learning.memory_selected_food += 1
        if should_log:
            position = Position(x=trace.target_x, y=trace.target_y)
            confidence = self._format_confidence(trace.memory_confidence)
            if is_search:
                memory = self._memory_at(ResourceKind(trace.target_kind), position)
                radius = 0 if memory is None else memory.search_radius
                message = self._searching_memory_message(
                    ResourceKind(trace.target_kind), position, radius, confidence
                )
            else:
                message = self._using_memory_message(
                    ResourceKind(trace.target_kind), position, confidence
                )
            self._log_at("decision", message, position)
            self._last_memory_decision_key = key
            self._last_memory_decision_tick = self.tick

    def _record_memory_use_deltas(
        self, before_state: dict[tuple[str, int, int], tuple[int, int]]
    ) -> None:
        for memory in self.memory.resource_memories:
            key = (memory.kind.value, memory.position.x, memory.position.y)
            before_successes, before_failures = before_state.get(key, (0, 0))
            confidence = self._format_confidence(
                memory.decayed_confidence(self.tick, self.config)
            )
            if memory.successful_uses > before_successes:
                if memory.kind is ResourceKind.WATER:
                    self.learning.memory_reinforced_water += (
                        memory.successful_uses - before_successes
                    )
                else:
                    self.learning.memory_reinforced_food += (
                        memory.successful_uses - before_successes
                    )
                self._log_at(
                    "learning",
                    self._reinforced_memory_message(memory, confidence),
                    memory.position,
                )
            if memory.failed_uses > before_failures:
                if memory.kind is ResourceKind.WATER:
                    self.learning.memory_failed_water += (
                        memory.failed_uses - before_failures
                    )
                else:
                    self.learning.memory_failed_food += (
                        memory.failed_uses - before_failures
                    )
                self._log_at(
                    "learning",
                    self._weakened_memory_message(memory, confidence),
                    memory.position,
                )

    @staticmethod
    def _format_confidence(confidence: float) -> str:
        return f"{confidence:.2f}"

    @staticmethod
    def _learned_message(kind: ResourceKind, position: Position) -> str:
        return f"learned {kind.value} at {position.x},{position.y}"

    @staticmethod
    def _updated_memory_message(
        kind: ResourceKind, position: Position, confidence: str
    ) -> str:
        return (
            f"updated {kind.value} memory at {position.x},{position.y} "
            f"confidence={confidence}"
        )

    @staticmethod
    def _using_memory_message(
        kind: ResourceKind, position: Position, confidence: str
    ) -> str:
        return (
            f"using remembered {kind.value} at {position.x},{position.y} "
            f"confidence={confidence}"
        )

    @staticmethod
    def _searching_memory_message(
        kind: ResourceKind, position: Position, radius: int, confidence: str
    ) -> str:
        return (
            f"searching near remembered {kind.value} at {position.x},{position.y} "
            f"radius={radius} confidence={confidence}"
        )

    @staticmethod
    def _reinforced_memory_message(memory: ResourceMemory, confidence: str) -> str:
        return (
            f"reinforced {memory.kind.value} memory at "
            f"{memory.position.x},{memory.position.y} "
            f"successful_uses={memory.successful_uses} confidence={confidence}"
        )

    @staticmethod
    def _weakened_memory_message(memory: ResourceMemory, confidence: str) -> str:
        return (
            f"weakened {memory.kind.value} memory at "
            f"{memory.position.x},{memory.position.y} "
            f"failed_uses={memory.failed_uses} confidence={confidence}"
        )

    def _log(self, kind: str, message: str) -> None:
        if hasattr(self, "agent"):
            position = self.agent.position
        else:
            position = Position(x=-1, y=-1)
        self._log_at(kind, message, position)

    def _log_at(self, kind: str, message: str, position: Position) -> None:
        clock: SimClock = clock_from_tick(self.tick, self.config)
        self.events.append(
            TickEvent(
                tick=self.tick,
                day=clock.day,
                actor=(
                    f"agent:{self.agent.agent_id}"
                    if hasattr(self, "agent")
                    else "world"
                ),
                kind=kind,
                message=message,
                x=position.x,
                y=position.y,
            )
        )
