"""Simulation engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from village_sim.agent.actions import BUILD, CHOP, DIG, PLANT, normalize_action_payload
from village_sim.agent.decision import DecisionSource
from village_sim.agent.memory import (
    AgentMemory,
    DiscoverableAgentMemory,
    ResourceMemory,
)
from village_sim.agent.needs import update_needs_arrays
from village_sim.agent.perception import (
    Observation,
    max_resource_sightings,
    perceive,
    perceive_batch_resources,
    RESOURCE_KIND_WATER,
    RESOURCE_KIND_FOOD,
)
from village_sim.agent.policy import choose_and_execute_action
from village_sim.agent.state import (
    AgentArrays,
    AgentState,
    MemoryMarker,
    agent_arrays_from_states,
    sync_agent_from_arrays,
    sync_agent_to_arrays,
    validate_arrays_match_dataclasses,
)
from village_sim.core.config import SimConfig
from village_sim.core.time import SimClock, clock_from_tick
from village_sim.core.types import (
    ActionKind,
    DeathReason,
    Position,
    PrimitiveAction,
    ResourceKind,
    TerrainKind,
)
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
from village_sim.sim.profile import SimulationTimings, TimingCategory
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
from village_sim.world.grid import STRUCTURE_SHELTER, WorldGrids, index_of
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
from village_sim.world.water_system import WaterSystemState
from village_sim.world.world import World, choose_spawn_position, generate_world


@dataclass(slots=True)
class SimulationState:
    """Data-oriented runtime container for hot-loop state."""

    agent_arrays: AgentArrays
    world: World
    water_system: WaterSystemState


@dataclass(slots=True)
class Simulation:
    """Headless simulation runtime.

    The engine owns truth. Renderers and exporters consume snapshots.
    """

    config: SimConfig
    profiler: SimulationTimings | None = None
    rng: random.Random = field(init=False)
    world: World = field(init=False)
    agent: AgentState = field(init=False)
    agent_arrays: AgentArrays = field(init=False)
    state: SimulationState = field(init=False)
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
    _perception_agent_ids: NDArray[np.int64] = field(init=False)
    _perception_agent_x: NDArray[np.int32] = field(init=False)
    _perception_agent_y: NDArray[np.int32] = field(init=False)
    _perception_agent_alive: NDArray[np.bool_] = field(init=False)
    _perception_out_agent_ids: NDArray[np.int64] = field(init=False)
    _perception_out_tile_indices: NDArray[np.int64] = field(init=False)
    _perception_out_kinds: NDArray[np.int32] = field(init=False)
    _perception_out_amounts: NDArray[np.float64] = field(init=False)

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
        self.agent_arrays = agent_arrays_from_states([self.agent])
        self.memory = AgentMemory(agent_id=self.agent.agent_id)
        self.discoverable_memory = DiscoverableAgentMemory()
        self.orchestrator = Orchestrator()
        self.action_library = ActionLibrary()
        initial_clock: SimClock = clock_from_tick(self.tick, self.config)
        self.current_weather = make_weather_state(
            is_raining=False,
            is_night=initial_clock.is_night,
            config=self.config,
        )
        self.state = SimulationState(
            agent_arrays=self.agent_arrays,
            world=self.world,
            water_system=self.world.water_system,
        )
        self._init_perception_buffers()
        validate_arrays_match_dataclasses(self.agent_arrays, [self.agent])
        self._log("spawn", "agent spawned")
        self._log_weather_transition(self.current_weather)

    def _init_perception_buffers(self) -> None:
        self._perception_agent_ids = np.empty(1, dtype=np.int64)
        self._perception_agent_x = np.empty(1, dtype=np.int32)
        self._perception_agent_y = np.empty(1, dtype=np.int32)
        self._perception_agent_alive = np.empty(1, dtype=np.bool_)
        radius: int = max(
            self.config.vision_radius_day,
            self.config.vision_radius_night,
        )
        capacity: int = max_resource_sightings(1, radius)
        self._perception_out_agent_ids = np.empty(capacity, dtype=np.int64)
        self._perception_out_tile_indices = np.empty(capacity, dtype=np.int64)
        self._perception_out_kinds = np.empty(capacity, dtype=np.int32)
        self._perception_out_amounts = np.empty(capacity, dtype=np.float64)

    def _sync_agent_cache_to_arrays(self) -> None:
        sync_agent_to_arrays(self.agent_arrays, self.agent, 0)

    def _sync_agent_cache_from_arrays(self) -> None:
        was_alive: bool = self.agent.alive
        sync_agent_from_arrays(self.agent_arrays, self.agent, 0)
        if was_alive and not self.agent.alive:
            self.agent.death_reason = self._death_reason_from_arrays(0)

    def _update_agent_needs(
        self,
        *,
        is_night: bool = False,
        is_raining: bool = False,
        is_sheltered: bool = False,
        is_cold_exposed: bool | None = None,
    ) -> None:
        self._sync_agent_cache_to_arrays()
        update_needs_arrays(
            self.agent_arrays,
            self.config,
            is_night=is_night,
            is_raining=is_raining,
            is_sheltered=is_sheltered,
            is_cold_exposed=is_cold_exposed,
        )
        self._sync_agent_cache_from_arrays()

    def _death_reason_from_arrays(self, index: int) -> DeathReason:
        needs: dict[DeathReason, float] = {
            DeathReason.THIRST: float(self.agent_arrays.thirst[index]),
            DeathReason.HUNGER: float(self.agent_arrays.hunger[index]),
            DeathReason.EXHAUSTION: float(self.agent_arrays.fatigue[index]),
            DeathReason.COLD: float(self.agent_arrays.cold_stress[index]),
        }
        return max(needs, key=lambda reason: needs[reason])

    def step(self) -> None:
        if self.tick >= self.config.max_ticks():
            return

        environment_start: float = perf_counter()
        clock: SimClock = clock_from_tick(self.tick, self.config)
        raining: bool = self.world.step_environment(self.rng, self.config, self.tick)
        weather: WeatherState = make_weather_state(
            is_raining=raining,
            is_night=clock.is_night,
            config=self.config,
        )
        self.current_weather = weather
        if raining and self.tick % 12 == 0:
            self._log("weather", "rain fell")
        self._log_weather_transition(weather)
        self._record_timing("environment", perf_counter() - environment_start)

        if self.agent.alive:
            self._step_agent(clock, weather)

        self.tick += 1

    def run(self, snapshot_every: int = 0) -> SimResult:
        while self.tick < self.config.max_ticks() and self.agent.alive:
            self.step()
            if snapshot_every > 0 and self.tick % snapshot_every == 0:
                snapshot_start: float = perf_counter()
                self.snapshots.append(self.snapshot(include_ascii=True))
                self._record_timing("snapshots", perf_counter() - snapshot_start)
        aggregation_start: float = perf_counter()
        result = self.result()
        self._record_timing("result_aggregation", perf_counter() - aggregation_start)
        return result

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
        perception_start: float = perf_counter()
        self.agent.ensure_visit_buffer(self.world.width * self.world.height)
        position_index: int = index_of(self.world.width, self.agent.position)
        self.agent.visited_counts[position_index] += 1

        is_sheltered: bool = self.agent_is_sheltered()

        # Batch Perception (Hot Path execution pushing arrays directly)
        self._perception_agent_ids[0] = self.agent.agent_id
        self._perception_agent_x[0] = self.agent.position.x
        self._perception_agent_y[0] = self.agent.position.y
        self._perception_agent_alive[0] = self.agent.alive

        _, p_tiles, p_kinds, p_amounts = perceive_batch_resources(
            self._perception_agent_ids,
            self._perception_agent_x,
            self._perception_agent_y,
            self._perception_agent_alive,
            self.world,
            clock,
            self.config,
            self._perception_out_agent_ids,
            self._perception_out_tile_indices,
            self._perception_out_kinds,
            self._perception_out_amounts,
        )

        water_mask: NDArray[np.bool_] = p_kinds == RESOURCE_KIND_WATER
        food_mask: NDArray[np.bool_] = p_kinds == RESOURCE_KIND_FOOD
        visible_water_indices: NDArray[np.int64] = p_tiles[water_mask]
        visible_food_indices: NDArray[np.int64] = p_tiles[food_mask]

        observation: Observation = perceive(
            self.world,
            self.agent.position,
            clock,
            self.config,
            weather,
            is_sheltered,
        )

        for resource_kind, indices, amounts in (
            (
                ResourceKind.WATER,
                observation.visible_water_indices,
                observation.visible_water_amounts,
            ),
            (
                ResourceKind.FOOD,
                observation.visible_food_indices,
                observation.visible_food_amounts,
            ),
        ):
            for i in range(indices.shape[0]):
                tile_index: int = int(indices[i])
                position: Position = Position(
                    x=tile_index % self.world.width,
                    y=tile_index // self.world.width,
                )
                amount: float = float(amounts[i])
                is_new: bool = self.memory.observe_resource(
                    resource_kind,
                    position,
                    amount,
                    self.tick,
                )
                memory_record: ResourceMemory | None = self._memory_at(
                    resource_kind,
                    position,
                )
                if is_new:
                    if resource_kind is ResourceKind.WATER:
                        self.agent.water_discoveries += 1
                        self.learning.learned_water_sites += 1
                    else:
                        self.agent.food_discoveries += 1
                        self.learning.learned_food_sites += 1
                    self._log_at(
                        "learning",
                        self._learned_message(resource_kind, position),
                        position,
                    )
                elif memory_record is not None:
                    confidence: str = self._format_confidence(
                        memory_record.decayed_confidence(self.tick, self.config)
                    )
                    self._log_at(
                        "learning",
                        self._updated_memory_message(
                            resource_kind, position, confidence
                        ),
                        position,
                    )

        visible_water_indices = observation.visible_water_indices
        visible_food_indices = observation.visible_food_indices

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
        self._record_timing("perception", perf_counter() - perception_start)

        if self.config.enable_goap_control:
            goap_start: float = perf_counter()
            goal = self._urgent_goap_goal()
            if goal is not None:
                results = self.execute_goap_plan(goal)
                self._record_timing("goap", perf_counter() - goap_start)
                if results and all(result.success for result in results):
                    self._sync_memory_markers()
                    if not self.agent.alive:
                        self._log_agent_death()
                    return
            else:
                self._record_timing("goap", perf_counter() - goap_start)

        policy_start: float = perf_counter()
        interaction_ticks = self._try_record_discoverable_exploitation(
            clock, observation
        )
        if interaction_ticks > 0:
            self._update_agent_needs(
                is_night=clock.is_night,
                is_raining=weather.is_raining,
                is_sheltered=self.agent_is_sheltered(),
                is_cold_exposed=weather.feels_cold,
            )
            self._log_cold_status_transition()
            self._advance_interaction_ticks(interaction_ticks)
            self._sync_memory_markers()
            self._record_timing("policy_pathing", perf_counter() - policy_start)
            if not self.agent.alive:
                self._log_agent_death()
            return

        before_memory_state: dict[tuple[str, int, int], tuple[int, int]] = (
            self._memory_use_state()
        )

        # Policy is entirely separated from the legacy Observation object
        action_message: str = choose_and_execute_action(
            self.agent,
            self.memory,
            visible_water_indices,
            visible_food_indices,
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

        self._update_agent_needs(
            is_night=clock.is_night,
            is_raining=weather.is_raining,
            is_sheltered=self.agent_is_sheltered(),
            is_cold_exposed=weather.feels_cold,
        )
        self._log_cold_status_transition()
        self._record_timing("policy_pathing", perf_counter() - policy_start)
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

        snapshot_start: float = perf_counter()
        before_snapshot = make_state_snapshot(
            tick=self.tick,
            agent=self.agent,
            observation=observation,
            discoverable_memory=self.discoverable_memory,
            clock=clock,
        )
        self._record_timing("snapshots", perf_counter() - snapshot_start)
        if item.kind is DiscoverableKind.CAVE:
            self._log(
                "action", f"{SEEKING_SHELTER_ACTION_PREFIX}{item.discoverable_id}"
            )
        success: bool = exploit_discoverable(self.agent, item)
        after_tick: int = self.tick + item.interaction_ticks
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
        snapshot_start = perf_counter()
        after_snapshot = make_state_snapshot(
            tick=after_tick,
            agent=self.agent,
            observation=after_observation,
            discoverable_memory=self.discoverable_memory,
            clock=clock,
        )
        self._record_timing("snapshots", perf_counter() - snapshot_start)
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
            environment_start: float = perf_counter()
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
            self._record_timing("environment", perf_counter() - environment_start)
            self._update_agent_needs(
                is_night=busy_clock.is_night,
                is_raining=weather.is_raining,
                is_sheltered=self.agent_is_sheltered(),
                is_cold_exposed=weather.feels_cold,
            )
            self._log_cold_status_transition()
            extra_ticks -= 1
            if not self.agent.alive:
                return

    def _record_timing(self, category: TimingCategory, seconds: float) -> None:
        if self.profiler is not None:
            self.profiler.add(category, seconds)

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
            successful_delta = memory.successful_uses - before_successes
            failed_delta = memory.failed_uses - before_failures
            if successful_delta <= 0 and failed_delta <= 0:
                continue

            confidence = self._format_confidence(
                memory.decayed_confidence(self.tick, self.config)
            )

            if successful_delta > 0:
                if memory.kind is ResourceKind.WATER:
                    self.learning.memory_reinforced_water += successful_delta
                else:
                    self.learning.memory_reinforced_food += successful_delta
                self._log_at(
                    "learning",
                    self._reinforced_memory_message(memory, confidence),
                    memory.position,
                )
            if failed_delta > 0:
                if memory.kind is ResourceKind.WATER:
                    self.learning.memory_failed_water += failed_delta
                else:
                    self.learning.memory_failed_food += failed_delta
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


FounderObservation = NDArray[np.float32]
FounderInfo = dict[str, int | float]
FounderStepResult = tuple[FounderObservation, float, bool, bool, FounderInfo]
FounderResetResult = tuple[FounderObservation, FounderInfo]


class FounderTrainingEnv:
    """Gymnasium-compatible vectorized Founder training environment.

    The class intentionally implements the Env protocol without importing
    gymnasium at module import time, keeping the core simulator usable in lean
    runtime deployments while preserving Ray RLlib's reset/step tuple contract.
    """

    metadata: dict[str, object] = {"render_modes": ()}

    def __init__(
        self,
        *,
        width: int,
        height: int,
        max_steps: int,
        seed: int,
        receptive_field: int,
    ) -> None:
        if receptive_field <= 0 or receptive_field % 2 == 0:
            raise ValueError("receptive_field must be a positive odd integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.receptive_field = receptive_field
        self._radius = receptive_field // 2
        self.grids = WorldGrids(width=width, height=height)
        self._rng = np.random.default_rng(seed)
        self._seed: int = seed
        self._tick: int = 0
        self._agent_x: np.int32 = np.int32(width // 2)
        self._agent_y: np.int32 = np.int32(height // 2)
        self._obs: FounderObservation = np.empty(
            (6, receptive_field, receptive_field), dtype=np.float32
        )
        self._padded: FounderObservation = np.empty(
            (6, height + receptive_field - 1, width + receptive_field - 1),
            dtype=np.float32,
        )
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> FounderResetResult:
        """Reset grid buffers in place and return the initial local tensor."""

        del options
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        self._tick = 0
        self.grids.reset()
        random_values: NDArray[np.float32] = self._rng.random(
            self.grids.cell_count, dtype=np.float32
        )
        forest_mask: NDArray[np.bool_] = random_values > np.float32(0.72)
        water_mask: NDArray[np.bool_] = random_values < np.float32(0.08)
        hill_mask: NDArray[np.bool_] = (
            random_values >= np.float32(0.55)
        ) & ~forest_mask
        self.grids.terrain_kind[forest_mask] = np.int32(TerrainKind.FOREST)
        self.grids.terrain_kind[water_mask] = np.int32(TerrainKind.WATER)
        self.grids.terrain_kind[hill_mask] = np.int32(TerrainKind.HILL)
        self.grids.elevation[:] = random_values
        self.grids.water_table[:] = np.where(
            water_mask,
            np.float32(0.9),
            np.float32(0.25) + random_values * np.float32(0.25),
        ).astype(np.float32)
        self.grids.crop_growth.fill(np.float32(0.0))
        self.grids.structure_kind.fill(np.int32(0))
        self.grids.structure_health.fill(np.float32(0.0))
        self._agent_x = np.int32(self.width // 2)
        self._agent_y = np.int32(self.height // 2)
        return self._observation(), self._info(np.float32(0.0))

    def step(
        self, action_array: int | np.int32 | NDArray[np.int32]
    ) -> FounderStepResult:
        """Apply vectorized environmental mutations and return an RLlib tuple."""

        actions: NDArray[np.int32] = normalize_action_payload(action_array)
        if actions.shape[1] >= 3:
            action_kind: NDArray[np.int32] = actions[:, 0]
            x_coords: NDArray[np.int32] = np.asarray(
                np.clip(actions[:, 1], 0, self.width - 1), dtype=np.int32
            )
            y_coords: NDArray[np.int32] = np.asarray(
                np.clip(actions[:, 2], 0, self.height - 1), dtype=np.int32
            )
        else:
            action_kind = actions[:, 0]
            x_coords = np.full(actions.shape[0], self._agent_x, dtype=np.int32)
            y_coords = np.full(actions.shape[0], self._agent_y, dtype=np.int32)

        indices: NDArray[np.int32] = y_coords * np.int32(self.width) + x_coords
        terrain: NDArray[np.int32] = self.grids.terrain_kind[indices]
        structure: NDArray[np.int32] = self.grids.structure_kind[indices]

        chop_mask: NDArray[np.bool_] = (action_kind == CHOP) & (
            terrain == np.int32(TerrainKind.FOREST)
        )
        dig_mask: NDArray[np.bool_] = (action_kind == DIG) & (
            terrain != np.int32(TerrainKind.ROCK)
        )
        plant_mask: NDArray[np.bool_] = (action_kind == PLANT) & (
            terrain == np.int32(TerrainKind.GRASS)
        )
        build_mask: NDArray[np.bool_] = (action_kind == BUILD) & (structure == 0)

        self.grids.terrain_kind[indices[chop_mask]] = np.int32(TerrainKind.GRASS)
        self.grids.water_table[indices[dig_mask]] = np.minimum(
            np.float32(1.0),
            self.grids.water_table[indices[dig_mask]] + np.float32(0.18),
        )
        self.grids.crop_growth[indices[plant_mask]] = np.maximum(
            self.grids.crop_growth[indices[plant_mask]], np.float32(0.25)
        )
        self.grids.structure_kind[indices[build_mask]] = STRUCTURE_SHELTER
        self.grids.structure_health[indices[build_mask]] = np.float32(1.0)
        self.grids.crop_growth[:] = np.minimum(
            np.float32(1.0), self.grids.crop_growth + np.float32(0.01)
        )

        reward_value: np.float32 = np.float32(
            chop_mask.sum(dtype=np.int32) * np.int32(1)
            + dig_mask.sum(dtype=np.int32) * np.int32(2)
            + plant_mask.sum(dtype=np.int32) * np.int32(2)
            + build_mask.sum(dtype=np.int32) * np.int32(3)
        )
        if actions.shape[0] > 0:
            self._agent_x = np.int32(x_coords[-1])
            self._agent_y = np.int32(y_coords[-1])
        self._tick += 1
        terminated: bool = bool(self._mission_complete())
        truncated: bool = self._tick >= self.max_steps
        return (
            self._observation(),
            float(reward_value),
            terminated,
            truncated,
            self._info(reward_value),
        )

    def _mission_complete(self) -> bool:
        farm_count: int = int(np.count_nonzero(self.grids.crop_growth >= 1.0))
        shelter_count: int = int(np.count_nonzero(self.grids.structure_kind > 0))
        water_count: int = int(np.count_nonzero(self.grids.water_table >= 0.85))
        return farm_count >= 3 and shelter_count >= 1 and water_count >= 1

    def _observation(self) -> FounderObservation:
        terrain_plane: NDArray[np.float32] = self.grids.terrain_kind.reshape(
            self.height, self.width
        ).astype(np.float32, copy=False)
        structure_plane: NDArray[np.float32] = self.grids.structure_kind.reshape(
            self.height, self.width
        ).astype(np.float32, copy=False)
        elevation_plane: NDArray[np.float32] = self.grids.elevation.reshape(
            self.height, self.width
        )
        health_plane: NDArray[np.float32] = self.grids.structure_health.reshape(
            self.height, self.width
        )
        crop_plane: NDArray[np.float32] = self.grids.crop_growth.reshape(
            self.height, self.width
        )
        water_plane: NDArray[np.float32] = self.grids.water_table.reshape(
            self.height, self.width
        )
        self._padded.fill(np.float32(0.0))
        y_start: int = self._radius
        y_end: int = y_start + self.height
        x_start: int = self._radius
        x_end: int = x_start + self.width
        self._padded[0, y_start:y_end, x_start:x_end] = terrain_plane
        self._padded[1, y_start:y_end, x_start:x_end] = structure_plane
        self._padded[2, y_start:y_end, x_start:x_end] = elevation_plane
        self._padded[3, y_start:y_end, x_start:x_end] = health_plane
        self._padded[4, y_start:y_end, x_start:x_end] = crop_plane
        self._padded[5, y_start:y_end, x_start:x_end] = water_plane
        crop_y: int = int(self._agent_y)
        crop_x: int = int(self._agent_x)
        self._obs[:] = self._padded[
            :,
            crop_y : crop_y + self.receptive_field,
            crop_x : crop_x + self.receptive_field,
        ]
        return self._obs.copy()

    def _info(self, reward_value: np.float32) -> FounderInfo:
        return {
            "tick": self._tick,
            "agent_x": int(self._agent_x),
            "agent_y": int(self._agent_y),
            "reward": float(reward_value),
        }
