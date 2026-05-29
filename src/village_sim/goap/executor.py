"""Execution payload dispatcher and minimal live GOAP plan executor (§19)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from village_sim.agent.memory import DiscoverableAgentMemory
from village_sim.agent.needs import update_needs
from village_sim.agent.perception import Observation, perceive
from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.time import SimClock, clock_from_tick
from village_sim.core.types import ActionKind, Position, PrimitiveAction
from village_sim.goap.planner import PlanStep
from village_sim.orchestrator.action_model import ExecutionPayload, ExecutorType
from village_sim.orchestrator.orchestrator import Orchestrator
from village_sim.orchestrator.snapshotting import make_state_snapshot
from village_sim.orchestrator.trajectory import Trajectory, TrajectoryRecorder
from village_sim.orchestrator.travel import should_record_travel_segment
from village_sim.world.discoverables import (
    Discoverable,
    update_discoverable_memory,
)
from village_sim.world.grid import iter_neighbor_positions
from village_sim.world.world import World


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    death: bool
    timeout: bool
    ticks_elapsed: int
    trajectory: Trajectory | None
    message: str


class PolicyRunner(Protocol):
    """Minimal interface a concrete policy must satisfy."""

    def run(self, target_id: str | None) -> str:
        """Execute one step of the policy; return a log message."""
        ...


class _SimulationSurface(Protocol):
    world: World
    agent: AgentState
    config: SimConfig
    tick: int
    discoverable_memory: DiscoverableAgentMemory
    orchestrator: Orchestrator
    recorded_trajectories: list[Trajectory]
    rng: random.Random

    def record_discoverable_exploitation(
        self,
        clock: SimClock,
        observation: Observation,
    ) -> int: ...

    def advance_interaction_ticks(self, interaction_ticks: int) -> None: ...


class ExecutorRegistry:
    """Maps policy_id strings to live PolicyRunner objects."""

    def __init__(self) -> None:
        self._runners: dict[str, PolicyRunner] = {}

    def register(self, policy_id: str, runner: PolicyRunner) -> None:
        self._runners[policy_id] = runner

    def dispatch(self, payload: ExecutionPayload) -> str:
        """Invoke the executor referenced by payload and return a log message."""
        if payload.type is ExecutorType.SCRIPTED_PRIMITIVE:
            return f"primitive:{payload.policy_id}"

        runner = self._runners.get(payload.policy_id)
        if runner is None:
            return f"no runner registered for policy_id={payload.policy_id}"

        target_id = payload.target_binding.resource_id
        return runner.run(target_id)


class PlanExecutor:
    """Execute the MVP subset of synthesized GOAP actions in a live simulation."""

    def __init__(self, sim: _SimulationSurface) -> None:
        self._sim = sim

    def execute_step(self, step: PlanStep) -> ExecutionResult:
        payload = step.action.execution_payload
        if payload.type is ExecutorType.PATHFINDER:
            return self._execute_pathfinder(step)
        if payload.type is ExecutorType.SCRIPTED_PRIMITIVE:
            return self._execute_scripted_primitive(step)
        return ExecutionResult(
            success=False,
            death=False,
            timeout=False,
            ticks_elapsed=0,
            trajectory=None,
            message=f"unsupported executor type {payload.type.value}",
        )

    def execute_plan(self, steps: list[PlanStep]) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for step in steps:
            result = self.execute_step(step)
            results.append(result)
            if not result.success or result.death or result.timeout:
                break
        return results

    def _execute_pathfinder(self, step: PlanStep) -> ExecutionResult:
        target = self._resolve_target(step.action.execution_payload)
        if target is None:
            return ExecutionResult(
                success=False,
                death=False,
                timeout=False,
                ticks_elapsed=0,
                trajectory=None,
                message="pathfinder target could not be resolved",
            )

        sim = self._sim
        start_tick = sim.tick
        start_clock = clock_from_tick(sim.tick, sim.config)
        start_observation = perceive(
            sim.world, sim.agent.position, start_clock, sim.config
        )
        update_discoverable_memory(
            sim.discoverable_memory,
            start_observation.discoverables,
            sim.tick,
        )
        before_snapshot = make_state_snapshot(
            tick=sim.tick,
            agent=sim.agent,
            observation=start_observation,
            discoverable_memory=sim.discoverable_memory,
            clock=start_clock,
        )

        max_ticks = self._max_pathfinder_ticks(step, target)
        timed_out = False
        while _chebyshev(sim.agent.position, Position(target.x, target.y)) > 1:
            if sim.tick - start_tick >= max_ticks:
                timed_out = True
                break
            next_position = _choose_greedy_step(sim.world, sim.agent.position, target)
            if next_position is None:
                break
            sim.agent.position = next_position
            sim.agent.distance_walked += 1
            sim.agent.current_action = ActionKind.MOVE
            sim.tick += 1
            clock = clock_from_tick(sim.tick, sim.config)
            sim.world.step_environment(sim.rng, sim.config, clock.tick_of_day)
            update_needs(sim.agent, sim.config)
            if not sim.agent.alive:
                break

        end_clock = clock_from_tick(sim.tick, sim.config)
        end_observation = perceive(sim.world, sim.agent.position, end_clock, sim.config)
        update_discoverable_memory(
            sim.discoverable_memory,
            end_observation.discoverables,
            sim.tick,
        )
        after_snapshot = make_state_snapshot(
            tick=sim.tick,
            agent=sim.agent,
            observation=end_observation,
            discoverable_memory=sim.discoverable_memory,
            clock=end_clock,
        )
        success = should_record_travel_segment(before_snapshot, after_snapshot)
        trajectory: Trajectory | None = None
        if success:
            recorder = TrajectoryRecorder(
                trajectory_id=f"traj_travel_{sim.agent.agent_id}_{start_tick}_{target.discoverable_id}",
                policy_id="policy_move_to_known_discoverable_v1",
                task_name="travel",
            )
            recorder.record(
                before=before_snapshot,
                action=PrimitiveAction.MOVE_TO_TARGET,
                after=after_snapshot,
                reward=1.0,
                events=[f"travel:{target.discoverable_id}:success"],
            )
            trajectory = recorder.finish()

        return ExecutionResult(
            success=success,
            death=not sim.agent.alive,
            timeout=timed_out,
            ticks_elapsed=sim.tick - start_tick,
            trajectory=trajectory,
            message=(
                f"moved to {target.discoverable_id}" if success else "travel failed"
            ),
        )

    def _execute_scripted_primitive(self, step: PlanStep) -> ExecutionResult:
        sim = self._sim
        start_tick = sim.tick
        before_count = len(sim.recorded_trajectories)
        clock = clock_from_tick(sim.tick, sim.config)
        observation = perceive(sim.world, sim.agent.position, clock, sim.config)
        interaction_ticks = sim.record_discoverable_exploitation(clock, observation)
        if interaction_ticks <= 0:
            return ExecutionResult(
                success=False,
                death=not sim.agent.alive,
                timeout=False,
                ticks_elapsed=0,
                trajectory=None,
                message="scripted primitive did not exploit a discoverable",
            )

        sim.advance_interaction_ticks(interaction_ticks)
        if sim.tick < sim.config.max_ticks():
            sim.tick += 1
        trajectory: Trajectory | None = None
        if len(sim.recorded_trajectories) > before_count:
            trajectory = sim.recorded_trajectories[-1]
        return ExecutionResult(
            success=trajectory is not None,
            death=not sim.agent.alive,
            timeout=False,
            ticks_elapsed=sim.tick - start_tick,
            trajectory=trajectory,
            message=f"executed {step.action.action_id}",
        )

    def _resolve_target(self, payload: ExecutionPayload) -> Discoverable | None:
        binding = payload.target_binding
        if binding.mode == "resource_id":
            if binding.resource_id is None:
                return None
            return self._sim.world.discoverables.get(binding.resource_id)
        if binding.mode != "current_target":
            return None

        clock = clock_from_tick(self._sim.tick, self._sim.config)
        observation = perceive(
            self._sim.world, self._sim.agent.position, clock, self._sim.config
        )
        symbolic = make_state_snapshot(
            tick=self._sim.tick,
            agent=self._sim.agent,
            observation=observation,
            discoverable_memory=self._sim.discoverable_memory,
            clock=clock,
        ).symbolic
        target_id = symbolic.get("target_id")
        if isinstance(target_id, str) and target_id != "none":
            target = self._sim.world.discoverables.get(target_id)
            if target is not None and _matches_required_type(
                target, binding.required_type
            ):
                return target

        for memory in self._sim.discoverable_memory.discoverables.values():
            if str(memory.kind) != binding.required_type:
                continue
            target = self._sim.world.discoverables.get(memory.discoverable_id)
            if target is not None:
                return target
        return None

    def _max_pathfinder_ticks(self, step: PlanStep, target: Discoverable) -> int:
        target_position = Position(target.x, target.y)
        distance = self._sim.agent.position.manhattan_to(target_position)
        model = step.action.cost_model
        expected = model.base_ticks + model.distance_weight * float(distance)
        return max(8, int(expected * 4.0) + 16)


def _matches_required_type(target: Discoverable, required_type: str | None) -> bool:
    return required_type is None or str(target.kind) == required_type


def _chebyshev(left: Position, right: Position) -> int:
    return max(abs(left.x - right.x), abs(left.y - right.y))


def _choose_greedy_step(
    world: World,
    position: Position,
    target: Discoverable,
) -> Position | None:
    target_position = Position(target.x, target.y)
    candidates: list[Position] = []
    for neighbor in iter_neighbor_positions(world.width, world.height, position, False):
        if world.is_passable(neighbor):
            candidates.append(neighbor)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item.manhattan_to(target_position),
            world.movement_cost(item),
            item.y,
            item.x,
        )
    )
    return candidates[0]
