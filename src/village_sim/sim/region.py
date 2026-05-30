"""Ray actor wrapper for deterministic regional simulation chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import ray
from numpy.typing import NDArray

from village_sim.agent.handoff import (
    RegionBounds,
    group_departing_agents_by_neighbor,
    receive_agent_handoff,
)
from village_sim.agent.memory import GlobalMemory
from village_sim.agent.state import AgentArrays, make_agent_arrays, sync_agent_to_arrays
from village_sim.core.config import SimConfig
from village_sim.sim.engine import Simulation
from village_sim.world.grid import WorldGrids

NeighborDelta = tuple[int, int]
RegionActorHandleMap = dict[NeighborDelta, Any]


@dataclass(frozen=True, slots=True)
class RegionTickResult:
    """Deterministic summary returned to the external heartbeat driver."""

    region_id: str
    tick: int
    active_agents: int
    outgoing_agents: int


@ray.remote
class RegionActor:
    """Own one static map chunk and exchange boundary agents by message passing."""

    def __init__(
        self,
        region_id: str,
        config: SimConfig,
        bounds: RegionBounds,
        agent_capacity: int,
    ) -> None:
        if agent_capacity <= 0:
            raise ValueError("agent_capacity must be positive")
        self.region_id = region_id
        self.bounds = bounds
        local_config = replace(config, width=bounds.width, height=bounds.height)
        self.sim = Simulation(local_config)
        self.grids = WorldGrids(bounds.width, bounds.height)
        self.grids.copy_from_world_arrays(
            terrain=np.asarray(self.sim.world.terrain),
            elevation=np.asarray(self.sim.world.height_map),
            water=np.asarray(self.sim.world.water),
            food=np.asarray(self.sim.world.food),
        )
        if agent_capacity != self.sim.agents.count:
            self.sim.agents = make_agent_arrays(agent_capacity)
            self.sim.agent_arrays = self.sim.agents
            self.sim.state.agent_arrays = self.sim.agents
            sync_agent_to_arrays(self.sim.agents, self.sim.agent, 0)
            self.sim._init_perception_buffers()
        self.agent_ids = np.zeros(self.sim.agents.count, dtype=np.int64)
        self.agent_ids[0] = np.int64(1)
        global_memory = self.sim.memory.global_memory
        if global_memory is None:
            raise RuntimeError("simulation memory was not initialized")
        self.global_memory = global_memory
        self.expected_tick = 0

    def tick(
        self,
        expected_tick: int,
        neighbor_actors: RegionActorHandleMap,
    ) -> RegionTickResult:
        """Advance exactly one heartbeat and enqueue handoffs to neighbors."""

        if expected_tick != self.expected_tick:
            raise ValueError("region heartbeat tick mismatch")
        self.sim.step()
        outgoing = group_departing_agents_by_neighbor(
            self.sim.agents,
            self.global_memory,
            self.bounds,
            self.agent_ids,
        )
        outgoing_count = 0
        for delta, buffer in outgoing.items():
            outgoing_count += int(buffer.size > 0)
            target_actor = neighbor_actors.get(delta)
            if target_actor is not None:
                target_actor.receive_agents.remote(buffer)
        self.expected_tick += 1
        active_agents = int(np.count_nonzero(self.sim.agents.active))
        return RegionTickResult(
            region_id=self.region_id,
            tick=self.expected_tick,
            active_agents=active_agents,
            outgoing_agents=outgoing_count,
        )

    def receive_agents(self, buffer: NDArray[np.uint8]) -> int:
        """Receive a MessagePack handoff buffer from an adjacent region actor."""

        return receive_agent_handoff(
            self.sim.agents,
            self.global_memory,
            self.bounds,
            self.agent_ids,
            buffer,
        )

    def active_agent_count(self) -> int:
        """Return active local agents in O(n) for driver diagnostics."""

        return int(np.count_nonzero(self.sim.agents.active))

    def snapshot_agent_arrays(self) -> AgentArrays:
        """Return local agent arrays for deterministic testing and diagnostics."""

        return self.sim.agents

    def snapshot_memory(self) -> GlobalMemory:
        """Return local Polars-backed memory for deterministic diagnostics."""

        return self.global_memory
