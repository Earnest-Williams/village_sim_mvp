"""Semantic map renderer with plain-text output compatibility.

This renderer is intentionally read-only. It consumes world and agent state but does
not own simulation truth. CLI and replay callers use the safe plain-text wrapper;
GUI callers can use the semantic glyph rows for colored rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from village_sim.agent.state import (
    AgentArrays,
    AgentState,
    ID_TO_ACTION,
    ID_TO_GOAL,
    make_agent_arrays,
    sync_agent_to_arrays,
)
from village_sim.core.types import ActionKind, Position, TerrainKind
from village_sim.world.discoverables import DiscoverableKind
from village_sim.world.grid import index_of, iter_neighbor_positions
from village_sim.world.world import World


@dataclass(frozen=True, slots=True)
class MapGlyph:
    """One rendered map glyph plus semantic styling metadata."""

    char: str
    role: str
    fg: str | None = None
    bg: str | None = None


@dataclass(frozen=True, slots=True)
class RenderedMap:
    """Structured map output for text and colored GUI renderers."""

    status: str
    legend: str
    rows: list[list[MapGlyph]]


ROLE_COLORS: dict[str, str] = {
    "agent": "#fff176",
    "agent_sleeping": "#d7b85b",
    "agent_other": "#ffe082",
    "agent_crowd": "#ffcc80",
    "water": "#4fc3f7",
    "broadleaf": "#66bb6a",
    "evergreen": "#2e7d32",
    "grass": "#a5d66a",
    "brush": "#c0a447",
    "wetland": "#4db6ac",
    "food": "#ff5c8a",
    "rock": "#b0aaa0",
    "cave": "#d7ccc8",
    "hill": "#d0a85f",
}


def render_agent_arrays_map_model(
    world: World,
    agents: AgentArrays,
    selected_agent_index: int,
    radius: int | None = None,
    *,
    tick: int = 0,
    day: int = 0,
    temperature_c: float = 0.0,
    is_raining: bool = False,
    feels_cold: bool = False,
) -> RenderedMap:
    """Render a semantic map model from population SoA state."""

    active_mask: NDArray[np.bool_] = agents.active
    selected_index: int = _selected_active_index(agents, selected_agent_index)
    center_x: int = (
        int(agents.x[selected_index]) if selected_index >= 0 else world.width // 2
    )
    center_y: int = (
        int(agents.y[selected_index]) if selected_index >= 0 else world.height // 2
    )
    min_x: int = 0
    max_x: int = world.width - 1
    min_y: int = 0
    max_y: int = world.height - 1
    if radius is not None:
        min_x = max(0, center_x - radius)
        max_x = min(world.width - 1, center_x + radius)
        min_y = max(0, center_y - radius)
        max_y = min(world.height - 1, center_y + radius)

    world_size: int = world.width * world.height
    active_tiles: NDArray[np.int64] = agents.y.astype(
        np.int64
    ) * world.width + agents.x.astype(np.int64)
    active_rows: NDArray[np.int64] = np.flatnonzero(active_mask).astype(np.int64)
    occupancy_counts: NDArray[np.int64] = np.bincount(
        active_tiles[active_rows], minlength=world_size
    ).astype(np.int64, copy=False)
    selected_tile: int = -1
    if selected_index >= 0 and bool(agents.active[selected_index]):
        selected_tile = int(active_tiles[selected_index])
    sleeping_tiles: NDArray[np.bool_] = np.zeros(world_size, dtype=np.bool_)
    sleep_action_id: int = next(
        action_id
        for action_id, action in ID_TO_ACTION.items()
        if action is ActionKind.SLEEP
    )
    sleeping_rows: NDArray[np.int64] = active_rows[
        agents.current_action[active_rows] == np.int16(sleep_action_id)
    ]
    sleeping_tiles[active_tiles[sleeping_rows]] = True

    rows: list[list[MapGlyph]] = []
    for y in range(min_y, max_y + 1):
        row: list[MapGlyph] = []
        for x in range(min_x, max_x + 1):
            tile_index: int = y * world.width + x
            row.append(
                _glyph_for_population_position(
                    world,
                    Position(x=x, y=y),
                    tile_index,
                    occupancy_counts,
                    sleeping_tiles,
                    selected_tile,
                )
            )
        rows.append(row)

    return RenderedMap(
        status=_population_status(
            agents,
            selected_index,
            tick=tick,
            day=day,
            temperature_c=temperature_c,
            is_raining=is_raining,
            feels_cold=feels_cold,
        ),
        legend=_legend(world),
        rows=rows,
    )


def render_map_model(
    world: World, agent: AgentState, radius: int | None = None
) -> RenderedMap:
    """Render a semantic top-down map model for one compatibility agent."""

    arrays: AgentArrays = make_agent_arrays(1)
    sync_agent_to_arrays(arrays, agent, 0)
    rendered: RenderedMap = render_agent_arrays_map_model(
        world, arrays, 0, radius=radius
    )
    decision_status: str = _decision_status(agent)
    status: str = (
        f"Agent: x={agent.position.x} y={agent.position.y} "
        f"goal={agent.current_goal.value} action={agent.current_action.value} "
        f"health={agent.health:.2f} thirst={agent.thirst:.2f} "
        f"hunger={agent.hunger:.2f} fatigue={agent.fatigue:.2f}"
        f"{decision_status}"
    )
    return RenderedMap(status=status, legend=rendered.legend, rows=rendered.rows)


def render_ascii_map(world: World, agent: AgentState, radius: int | None = None) -> str:
    """Render a compact plain-text map for CLI and replay output."""

    rendered: RenderedMap = render_map_model(world, agent, radius=radius)
    lines: list[str] = [rendered.status, rendered.legend]
    for row in rendered.rows:
        lines.append("".join(glyph.char for glyph in row))
    return "\n".join(lines)


def rendered_map_to_text(rendered: RenderedMap) -> str:
    """Convert a semantic map to plain text."""

    lines: list[str] = [rendered.status, rendered.legend]
    for row in rendered.rows:
        lines.append("".join(glyph.char for glyph in row))
    return "\n".join(lines)


def _glyph_for_population_position(
    world: World,
    position: Position,
    tile_index: int,
    occupancy_counts: NDArray[np.int64],
    sleeping_tiles: NDArray[np.bool_],
    selected_tile: int,
) -> MapGlyph:
    occupant_count: int = int(occupancy_counts[tile_index])
    if occupant_count > 1:
        return _glyph("+", "agent_crowd")
    if occupant_count == 1:
        if bool(sleeping_tiles[tile_index]):
            return _glyph("z", "agent_sleeping")
        if tile_index == selected_tile:
            return _glyph("@", "agent")
        return _glyph("a", "agent_other")
    return _terrain_glyph_for_position(world, position)


def _selected_active_index(agents: AgentArrays, selected_agent_index: int) -> int:
    if 0 <= selected_agent_index < agents.count and bool(
        agents.active[selected_agent_index]
    ):
        return selected_agent_index
    active_rows: NDArray[np.int64] = np.flatnonzero(agents.active).astype(np.int64)
    if active_rows.size == 0:
        return -1
    return int(active_rows[0])


def _population_status(
    agents: AgentArrays,
    selected_index: int,
    *,
    tick: int,
    day: int,
    temperature_c: float,
    is_raining: bool,
    feels_cold: bool,
) -> str:
    active_agents: int = int(np.count_nonzero(agents.active))
    dead_agents: int = int(np.count_nonzero(agents.death_reason >= np.int16(0)))
    village = (
        f"Village: tick={tick} day={day} active={active_agents} dead={dead_agents} "
        f"temp={temperature_c:.1f}C raining={is_raining} cold={feels_cold}"
    )
    if selected_index < 0:
        return f"{village} | Selected agent: none"
    action = ID_TO_ACTION.get(
        int(agents.current_action[selected_index]), ActionKind.IDLE
    )
    goal = ID_TO_GOAL.get(int(agents.current_goal[selected_index]), None)
    goal_value: str = "unknown" if goal is None else goal.value
    selected = (
        f"Selected agent: id={selected_index + 1} x={int(agents.x[selected_index])} "
        f"y={int(agents.y[selected_index])} goal={goal_value} action={action.value} "
        f"health={float(agents.health[selected_index]):.2f} "
        f"thirst={float(agents.thirst[selected_index]):.2f} "
        f"hunger={float(agents.hunger[selected_index]):.2f} "
        f"fatigue={float(agents.fatigue[selected_index]):.2f} "
        f"cold_stress={float(agents.cold_stress[selected_index]):.2f}"
    )
    return f"{village} | {selected}"


def _legend(world: World) -> str:
    tile_scale: str = _format_tile_scale(world.tile_size_meters)
    return (
        "Legend: @ selected agent, a active agent, z sleeping, + stacked agents, "
        "~ stream/water, * food/berries, "
        '♣ broadleaf, ♠ evergreen, . short grass, , uneven grass, " brush, '
        "; wetland/reeds, ^ hill/elevation, # rock, C cave | "
        f"Scale: 1 tile ≈ {tile_scale} x {tile_scale}"
    )


def _glyph_for_position(
    world: World, agent: AgentState, position: Position
) -> MapGlyph:
    if position == agent.position:
        if agent.current_action is ActionKind.SLEEP:
            return _glyph("z", "agent_sleeping")
        return _glyph("@", "agent")

    return _terrain_glyph_for_position(world, position)


def _terrain_glyph_for_position(world: World, position: Position) -> MapGlyph:
    discoverable_glyph: MapGlyph | None = _discoverable_glyph_at(world, position)
    if discoverable_glyph is not None:
        return discoverable_glyph

    index: int = index_of(world.width, position)
    if world.food[index] >= 0.18:
        return _glyph("*", "food")
    kind: TerrainKind = TerrainKind(world.terrain[index])
    if kind is TerrainKind.WATER:
        return _glyph("~", "water")
    if _is_wetland_edge(world, position):
        return _glyph(";", "wetland")
    if kind is TerrainKind.GRASS:
        return _grass_glyph_for_cell(world, position)
    if kind is TerrainKind.FOREST:
        return _vegetation_glyph_for_cell(world, position)
    if kind is TerrainKind.HILL:
        return _hill_glyph_for_cell(world, position)
    return _glyph("#", "rock")


def _decision_status(agent: AgentState) -> str:
    trace = agent.decision_trace
    if trace.target_x < 0 or trace.target_y < 0 or trace.target_kind == "none":
        if trace.source.value == "none":
            return ""
        return f" decision={trace.source.value}"
    status = (
        f" decision={trace.source.value} "
        f"target={trace.target_kind}@{trace.target_x},{trace.target_y}"
    )
    if trace.memory_confidence > 0.0:
        status = f"{status} conf={trace.memory_confidence:.2f}"
    return status


def _discoverable_glyph_at(world: World, position: Position) -> MapGlyph | None:
    for item in world.discoverables.values():
        if item.x != position.x or item.y != position.y:
            continue
        if item.kind is DiscoverableKind.CAVE:
            return _glyph("C", "cave")
        if item.kind is DiscoverableKind.BERRY_BUSH and item.amount > 0.0:
            return _glyph("*", "food")
        if item.kind is DiscoverableKind.FRESHWATER_SPRING and item.amount > 0.0:
            return _glyph("~", "water")
    return None


def _vegetation_glyph_for_cell(world: World, position: Position) -> MapGlyph:
    noise: float = _cell_noise(world, position, salt=11)
    adjacent_trees: int = _adjacent_tree_noise_count(world, position)
    height_value: float = world.height_at(position)
    if adjacent_trees >= 4:
        return _glyph(
            '"' if noise > 0.48 else ",", "brush" if noise > 0.48 else "grass"
        )
    if noise > 0.76:
        if height_value > 0.58 and _cell_noise(world, position, salt=29) > 0.35:
            return _glyph("♠", "evergreen")
        return _glyph("♣", "broadleaf")
    if noise > 0.48:
        return _glyph('"', "brush")
    if noise > 0.24:
        return _glyph(",", "grass")
    return _glyph(".", "grass")


def _grass_glyph_for_cell(world: World, position: Position) -> MapGlyph:
    noise: float = _cell_noise(world, position, salt=7)
    if noise > 0.78:
        return _glyph('"', "brush")
    if noise > 0.42:
        return _glyph(",", "grass")
    return _glyph(".", "grass")


def _hill_glyph_for_cell(world: World, position: Position) -> MapGlyph:
    height_value: float = world.height_at(position)
    if height_value > 0.86:
        return _glyph("#", "rock")
    return _glyph("^", "hill")


def _is_wetland_edge(world: World, position: Position) -> bool:
    kind: TerrainKind = world.terrain_at(position)
    if kind is TerrainKind.WATER or kind is TerrainKind.ROCK:
        return False
    for neighbor in iter_neighbor_positions(world.width, world.height, position, True):
        neighbor_index: int = index_of(world.width, neighbor)
        neighbor_kind: TerrainKind = TerrainKind(world.terrain[neighbor_index])
        if neighbor_kind is TerrainKind.WATER:
            return True
    return False


def _adjacent_tree_noise_count(world: World, position: Position) -> int:
    count: int = 0
    for neighbor in iter_neighbor_positions(world.width, world.height, position, True):
        neighbor_index: int = index_of(world.width, neighbor)
        if TerrainKind(world.terrain[neighbor_index]) is not TerrainKind.FOREST:
            continue
        if _cell_noise(world, neighbor, salt=11) > 0.76:
            count += 1
    return count


def _cell_noise(world: World, position: Position, *, salt: int) -> float:
    value: int = world.seed & 0xFFFF_FFFF
    value ^= (position.x + 0x9E37_79B9 + (value << 6) + (value >> 2)) & 0xFFFF_FFFF
    value ^= (position.y + 0x85EB_CA6B + (value << 6) + (value >> 2)) & 0xFFFF_FFFF
    value ^= (salt * 0xC2B2_AE35) & 0xFFFF_FFFF
    value = (value ^ (value >> 16)) * 0x7FEB_352D & 0xFFFF_FFFF
    value = (value ^ (value >> 15)) * 0x846C_A68B & 0xFFFF_FFFF
    value ^= value >> 16
    return float(value & 0xFFFF_FFFF) / float(0xFFFF_FFFF)


def _glyph(char: str, role: str) -> MapGlyph:
    return MapGlyph(char=char, role=role, fg=ROLE_COLORS.get(role), bg=None)


def _format_tile_scale(tile_size_meters: float) -> str:
    if tile_size_meters.is_integer():
        return f"{tile_size_meters:.0f}m"
    return f"{tile_size_meters:.1f}m"
