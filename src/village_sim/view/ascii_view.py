"""Semantic map renderer with plain-text output compatibility.

This renderer is intentionally read-only. It consumes world and agent state but does
not own simulation truth. CLI and replay callers use the safe plain-text wrapper;
GUI callers can use the semantic glyph rows for colored rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

from village_sim.agent.state import AgentState
from village_sim.core.types import ActionKind, Position, ResourceKind, TerrainKind
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
    "water": "#4fc3f7",
    "remembered_water": "#26c6da",
    "remembered_food": "#ec407a",
    "stale_memory": "#8d7a65",
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


def render_map_model(
    world: World, agent: AgentState, radius: int | None = None
) -> RenderedMap:
    """Render a semantic top-down map model."""

    min_x: int = 0
    max_x: int = world.width - 1
    min_y: int = 0
    max_y: int = world.height - 1
    if radius is not None:
        min_x = max(0, agent.position.x - radius)
        max_x = min(world.width - 1, agent.position.x + radius)
        min_y = max(0, agent.position.y - radius)
        max_y = min(world.height - 1, agent.position.y + radius)

    rows: list[list[MapGlyph]] = []
    for y in range(min_y, max_y + 1):
        row: list[MapGlyph] = []
        for x in range(min_x, max_x + 1):
            row.append(_glyph_for_position(world, agent, Position(x=x, y=y)))
        rows.append(row)

    tile_scale: str = _format_tile_scale(world.tile_size_meters)
    legend: str = (
        "Legend: @ agent, z sleeping, ~ stream/water, * food/berries, "
        "W/w remembered water, F/f remembered food, "
        '♣ broadleaf, ♠ evergreen, . short grass, , uneven grass, " brush, '
        "; wetland/reeds, ^ hill/elevation, # rock, C cave | "
        f"Scale: 1 tile ≈ {tile_scale} x {tile_scale}"
    )
    decision_status: str = _decision_status(agent)
    status: str = (
        f"Agent: x={agent.position.x} y={agent.position.y} "
        f"goal={agent.current_goal.value} action={agent.current_action.value} "
        f"health={agent.health:.2f} thirst={agent.thirst:.2f} "
        f"hunger={agent.hunger:.2f} fatigue={agent.fatigue:.2f}"
        f"{decision_status}"
    )
    return RenderedMap(status=status, legend=legend, rows=rows)


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


def _glyph_for_position(
    world: World, agent: AgentState, position: Position
) -> MapGlyph:
    if position == agent.position:
        if agent.current_action is ActionKind.SLEEP:
            return _glyph("z", "agent_sleeping")
        return _glyph("@", "agent")

    discoverable_glyph: MapGlyph | None = _discoverable_glyph_at(world, position)
    if discoverable_glyph is not None:
        return discoverable_glyph

    index: int = index_of(world.width, position)
    if world.food[index] >= 0.18:
        return _glyph("*", "food")
    kind: TerrainKind = TerrainKind(world.terrain[index])
    if kind is TerrainKind.WATER:
        return _glyph("~", "water")
    memory_glyph: MapGlyph | None = _memory_glyph_at(agent, position)
    if memory_glyph is not None:
        return memory_glyph
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


def _memory_glyph_at(agent: AgentState, position: Position) -> MapGlyph | None:
    for marker in agent.memory_markers:
        if marker.position != position:
            continue
        if marker.confidence < 0.18:
            return _glyph("?", "stale_memory")
        if marker.kind is ResourceKind.WATER:
            char = "W" if marker.confidence >= 0.75 else "w"
            return _glyph(char, "remembered_water")
        if marker.kind is ResourceKind.FOOD:
            char = "F" if marker.confidence >= 0.75 else "f"
            return _glyph(char, "remembered_food")
    return None


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
