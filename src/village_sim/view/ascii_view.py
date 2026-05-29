"""ASCII debug renderer.

This renderer is intentionally read-only. It consumes world and agent state but does
not own simulation truth.
"""

from __future__ import annotations

from village_sim.agent.state import AgentState
from village_sim.core.types import ActionKind, Position, TerrainKind
from village_sim.world.grid import index_of
from village_sim.world.world import World


def render_ascii_map(world: World, agent: AgentState, radius: int | None = None) -> str:
    """Render a compact top-down ASCII map."""

    min_x: int = 0
    max_x: int = world.width - 1
    min_y: int = 0
    max_y: int = world.height - 1
    if radius is not None:
        min_x = max(0, agent.position.x - radius)
        max_x = min(world.width - 1, agent.position.x + radius)
        min_y = max(0, agent.position.y - radius)
        max_y = min(world.height - 1, agent.position.y + radius)

    lines: list[str] = []
    for y in range(min_y, max_y + 1):
        chars: list[str] = []
        for x in range(min_x, max_x + 1):
            position: Position = Position(x=x, y=y)
            if position == agent.position:
                chars.append("z" if agent.current_action is ActionKind.SLEEP else "@")
                continue
            index: int = index_of(world.width, position)
            if world.food[index] >= 0.18:
                chars.append("*")
                continue
            if world.water[index] >= 0.25:
                chars.append("~")
                continue
            kind: TerrainKind = TerrainKind(world.terrain[index])
            if kind is TerrainKind.WATER:
                chars.append("~")
            elif kind is TerrainKind.GRASS:
                chars.append(".")
            elif kind is TerrainKind.FOREST:
                chars.append("T")
            elif kind is TerrainKind.HILL:
                chars.append("^")
            else:
                chars.append("#")
        lines.append("".join(chars))
    legend: str = (
        "Legend: @ agent, z sleeping, ~ water, * food, T forest, ^ hill, # rock, . grass"
    )
    status: str = (
        f"Agent: x={agent.position.x} y={agent.position.y} "
        f"goal={agent.current_goal.value} action={agent.current_action.value} "
        f"health={agent.health:.2f} thirst={agent.thirst:.2f} "
        f"hunger={agent.hunger:.2f} fatigue={agent.fatigue:.2f}"
    )
    return "\n".join([status, legend, *lines])
