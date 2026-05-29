from __future__ import annotations

import random
import unittest
from collections import deque

from village_sim.agent.state import AgentState
from village_sim.core.config import SimConfig
from village_sim.core.types import ActionKind, Position, TerrainKind
from village_sim.view.ascii_view import render_map_model
from village_sim.view.scroll import clamp_insertion_point
from village_sim.world.discoverables import Discoverable, DiscoverableKind
from village_sim.world.grid import index_of, iter_neighbor_positions
from village_sim.world.world import World, generate_world


class StreamGenerationTests(unittest.TestCase):
    def test_generated_stream_is_connected_edge_to_edge_and_small(self) -> None:
        config = SimConfig(width=64, height=64, seed=23)
        world = generate_world(config, random.Random(config.seed))

        water_positions: set[Position] = {
            Position(x=index % world.width, y=index // world.width)
            for index, terrain_value in enumerate(world.terrain)
            if TerrainKind(terrain_value) is TerrainKind.WATER
        }

        self.assertGreater(len(water_positions), 0)
        self.assertLessEqual(
            len(water_positions), int(world.width * world.height * 0.08)
        )
        self.assertGreaterEqual(_touched_edge_count(world, water_positions), 2)
        self.assertEqual(
            len(_connected_water_component(world, next(iter(water_positions)))),
            len(water_positions),
        )


class MapRenderingTests(unittest.TestCase):
    def test_required_glyphs_and_priorities_render(self) -> None:
        world = _make_rendering_world()
        agent = AgentState(agent_id=1, position=Position(x=0, y=0))
        agent.current_action = ActionKind.IDLE

        chars = _render_chars(world, agent)

        self.assertEqual(chars[0][0], "@")
        self.assertEqual(chars[0][1], "~")
        self.assertEqual(chars[0][2], "*")
        self.assertEqual(chars[0][3], "C")
        self.assertEqual(chars[1][1], ";")
        self.assertIn("^", "".join(chars[2]))
        self.assertIn("#", "".join(chars[2]))

    def test_sleeping_agent_overrides_terrain(self) -> None:
        world = _make_rendering_world()
        agent = AgentState(agent_id=1, position=Position(x=1, y=0))
        agent.current_action = ActionKind.SLEEP

        chars = _render_chars(world, agent)

        self.assertEqual(chars[0][1], "z")

    def test_vegetation_and_grass_palettes_are_varied(self) -> None:
        config = SimConfig(width=48, height=48, seed=3)
        world = generate_world(config, random.Random(config.seed))
        agent = AgentState(agent_id=1, position=Position(x=0, y=0))

        rendered = render_map_model(world, agent)
        chars: set[str] = {glyph.char for row in rendered.rows for glyph in row}

        self.assertTrue({"♣", "♠"} & chars)
        self.assertTrue({".", ",", '"'} & chars)
        self.assertIn(";", chars)

    def test_clamp_insertion_point_for_scroll_preservation(self) -> None:
        self.assertEqual(clamp_insertion_point(-1, 10), 0)
        self.assertEqual(clamp_insertion_point(4, 10), 4)
        self.assertEqual(clamp_insertion_point(12, 10), 10)


def _make_rendering_world() -> World:
    width = 6
    height = 4
    terrain: list[int] = [int(TerrainKind.GRASS) for _ in range(width * height)]
    terrain[index_of(width, Position(x=1, y=0))] = int(TerrainKind.WATER)
    terrain[index_of(width, Position(x=0, y=2))] = int(TerrainKind.HILL)
    terrain[index_of(width, Position(x=1, y=2))] = int(TerrainKind.ROCK)
    terrain[index_of(width, Position(x=2, y=2))] = int(TerrainKind.FOREST)
    water: list[float] = [0.0 for _ in terrain]
    water[index_of(width, Position(x=1, y=0))] = 1.0
    food: list[float] = [0.0 for _ in terrain]
    food[index_of(width, Position(x=2, y=0))] = 0.5
    food_capacity: list[float] = [0.0 for _ in terrain]
    discoverable = Discoverable(
        discoverable_id="cave_test",
        kind=DiscoverableKind.CAVE,
        x=3,
        y=0,
        visible_name="cave",
        discovered=True,
        amount=10_000.0,
        max_amount=10_000.0,
        regrowth_per_day=0.0,
        satisfies_need="cold_stress",
        need_delta=-0.5,
        interaction_ticks=4,
    )
    return World(
        width=width,
        height=height,
        height_map=[0.5 for _ in terrain],
        terrain=terrain,
        water=water,
        food=food,
        food_capacity=food_capacity,
        discoverables={discoverable.discoverable_id: discoverable},
        seed=5,
        tile_size_meters=2.0,
    )


def _render_chars(world: World, agent: AgentState) -> list[list[str]]:
    rendered = render_map_model(world, agent)
    return [[glyph.char for glyph in row] for row in rendered.rows]


def _touched_edge_count(world: World, water_positions: set[Position]) -> int:
    edges: set[str] = set()
    for position in water_positions:
        if position.x == 0:
            edges.add("west")
        if position.x == world.width - 1:
            edges.add("east")
        if position.y == 0:
            edges.add("north")
        if position.y == world.height - 1:
            edges.add("south")
    return len(edges)


def _connected_water_component(world: World, start: Position) -> set[Position]:
    seen: set[Position] = {start}
    queue: deque[Position] = deque([start])
    while queue:
        position = queue.popleft()
        for neighbor in iter_neighbor_positions(
            world.width, world.height, position, False
        ):
            if neighbor in seen:
                continue
            terrain_value = world.terrain[index_of(world.width, neighbor)]
            if TerrainKind(terrain_value) is TerrainKind.WATER:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


if __name__ == "__main__":
    unittest.main()
