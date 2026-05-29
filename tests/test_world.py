from __future__ import annotations

import random
import unittest

from village_sim.core.config import SimConfig
from village_sim.core.types import Position, TerrainKind
from village_sim.world.world import generate_world


class WorldTests(unittest.TestCase):
    def test_world_generation_sizes_arrays(self) -> None:
        config = SimConfig(width=24, height=16, seed=42)
        world = generate_world(config, random.Random(config.seed))

        self.assertEqual(len(world.height_map), 24 * 16)
        self.assertEqual(len(world.terrain), 24 * 16)
        self.assertEqual(len(world.water), 24 * 16)
        self.assertEqual(len(world.food), 24 * 16)
        self.assertEqual(len(world.food_capacity), 24 * 16)

    def test_water_terrain_has_persistent_water(self) -> None:
        config = SimConfig(width=24, height=16, seed=7)
        world = generate_world(config, random.Random(config.seed))

        water_indices = [
            index
            for index, terrain_value in enumerate(world.terrain)
            if TerrainKind(terrain_value) is TerrainKind.WATER
        ]

        self.assertTrue(len(water_indices) > 0)
        self.assertTrue(all(world.water[index] >= 0.65 for index in water_indices))

    def test_consume_food_reduces_food(self) -> None:
        config = SimConfig(width=12, height=12, seed=9)
        world = generate_world(config, random.Random(config.seed))
        position = Position(2, 2)
        index = world.index(position)
        world.food[index] = 0.5

        consumed = world.consume_food(position, 0.2)

        self.assertAlmostEqual(consumed, 0.2)
        self.assertAlmostEqual(world.food[index], 0.3)


if __name__ == "__main__":
    unittest.main()
