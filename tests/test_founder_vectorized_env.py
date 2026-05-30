from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from village_sim.agent.actions import BUILD, DIG, PLANT
from village_sim.goap.knowledge import load_founder_knowledge, save_founder_knowledge
from village_sim.core.types import TerrainKind
from village_sim.sim.engine import FounderTrainingEnv
from village_sim.world.grid import WorldGrids


class FounderVectorizedEnvironmentTests(unittest.TestCase):
    def test_world_grids_use_flat_numpy_buffers(self) -> None:
        grids = WorldGrids(width=4, height=3)

        self.assertEqual(grids.terrain_kind.shape, (12,))
        self.assertEqual(grids.terrain_kind.dtype, np.int32)
        self.assertEqual(grids.structure_kind.dtype, np.int32)
        self.assertEqual(grids.elevation.dtype, np.float32)
        self.assertEqual(grids.structure_health.dtype, np.float32)
        self.assertEqual(grids.crop_growth.dtype, np.float32)
        self.assertEqual(grids.water_table.dtype, np.float32)

    def test_world_grids_copy_from_2d_arrays(self) -> None:
        grids = WorldGrids(width=4, height=3)

        terrain = np.arange(12, dtype=np.int32).reshape(3, 4)
        elevation = np.arange(12, dtype=np.float32).reshape(3, 4)
        water = (np.arange(12, dtype=np.float32) / np.float32(10.0)).reshape(3, 4)
        food = (np.arange(12, dtype=np.float32) / np.float32(20.0)).reshape(3, 4)

        grids.copy_from_world_arrays(
            terrain=terrain, elevation=elevation, water=water, food=food
        )

        np.testing.assert_array_equal(grids.terrain_kind, terrain.ravel())
        np.testing.assert_array_equal(grids.elevation, elevation.ravel())
        np.testing.assert_array_equal(grids.water_table, water.ravel())
        np.testing.assert_array_equal(grids.crop_growth, food.ravel())

    def test_founder_env_returns_gymnasium_step_tuple(self) -> None:
        env = FounderTrainingEnv(
            width=8,
            height=8,
            max_steps=4,
            seed=5,
            receptive_field=9,
        )
        obs, info = env.reset(seed=5)
        self.assertEqual(obs.shape, (6, 9, 9))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(info["tick"], 0)

        action_array = np.array(
            [[int(DIG), 2, 2], [int(PLANT), 3, 3], [int(BUILD), 4, 4]],
            dtype=np.int32,
        )
        next_obs, reward, terminated, truncated, step_info = env.step(action_array)

        self.assertEqual(next_obs.shape, (6, 9, 9))
        self.assertGreaterEqual(reward, 3.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["tick"], 1)

    def test_founder_env_only_grows_planted_cells(self) -> None:
        env = FounderTrainingEnv(
            width=4,
            height=4,
            max_steps=4,
            seed=5,
            receptive_field=3,
        )
        env.reset(seed=5)
        env.grids.terrain_kind.fill(np.int32(TerrainKind.GRASS))
        env.grids.crop_growth.fill(np.float32(0.0))

        env.step(np.array([[int(PLANT), 1, 1]], dtype=np.int32))

        planted_index = 1 * env.width + 1
        self.assertEqual(env.grids.crop_growth[0], np.float32(0.0))
        self.assertEqual(env.grids.crop_growth[planted_index], np.float32(0.26))

    def test_founder_env_accumulates_multiple_digs_on_same_cell(self) -> None:
        env = FounderTrainingEnv(
            width=4,
            height=4,
            max_steps=4,
            seed=5,
            receptive_field=3,
        )
        env.reset(seed=5)
        env.grids.terrain_kind.fill(np.int32(TerrainKind.GRASS))
        env.grids.water_table.fill(np.float32(0.1))

        env.step(np.array([[int(DIG), 2, 2], [int(DIG), 2, 2]], dtype=np.int32))

        dug_index = 2 * env.width + 2
        self.assertEqual(env.grids.water_table[dug_index], np.float32(0.46))

    def test_founder_knowledge_round_trips_float32_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "founder.msgpack"
            weights = np.arange(6, dtype=np.float32).reshape(2, 3)

            save_founder_knowledge({"weights": weights}, path)
            loaded = load_founder_knowledge(path)

        self.assertIn("weights", loaded)
        np.testing.assert_array_equal(loaded["weights"], weights)
        self.assertEqual(loaded["weights"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
