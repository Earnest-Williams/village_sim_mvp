from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from village_sim.agent.actions import BUILD, DIG, PLANT
from village_sim.goap.knowledge import load_founder_knowledge, save_founder_knowledge
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
