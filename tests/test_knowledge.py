"""Tests for knowledge packet serialisation and confidence degradation (§31)."""

from __future__ import annotations

import msgpack
import tempfile
import unittest
from pathlib import Path

from village_sim.goap.knowledge import (
    ActionKnowledgePacket,
    WorldFactPacket,
    imported_confidence,
    load_packets,
    save_packets,
)


class TestImportedConfidence(unittest.TestCase):
    def test_full_trust_and_quality(self) -> None:
        result = imported_confidence(
            source_action_confidence=0.9,
            trust_in_source=1.0,
            transfer_quality=1.0,
        )
        self.assertAlmostEqual(result, 0.9, places=4)

    def test_partial_trust(self) -> None:
        result = imported_confidence(
            source_action_confidence=0.9,
            trust_in_source=0.8,
            transfer_quality=1.0,
        )
        self.assertAlmostEqual(result, 0.72, places=4)

    def test_reduced_transfer_quality(self) -> None:
        result = imported_confidence(
            source_action_confidence=0.9,
            trust_in_source=1.0,
            transfer_quality=0.5,
        )
        self.assertAlmostEqual(result, 0.45, places=4)

    def test_combined_degradation(self) -> None:
        result = imported_confidence(
            source_action_confidence=0.8,
            trust_in_source=0.75,
            transfer_quality=0.5,
        )
        self.assertAlmostEqual(result, 0.3, places=4)

    def test_zero_trust_gives_zero(self) -> None:
        result = imported_confidence(
            source_action_confidence=0.9,
            trust_in_source=0.0,
            transfer_quality=1.0,
        )
        self.assertEqual(result, 0.0)

    def test_acceptance_confidence_example(self) -> None:
        result = imported_confidence(
            source_action_confidence=0.9,
            trust_in_source=0.7,
            transfer_quality=0.8,
        )
        self.assertEqual(result, 0.504)


class TestPacketSerialisationRoundTrip(unittest.TestCase):
    def test_world_fact_packet_round_trip(self) -> None:
        packet = WorldFactPacket(
            knowledge_type="world_fact",
            fact_type="resource_location",
            source_agent_id="pioneer_001",
            confidence=0.95,
            data={
                "resource_id": "spring_001",
                "resource_type": "freshwater_spring",
                "coordinates": {"x": 10, "y": 12},
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "packets.msgpack"
            save_packets([packet], path)
            loaded = load_packets(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["knowledge_type"], "world_fact")
        self.assertEqual(loaded[0]["confidence"], 0.95)
        loaded_data = loaded[0]["data"]
        self.assertIsInstance(loaded_data, dict)
        assert isinstance(loaded_data, dict)
        self.assertEqual(loaded_data["resource_id"], "spring_001")

    def test_action_knowledge_packet_round_trip(self) -> None:
        packet = ActionKnowledgePacket(
            knowledge_type="action_model",
            source_agent_id="pioneer_001",
            confidence=0.8,
            action_id="drink_at_spring",
            policy_id="rl_spring_v1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "packets.msgpack"
            save_packets([packet], path)
            loaded = load_packets(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["knowledge_type"], "action_model")
        self.assertEqual(loaded[0]["action_id"], "drink_at_spring")

    def test_mixed_packets_round_trip(self) -> None:
        world_packet = WorldFactPacket(
            knowledge_type="world_fact",
            fact_type="resource_location",
            source_agent_id="pioneer_001",
            confidence=0.9,
            data={"resource_id": "bush_001"},
        )
        action_packet = ActionKnowledgePacket(
            knowledge_type="action_model",
            source_agent_id="pioneer_001",
            confidence=0.7,
            action_id="eat_berries",
            policy_id="rl_bush_v1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "packets.msgpack"
            save_packets([world_packet, action_packet], path)
            loaded = load_packets(path)

        self.assertEqual(len(loaded), 2)
        types: list[object] = [packet["knowledge_type"] for packet in loaded]
        self.assertIn("world_fact", types)
        self.assertIn("action_model", types)

    def test_msgpack_file_is_valid_msgpack(self) -> None:
        packet = WorldFactPacket(
            knowledge_type="world_fact",
            fact_type="resource_location",
            source_agent_id="pioneer_001",
            confidence=0.85,
            data={"resource_id": "spring_001"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "packets.msgpack"
            save_packets([packet], path)
            raw = path.read_bytes()
            parsed = msgpack.unpackb(raw, raw=False)

        self.assertIsInstance(parsed, list)


if __name__ == "__main__":
    unittest.main()
