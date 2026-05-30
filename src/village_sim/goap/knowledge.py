"""Knowledge transfer packets and confidence degradation (§20, §21)."""

from __future__ import annotations

import msgpack
from dataclasses import dataclass
from pathlib import Path

from village_sim.msgpack_codec import pack_default, unpack_object_hook
from village_sim.orchestrator.symbolic import FactValue

MsgpackValue = FactValue | None | list["MsgpackValue"] | dict[str, "MsgpackValue"]


# ── Packet types (§20) ────────────────────────────────────────────────────────


@dataclass
class WorldFactPacket:
    knowledge_type: str  # always "world_fact"
    fact_type: str  # e.g. "resource_location"
    source_agent_id: str
    confidence: float
    data: dict[str, MsgpackValue]  # resource_id, resource_type, coordinates

    def to_dict(self) -> dict[str, MsgpackValue]:
        return {
            "knowledge_type": self.knowledge_type,
            "fact_type": self.fact_type,
            "source_agent_id": self.source_agent_id,
            "confidence": self.confidence,
            "data": self.data,
        }


@dataclass
class ActionKnowledgePacket:
    knowledge_type: str  # always "action_model"
    source_agent_id: str
    confidence: float
    action_id: str
    policy_id: str

    def to_dict(self) -> dict[str, MsgpackValue]:
        return {
            "knowledge_type": self.knowledge_type,
            "source_agent_id": self.source_agent_id,
            "confidence": self.confidence,
            "action_id": self.action_id,
            "policy_id": self.policy_id,
        }


KnowledgePacket = WorldFactPacket | ActionKnowledgePacket


# ── Confidence degradation on import (§21) ────────────────────────────────────


def imported_confidence(
    source_action_confidence: float,
    trust_in_source: float,
    transfer_quality: float = 1.0,
) -> float:
    """Degrade imported confidence by source trust and transfer quality.

    imported = source_confidence * transfer_quality * trust_in_source
    """
    return round(
        source_action_confidence * transfer_quality * trust_in_source,
        4,
    )


# ── Serialisation helpers ─────────────────────────────────────────────────────


def save_packets(packets: list[KnowledgePacket], path: Path) -> None:
    """Serialise knowledge packets to a MessagePack file (generates data, not code §34)."""
    data: list[KnowledgePacket] = packets
    with path.open("wb") as packet_file:
        msgpack.pack(data, packet_file, default=pack_default, use_bin_type=True)


def load_packets(path: Path) -> list[dict[str, MsgpackValue]]:
    """Load knowledge packets from a MessagePack file."""
    with path.open("rb") as packet_file:
        raw: object = msgpack.unpack(
            packet_file, raw=False, object_hook=unpack_object_hook
        )
    if not isinstance(raw, list):
        raise ValueError("knowledge packet file must contain a MessagePack list")

    packets: list[dict[str, MsgpackValue]] = []
    for item in raw:
        if isinstance(item, WorldFactPacket | ActionKnowledgePacket):
            item = item.to_dict()
        if not isinstance(item, dict):
            raise ValueError("each knowledge packet must be a MessagePack object")
        packet: dict[str, MsgpackValue] = {}
        for key, value in item.items():
            if not isinstance(key, str):
                raise ValueError("knowledge packet keys must be strings")
            if not _is_msgpack_value(value):
                raise ValueError(f"unsupported MessagePack value for key {key!r}")
            packet[key] = value
        packets.append(packet)
    return packets


def _is_msgpack_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_msgpack_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_msgpack_value(item)
            for key, item in value.items()
        )
    return False
