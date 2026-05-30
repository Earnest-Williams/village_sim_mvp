"""Knowledge transfer packets and confidence degradation (§20, §21)."""

from __future__ import annotations

import msgpack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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


FounderKnowledgePayload = dict[str, NDArray[np.float32]]


def save_founder_knowledge(
    payload: FounderKnowledgePayload,
    path: Path,
) -> None:
    """Serialize finalized Founder policy/heuristic arrays to MessagePack."""

    packed_payload: dict[str, dict[str, object]] = {}
    for name, array in payload.items():
        if not isinstance(name, str):
            raise ValueError("Founder knowledge keys must be strings")
        normalized: NDArray[np.float32] = np.asarray(array, dtype=np.float32)
        packed_payload[name] = {
            "dtype": "float32",
            "shape": tuple(int(axis) for axis in normalized.shape),
            "data": normalized.tobytes(order="C"),
        }
    with path.open("wb") as knowledge_file:
        msgpack.pack(packed_payload, knowledge_file, use_bin_type=True)


def load_founder_knowledge(path: Path) -> FounderKnowledgePayload:
    """Load Founder policy/heuristic arrays from a MessagePack file."""

    with path.open("rb") as knowledge_file:
        raw: object = msgpack.unpack(knowledge_file, raw=False)
    if not isinstance(raw, dict):
        raise ValueError("Founder knowledge file must contain a MessagePack object")

    payload: FounderKnowledgePayload = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("Founder knowledge keys must be strings")
        if not isinstance(value, dict):
            raise ValueError("Founder knowledge entries must be MessagePack objects")
        dtype_value: object = value.get("dtype")
        shape_value: object = value.get("shape")
        data_value: object = value.get("data")
        if dtype_value != "float32":
            raise ValueError("Founder knowledge arrays must use float32 dtype")
        if not isinstance(shape_value, list):
            raise ValueError("Founder knowledge shape must be a MessagePack array")
        if not isinstance(data_value, bytes):
            raise ValueError("Founder knowledge data must be bytes")
        shape: tuple[int, ...] = tuple(
            _validate_shape_axis(axis) for axis in shape_value
        )
        array: NDArray[np.float32] = np.frombuffer(data_value, dtype=np.float32).copy()
        expected_size: int = int(np.prod(shape, dtype=np.int64))
        if array.size != expected_size:
            raise ValueError("Founder knowledge data size does not match shape")
        payload[key] = array.reshape(shape)
    return payload


def _validate_shape_axis(axis: object) -> int:
    if not isinstance(axis, int):
        raise ValueError("Founder knowledge shape axes must be integers")
    if axis < 0:
        raise ValueError("Founder knowledge shape axes must be non-negative")
    return axis
