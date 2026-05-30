"""Knowledge transfer packets and confidence degradation (§20, §21)."""

from __future__ import annotations

import msgpack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from village_sim.msgpack_codec import pack_default, unpack_object_hook
from village_sim.orchestrator.symbolic import FactValue

MsgpackValue = FactValue | None | list["MsgpackValue"] | dict[str, "MsgpackValue"]


ACTION_AGENT_ID = "agent_id"
ACTION_SOURCE_AGENT_ID = "source_agent_id"
ACTION_CONFIDENCE = "confidence"
ACTION_ID = "action_id"
ACTION_POLICY_ID = "policy_id"
ACTION_KNOWLEDGE_TYPE = "knowledge_type"
ACTION_MODEL_TYPE = "action_model"
ACTION_TRANSFER_THRESHOLD = 0.2
ACTION_TRANSFER_QUALITY = 1.0
ACTION_SOURCE_TRUST = 0.85
ACTION_KNOWLEDGE_SCHEMA: pl.Schema = pl.Schema(
    {
        ACTION_AGENT_ID: pl.Int64,
        ACTION_SOURCE_AGENT_ID: pl.String,
        ACTION_CONFIDENCE: pl.Float32,
        ACTION_ID: pl.String,
        ACTION_POLICY_ID: pl.String,
    }
)


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


def action_packets_to_frame(
    packets: list[dict[str, MsgpackValue]],
    owner_agent_id: int,
) -> pl.DataFrame:
    """Load serialized ActionKnowledgePacket dictionaries into Polars memory."""

    rows: list[dict[str, int | str | float]] = []
    for packet in packets:
        if packet.get(ACTION_KNOWLEDGE_TYPE) != ACTION_MODEL_TYPE:
            continue
        action_id: MsgpackValue | None = packet.get(ACTION_ID)
        policy_id: MsgpackValue | None = packet.get(ACTION_POLICY_ID)
        source_agent_id: MsgpackValue | None = packet.get(ACTION_SOURCE_AGENT_ID)
        confidence: MsgpackValue | None = packet.get(ACTION_CONFIDENCE)
        if not isinstance(action_id, str):
            raise ValueError("action knowledge packets require string action_id")
        if not isinstance(policy_id, str):
            raise ValueError("action knowledge packets require string policy_id")
        if not isinstance(source_agent_id, str):
            raise ValueError("action knowledge packets require string source_agent_id")
        if not isinstance(confidence, int | float):
            raise ValueError("action knowledge packets require numeric confidence")
        rows.append(
            {
                ACTION_AGENT_ID: owner_agent_id,
                ACTION_SOURCE_AGENT_ID: source_agent_id,
                ACTION_CONFIDENCE: float(confidence),
                ACTION_ID: action_id,
                ACTION_POLICY_ID: policy_id,
            }
        )
    return pl.DataFrame(rows, schema=ACTION_KNOWLEDGE_SCHEMA, orient="row")


def load_action_knowledge_frame(path: Path, owner_agent_id: int) -> pl.DataFrame:
    """Load ActionKnowledgePacket rows from a MessagePack packet file."""

    if not path.exists():
        raise FileNotFoundError(path)
    return action_packets_to_frame(load_packets(path), owner_agent_id)


def seed_agent_action_knowledge(
    frame: pl.DataFrame,
    founder_agent_id: int,
    new_agent_indices: NDArray[np.int64],
    perturbations: NDArray[np.float32],
) -> pl.DataFrame:
    """Clone Founder action knowledge into spawned settler rows."""

    if new_agent_indices.shape != perturbations.shape:
        raise ValueError("settler indices and perturbations must share one shape")
    founder_rows: pl.DataFrame = frame.filter(
        pl.col(ACTION_AGENT_ID) == founder_agent_id
    )
    if founder_rows.is_empty() or new_agent_indices.size == 0:
        return frame

    settler_frame: pl.DataFrame = pl.DataFrame(
        {
            ACTION_AGENT_ID: new_agent_indices.astype(np.int64, copy=False),
            "confidence_perturbation": perturbations.astype(np.float32, copy=False),
        }
    )
    spawned_rows: pl.DataFrame = settler_frame.join(founder_rows, how="cross").select(
        pl.col(ACTION_AGENT_ID),
        pl.lit(str(founder_agent_id)).alias(ACTION_SOURCE_AGENT_ID),
        (pl.col(ACTION_CONFIDENCE) + pl.col("confidence_perturbation"))
        .clip(0.0, 1.0)
        .cast(pl.Float32)
        .alias(ACTION_CONFIDENCE),
        pl.col(ACTION_ID),
        pl.col(ACTION_POLICY_ID),
    )
    return pl.concat([frame, spawned_rows], how="vertical")


def transfer_action_knowledge(
    frame: pl.DataFrame,
    teacher_idx: NDArray[np.int64],
    learner_idx: NDArray[np.int64],
    *,
    trust_in_source: float = ACTION_SOURCE_TRUST,
    transfer_quality: float = ACTION_TRANSFER_QUALITY,
    confidence_margin: float = ACTION_TRANSFER_THRESHOLD,
) -> pl.DataFrame:
    """Vectorized action-confidence transfer for proximity-matched pairs."""

    if teacher_idx.shape != learner_idx.shape:
        raise ValueError("teacher and learner arrays must share one shape")
    if teacher_idx.size == 0 or frame.is_empty():
        return frame

    pair_frame: pl.DataFrame = pl.DataFrame(
        {
            "teacher_agent_id": teacher_idx.astype(np.int64, copy=False),
            "learner_agent_id": learner_idx.astype(np.int64, copy=False),
        }
    ).filter(pl.col("teacher_agent_id") != pl.col("learner_agent_id"))
    if pair_frame.is_empty():
        return frame

    teacher_rows: pl.DataFrame = frame.rename(
        {
            ACTION_AGENT_ID: "teacher_agent_id",
            ACTION_CONFIDENCE: "teacher_confidence",
            ACTION_SOURCE_AGENT_ID: "teacher_source_agent_id",
        }
    )
    learner_rows: pl.DataFrame = frame.rename(
        {
            ACTION_AGENT_ID: "learner_agent_id",
            ACTION_CONFIDENCE: "learner_confidence",
            ACTION_SOURCE_AGENT_ID: "learner_source_agent_id",
        }
    )
    candidates: pl.DataFrame = (
        pair_frame.join(teacher_rows, on="teacher_agent_id", how="inner")
        .join(
            learner_rows,
            on=["learner_agent_id", ACTION_ID, ACTION_POLICY_ID],
            how="left",
        )
        .with_columns(
            pl.col("learner_confidence").fill_null(0.0),
            (
                pl.col("teacher_confidence")
                * pl.lit(transfer_quality)
                * pl.lit(trust_in_source)
            )
            .round(4)
            .clip(0.0, 1.0)
            .cast(pl.Float32)
            .alias("imported_confidence"),
        )
        .filter(
            pl.col("teacher_confidence")
            > pl.col("learner_confidence") + pl.lit(confidence_margin)
        )
        .select(
            pl.col("learner_agent_id").alias(ACTION_AGENT_ID),
            pl.col("teacher_agent_id").cast(pl.String).alias(ACTION_SOURCE_AGENT_ID),
            pl.col("imported_confidence").alias(ACTION_CONFIDENCE),
            pl.col(ACTION_ID),
            pl.col(ACTION_POLICY_ID),
        )
        .group_by([ACTION_AGENT_ID, ACTION_ID, ACTION_POLICY_ID])
        .agg(
            pl.col(ACTION_SOURCE_AGENT_ID).first(),
            pl.col(ACTION_CONFIDENCE).max(),
        )
        .select(
            ACTION_AGENT_ID,
            ACTION_SOURCE_AGENT_ID,
            ACTION_CONFIDENCE,
            ACTION_ID,
            ACTION_POLICY_ID,
        )
    )
    if candidates.is_empty():
        return frame

    keys: list[str] = [ACTION_AGENT_ID, ACTION_ID, ACTION_POLICY_ID]
    retained: pl.DataFrame = frame.join(candidates.select(keys), on=keys, how="anti")
    return pl.concat([retained, candidates], how="vertical_relaxed").sort(keys)
