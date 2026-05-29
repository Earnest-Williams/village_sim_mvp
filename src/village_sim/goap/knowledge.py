"""Knowledge transfer packets and confidence degradation (§20, §21)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# ── Packet types (§20) ────────────────────────────────────────────────────────


@dataclass
class WorldFactPacket:
    knowledge_type: str  # always "world_fact"
    fact_type: str  # e.g. "resource_location"
    source_agent_id: str
    confidence: float
    data: dict  # resource_id, resource_type, coordinates

    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[arg-type]


@dataclass
class ActionKnowledgePacket:
    knowledge_type: str  # always "action_model"
    source_agent_id: str
    confidence: float
    action_id: str
    policy_id: str

    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[arg-type]


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
    """Serialise knowledge packets to a JSON file (generates data, not code §34)."""
    data = [p.to_dict() for p in packets]
    path.write_text(json.dumps(data, indent=2))


def load_packets(path: Path) -> list[dict]:
    """Load knowledge packets from a JSON file."""
    return json.loads(path.read_text())
