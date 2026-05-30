"""MessagePack serialization hooks for Village Sim state objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import StrEnum
from typing import Any

TYPE_KEY = "__village_sim_type__"


def pack_default(value: object) -> object:
    """Convert custom simulation objects into MessagePack-compatible values."""
    from village_sim.goap.knowledge import ActionKnowledgePacket, WorldFactPacket
    from village_sim.orchestrator.action_model import (
        ActionConfidence,
        CostModel,
        ExecutionPayload,
        SynthesizedAction,
        TargetBinding,
    )
    from village_sim.orchestrator.induction import EffectEstimate
    from village_sim.orchestrator.trajectory import NeedState, StateSnapshot
    from village_sim.sim.events import TickEvent
    from village_sim.sim.metrics import LearningStats, SimResult
    from village_sim.sim.profile import SimulationTimings, TimingBucket
    from village_sim.sim.snapshot import AgentSnapshot, WorldSnapshot

    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, AgentSnapshot):
        return {TYPE_KEY: "AgentSnapshot", **_public_slots(value)}
    if isinstance(value, WorldSnapshot):
        return {TYPE_KEY: "WorldSnapshot", **_public_slots(value)}
    if isinstance(value, TickEvent):
        return {TYPE_KEY: "TickEvent", **_public_slots(value)}
    if isinstance(value, LearningStats):
        return {TYPE_KEY: "LearningStats", **_public_slots(value)}
    if isinstance(value, SimResult):
        return {TYPE_KEY: "SimResult", **_public_slots(value)}
    if isinstance(value, SimulationTimings):
        return {TYPE_KEY: "SimulationTimings", **_public_slots(value)}
    if isinstance(value, TimingBucket):
        return {TYPE_KEY: "TimingBucket", **_public_slots(value)}
    if isinstance(value, NeedState):
        return {TYPE_KEY: "NeedState", **_public_slots(value)}
    if isinstance(value, StateSnapshot):
        return {TYPE_KEY: "StateSnapshot", **_public_slots(value)}
    if isinstance(value, TargetBinding):
        return {TYPE_KEY: "TargetBinding", **_public_slots(value)}
    if isinstance(value, ExecutionPayload):
        return {TYPE_KEY: "ExecutionPayload", **_public_slots(value)}
    if isinstance(value, CostModel):
        return {TYPE_KEY: "CostModel", **_public_slots(value)}
    if isinstance(value, ActionConfidence):
        return {TYPE_KEY: "ActionConfidence", **_public_slots(value)}
    if isinstance(value, SynthesizedAction):
        return {TYPE_KEY: "SynthesizedAction", **_public_slots(value)}
    if isinstance(value, EffectEstimate):
        return {TYPE_KEY: "EffectEstimate", **_public_slots(value)}
    if isinstance(value, WorldFactPacket):
        return {TYPE_KEY: "WorldFactPacket", **_public_slots(value)}
    if isinstance(value, ActionKnowledgePacket):
        return {TYPE_KEY: "ActionKnowledgePacket", **_public_slots(value)}
    raise TypeError(
        f"Object of type {type(value).__name__} is not MessagePack serializable"
    )


def unpack_object_hook(value: dict[str, Any]) -> object:
    """Rebuild custom simulation objects from MessagePack map values."""
    type_name = value.get(TYPE_KEY)
    if not isinstance(type_name, str):
        return value

    data = dict(value)
    del data[TYPE_KEY]

    from village_sim.goap.knowledge import ActionKnowledgePacket, WorldFactPacket
    from village_sim.orchestrator.action_model import (
        ActionConfidence,
        CostModel,
        ExecutionPayload,
        SynthesizedAction,
        TargetBinding,
    )
    from village_sim.orchestrator.induction import EffectEstimate
    from village_sim.orchestrator.trajectory import NeedState, StateSnapshot
    from village_sim.sim.events import TickEvent
    from village_sim.sim.metrics import LearningStats, SimResult
    from village_sim.sim.profile import SimulationTimings, TimingBucket
    from village_sim.sim.snapshot import AgentSnapshot, WorldSnapshot

    if type_name == "AgentSnapshot":
        return AgentSnapshot(**data)
    if type_name == "WorldSnapshot":
        return WorldSnapshot(**data)
    if type_name == "TickEvent":
        return TickEvent(**data)
    if type_name == "LearningStats":
        return LearningStats(**data)
    if type_name == "SimResult":
        return SimResult(**data)
    if type_name == "SimulationTimings":
        return SimulationTimings(**data)
    if type_name == "TimingBucket":
        return TimingBucket(**data)
    if type_name == "NeedState":
        return NeedState(**data)
    if type_name == "StateSnapshot":
        return StateSnapshot(**data)
    if type_name == "TargetBinding":
        return TargetBinding(**data)
    if type_name == "ExecutionPayload":
        return ExecutionPayload(**data)
    if type_name == "CostModel":
        return CostModel(**data)
    if type_name == "ActionConfidence":
        return ActionConfidence(**data)
    if type_name == "SynthesizedAction":
        return SynthesizedAction.from_dict(data)
    if type_name == "EffectEstimate":
        return EffectEstimate(**data)
    if type_name == "WorldFactPacket":
        return WorldFactPacket(**data)
    if type_name == "ActionKnowledgePacket":
        return ActionKnowledgePacket(**data)
    raise ValueError(f"Unsupported Village Sim MessagePack type: {type_name}")


def _public_slots(value: object) -> dict[str, Any]:
    if is_dataclass(value):
        return {
            field.name: getattr(value, field.name)
            for field in fields(value)
            if not field.name.startswith("_")
        }
    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, Mapping):
        raise TypeError(
            f"Object of type {type(value).__name__} has no serializable fields"
        )
    return {
        str(name): field_value
        for name, field_value in attributes.items()
        if not str(name).startswith("_")
    }
