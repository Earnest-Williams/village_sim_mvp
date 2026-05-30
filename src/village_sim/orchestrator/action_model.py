"""Synthesized GOAP action model and action library (§12, §13, §17, §19)."""

from __future__ import annotations

import msgpack
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from village_sim.msgpack_codec import pack_default, unpack_object_hook
from village_sim.orchestrator.induction import EffectEstimate
from village_sim.orchestrator.symbolic import FactValue

# ── Supporting types ──────────────────────────────────────────────────────────


class ActionScope(StrEnum):
    INSTANCE = "instance"
    TEMPLATE = "template"


class ActionLifecycle(StrEnum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    TRUSTED = "trusted"
    DEPRECATED = "deprecated"


class ExecutorType(StrEnum):
    RL_POLICY = "RL_POLICY"
    PATHFINDER = "PATHFINDER"
    COMPOSITE = "COMPOSITE"
    SCRIPTED_PRIMITIVE = "SCRIPTED_PRIMITIVE"


@dataclass(slots=True)
class TargetBinding:
    mode: str  # "resource_id" | "current_target"
    resource_id: str | None = None  # used when mode == "resource_id"
    required_type: str | None = None  # used when mode == "current_target"


@dataclass(slots=True)
class ExecutionPayload:
    type: ExecutorType
    policy_id: str
    policy_version: int
    target_binding: TargetBinding


@dataclass(slots=True)
class CostModel:
    base_ticks: float
    distance_weight: float = 0.0
    fatigue_weight: float = 1.0
    night_multiplier: float = 1.0


@dataclass(slots=True)
class ActionConfidence:
    trials: int
    successful_trials: int
    failed_trials: int
    success_rate: float
    death_rate: float
    timeout_rate: float
    death_trials: int = 0
    timeout_trials: int = 0


# ── Main synthesized action (§12) ─────────────────────────────────────────────


@dataclass
class SynthesizedAction:
    schema_version: int
    action_id: str
    display_name: str
    scope: ActionScope
    lifecycle: ActionLifecycle

    preconditions: dict[str, FactValue]
    soft_preconditions: dict[str, float]

    effects: dict[str, EffectEstimate]  # primary effects
    side_effects: dict[str, EffectEstimate]  # secondary effects

    cost_model: CostModel
    confidence: ActionConfidence
    execution_payload: ExecutionPayload
    symbolic_effects: dict[str, FactValue] = field(
        default_factory=lambda: dict[str, FactValue]()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_msgpack(self) -> bytes:
        packed = msgpack.packb(self, default=pack_default, use_bin_type=True)
        if not isinstance(packed, bytes):
            raise TypeError("MessagePack packb did not return bytes")
        return packed

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SynthesizedAction:
        """Deserialise from a plain MessagePack-compatible dict."""
        effects = _effect_estimates_from_raw(data.get("effects", {}))
        side_effects = _effect_estimates_from_raw(data.get("side_effects", {}))
        cost_model = _cost_model_from_raw(data["cost_model"])
        confidence = _confidence_from_raw(data["confidence"])
        execution_payload = _execution_payload_from_raw(data["execution_payload"])
        return SynthesizedAction(
            schema_version=data["schema_version"],
            action_id=data["action_id"],
            display_name=data["display_name"],
            scope=ActionScope(data["scope"]),
            lifecycle=ActionLifecycle(data["lifecycle"]),
            preconditions=data["preconditions"],
            soft_preconditions=data.get("soft_preconditions", {}),
            effects=effects,
            side_effects=side_effects,
            cost_model=cost_model,
            confidence=confidence,
            execution_payload=execution_payload,
            symbolic_effects=dict(data.get("symbolic_effects", {})),
        )


def _effect_estimates_from_raw(raw: object) -> dict[str, EffectEstimate]:
    if not isinstance(raw, Mapping):
        raise ValueError("effect estimates must be stored as a MessagePack map")
    estimates: dict[str, EffectEstimate] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("effect estimate keys must be strings")
        if isinstance(value, EffectEstimate):
            estimates[key] = value
        elif isinstance(value, Mapping):
            estimates[key] = EffectEstimate(**dict(value))
        else:
            raise ValueError("effect estimate values must be MessagePack maps")
    return estimates


def _cost_model_from_raw(raw: object) -> CostModel:
    if isinstance(raw, CostModel):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("cost_model must be a MessagePack map")
    return CostModel(**dict(raw))


def _confidence_from_raw(raw: object) -> ActionConfidence:
    if isinstance(raw, ActionConfidence):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("confidence must be a MessagePack map")
    return ActionConfidence(**dict(raw))


def _target_binding_from_raw(raw: object) -> TargetBinding:
    if isinstance(raw, TargetBinding):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("target_binding must be a MessagePack map")
    return TargetBinding(**dict(raw))


def _execution_payload_from_raw(raw: object) -> ExecutionPayload:
    if isinstance(raw, ExecutionPayload):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("execution_payload must be a MessagePack map")
    data = dict(raw)
    return ExecutionPayload(
        type=ExecutorType(data["type"]),
        policy_id=data["policy_id"],
        policy_version=data["policy_version"],
        target_binding=_target_binding_from_raw(data["target_binding"]),
    )


# ── Promotion logic (§17) ─────────────────────────────────────────────────────


def is_promotable(confidence: ActionConfidence) -> bool:
    """True when an action has accumulated enough evidence to become trusted."""
    return (
        confidence.trials >= 100
        and confidence.success_rate >= 0.85
        and confidence.death_rate <= 0.02
        and confidence.timeout_rate <= 0.10
    )


def promote_action(action: SynthesizedAction) -> None:
    """Advance lifecycle: candidate → validated → trusted."""
    if action.lifecycle is ActionLifecycle.CANDIDATE:
        action.lifecycle = ActionLifecycle.VALIDATED
    elif action.lifecycle is ActionLifecycle.VALIDATED:
        if is_promotable(action.confidence):
            action.lifecycle = ActionLifecycle.TRUSTED


# ── Action library (§33 step 11) ─────────────────────────────────────────────


class ActionLibrary:
    """In-memory store for synthesized actions, with MessagePack persistence."""

    def __init__(self) -> None:
        self._actions: dict[str, SynthesizedAction] = {}

    def add(self, action: SynthesizedAction) -> None:
        self._actions[action.action_id] = action

    def get(self, action_id: str) -> SynthesizedAction | None:
        return self._actions.get(action_id)

    def all_actions(self) -> list[SynthesizedAction]:
        return list(self._actions.values())

    def trusted_actions(self) -> list[SynthesizedAction]:
        return [
            a for a in self._actions.values() if a.lifecycle is ActionLifecycle.TRUSTED
        ]

    def actions_for_lifecycle(
        self, lifecycle: ActionLifecycle
    ) -> list[SynthesizedAction]:
        return [a for a in self._actions.values() if a.lifecycle is lifecycle]

    def save(self, path: Path) -> None:
        """Serialise all actions to a MessagePack file. Generates data, not code (§34)."""
        data: list[SynthesizedAction] = list(self._actions.values())
        with path.open("wb") as action_file:
            msgpack.pack(data, action_file, default=pack_default, use_bin_type=True)

    @classmethod
    def load(cls, path: Path) -> ActionLibrary:
        library = cls()
        with path.open("rb") as action_file:
            raw: object = msgpack.unpack(
                action_file, raw=False, object_hook=unpack_object_hook
            )
        if not isinstance(raw, list):
            raise ValueError("action library file must contain a MessagePack list")
        raw_items = cast(list[object], raw)
        for item in raw_items:
            if isinstance(item, SynthesizedAction):
                library.add(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError("each action library entry must be a MessagePack map")
            item_mapping = cast(Mapping[object, object], item)
            action_data: dict[str, Any] = {}
            for key, value in item_mapping.items():
                if not isinstance(key, str):
                    raise ValueError("action library entry keys must be strings")
                action_data[key] = value
            library.add(SynthesizedAction.from_dict(action_data))
        return library


# ── Confidence update after execution (§19, §21) ─────────────────────────────


def update_confidence_after_execution(
    action: SynthesizedAction,
    success: bool,
    death: bool,
    timeout: bool,
) -> None:
    """Incrementally update a SynthesizedAction's confidence stats.

    Called each time a Townsfolk agent executes the action and reports the
    outcome.  Automatically promotes the action if the updated stats meet the
    threshold.
    """
    c = action.confidence
    c.trials += 1
    if success:
        c.successful_trials += 1
    else:
        c.failed_trials += 1

    c.success_rate = round(c.successful_trials / c.trials, 4)

    if death:
        c.death_trials += 1
    if timeout:
        c.timeout_trials += 1

    c.death_rate = round(c.death_trials / c.trials, 4)
    c.timeout_rate = round(c.timeout_trials / c.trials, 4)

    promote_action(action)  # re-evaluate lifecycle after each update
