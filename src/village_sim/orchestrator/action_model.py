"""Synthesized GOAP action model and action library (§12, §13, §17, §19)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not isinstance(data, dict):
            raise TypeError("synthesized action serialization must produce a dict")
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SynthesizedAction:
        """Deserialise from a plain dict (e.g. loaded from JSON)."""
        return SynthesizedAction(
            schema_version=data["schema_version"],
            action_id=data["action_id"],
            display_name=data["display_name"],
            scope=ActionScope(data["scope"]),
            lifecycle=ActionLifecycle(data["lifecycle"]),
            preconditions=data["preconditions"],
            soft_preconditions=data.get("soft_preconditions", {}),
            effects={
                k: EffectEstimate(**v) for k, v in data.get("effects", {}).items()
            },
            side_effects={
                k: EffectEstimate(**v) for k, v in data.get("side_effects", {}).items()
            },
            cost_model=CostModel(**data["cost_model"]),
            confidence=ActionConfidence(**data["confidence"]),
            execution_payload=ExecutionPayload(
                type=ExecutorType(data["execution_payload"]["type"]),
                policy_id=data["execution_payload"]["policy_id"],
                policy_version=data["execution_payload"]["policy_version"],
                target_binding=TargetBinding(
                    **data["execution_payload"]["target_binding"]
                ),
            ),
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
    """In-memory store for synthesized actions, with JSON persistence."""

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
        """Serialise all actions to a JSON file. Generates data, not code (§34)."""
        data: list[dict[str, Any]] = [
            action.to_dict() for action in self._actions.values()
        ]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ActionLibrary:
        library = cls()
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            raise ValueError("action library file must contain a JSON list")
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("each action library entry must be a JSON object")
            library.add(SynthesizedAction.from_dict(item))
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
