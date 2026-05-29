"""Orchestrator: observes trajectories and synthesizes GOAP actions (§4, §10–§17)."""

from __future__ import annotations

from village_sim.orchestrator.action_model import (
    ActionConfidence,
    ActionLifecycle,
    ActionScope,
    CostModel,
    ExecutionPayload,
    ExecutorType,
    SynthesizedAction,
    TargetBinding,
    is_promotable,
)
from village_sim.orchestrator.evaluator import (
    NeedName,
    TaskResult,
    cluster_key_for_trajectory,
    evaluate_cold_stress_task,
    evaluate_fatigue_task,
    evaluate_hunger_task,
    evaluate_thirst_task,
)
from village_sim.orchestrator.induction import (
    EffectEstimate,
    average_cost,
    infer_hard_preconditions,
    infer_need_effect,
    infer_soft_preconditions,
)
from village_sim.orchestrator.trajectory import Trajectory
from village_sim.orchestrator.travel import should_record_travel_segment

_EVALUATORS = {
    NeedName.HUNGER: evaluate_hunger_task,
    NeedName.THIRST: evaluate_thirst_task,
    NeedName.FATIGUE: evaluate_fatigue_task,
    NeedName.COLD_STRESS: evaluate_cold_stress_task,
}


class Orchestrator:
    """Consumes recorded trajectories and produces synthesized GOAP actions."""

    def __init__(self, schema_version: int = 1) -> None:
        self._schema_version = schema_version
        self._clusters: dict[str, list[Trajectory]] = {}  # key → successful
        self._failed: dict[str, list[Trajectory]] = {}  # key → failed

    def record(self, trajectory: Trajectory) -> None:
        """Classify and store one completed trajectory."""
        key = cluster_key_for_trajectory(trajectory)

        is_success = trajectory.task_name == "travel" and should_record_travel_segment(
            trajectory.start,
            trajectory.end,
        )
        if not is_success:
            for evaluator in _EVALUATORS.values():
                result: TaskResult = evaluator(trajectory)
                if result.success:
                    is_success = True
                    break

        if is_success:
            self._clusters.setdefault(key, []).append(trajectory)
        else:
            self._failed.setdefault(key, []).append(trajectory)

    def synthesize_all(self) -> list[SynthesizedAction]:
        """Attempt to synthesize instance and template actions per ready cluster."""
        actions: list[SynthesizedAction] = []
        for key, successful in self._clusters.items():
            if len(successful) < 10:
                continue  # insufficient evidence
            failed = self._failed.get(key, [])
            actions.extend(self._synthesize_cluster(key, successful, failed))
        return actions

    def _synthesize_cluster(
        self,
        cluster_key: str,
        successful: list[Trajectory],
        failed: list[Trajectory],
    ) -> list[SynthesizedAction]:
        parts = cluster_key.split(":")
        task_name = parts[0] if parts else "none"
        if task_name == "travel":
            return self._synthesize_travel_cluster(successful, failed)
        return self._synthesize_exploit_cluster(cluster_key, successful, failed)

    def _synthesize_exploit_cluster(
        self,
        cluster_key: str,
        successful: list[Trajectory],
        failed: list[Trajectory],
    ) -> list[SynthesizedAction]:
        parts = cluster_key.split(":")
        target_type = parts[1] if len(parts) > 1 else "none"

        # Representative trajectory for payload extraction.
        rep = successful[0]
        target_id = str(rep.start.symbolic.get("target_id", "unknown"))

        inferred_preconditions = infer_hard_preconditions(successful, failed)
        preconditions = _exploit_preconditions(inferred_preconditions, target_type)
        if not preconditions:
            return []  # cannot build a meaningful action without preconditions

        soft_preconditions = infer_soft_preconditions(successful, failed)

        effects: dict[str, EffectEstimate] = {}
        for need in NeedName:
            estimate = infer_need_effect(successful, need)
            if estimate is not None:
                effects[f"{need.value}_delta"] = estimate

        if not effects:
            return []

        confidence = _confidence_for(successful, failed)
        night_multiplier = 1.0 if target_type == "freshwater_spring" else 1.25
        cost_model = CostModel(
            base_ticks=round(average_cost(successful), 1),
            distance_weight=0.0,
            fatigue_weight=1.0,
            night_multiplier=night_multiplier,
        )

        actions: list[SynthesizedAction] = []
        if target_id not in ("none", "unknown"):
            actions.append(
                self._build_exploit_action(
                    target_type=target_type,
                    target_id=target_id,
                    scope=ActionScope.INSTANCE,
                    preconditions=preconditions,
                    soft_preconditions=soft_preconditions,
                    effects=effects,
                    cost_model=cost_model,
                    confidence=confidence,
                )
            )

        if target_type != "none":
            actions.append(
                self._build_exploit_action(
                    target_type=target_type,
                    target_id=target_id,
                    scope=ActionScope.TEMPLATE,
                    preconditions=preconditions,
                    soft_preconditions=soft_preconditions,
                    effects=effects,
                    cost_model=cost_model,
                    confidence=confidence,
                )
            )

        return actions

    def _synthesize_travel_cluster(
        self,
        successful: list[Trajectory],
        failed: list[Trajectory],
    ) -> list[SynthesizedAction]:
        rep = successful[0]
        target_type = str(rep.start.symbolic.get("target_type", "none"))
        target_id = str(rep.start.symbolic.get("target_id", "unknown"))
        if target_type == "none":
            return []

        confidence = _confidence_for(successful, failed)
        cost_model = CostModel(
            base_ticks=round(average_cost(successful), 1),
            distance_weight=round(_travel_distance_weight(successful), 4),
            fatigue_weight=1.0,
            night_multiplier=1.25,
        )
        preconditions: dict[str, bool | int | float | str] = {
            "has_target_location": True,
            "at_known_target": False,
            "target_type": target_type,
        }
        soft_preconditions = infer_soft_preconditions(successful, failed)

        actions: list[SynthesizedAction] = []
        if target_id not in ("none", "unknown"):
            actions.append(
                self._build_travel_action(
                    target_type=target_type,
                    target_id=target_id,
                    scope=ActionScope.INSTANCE,
                    preconditions=preconditions,
                    soft_preconditions=soft_preconditions,
                    cost_model=cost_model,
                    confidence=confidence,
                )
            )
        actions.append(
            self._build_travel_action(
                target_type=target_type,
                target_id=target_id,
                scope=ActionScope.TEMPLATE,
                preconditions=preconditions,
                soft_preconditions=soft_preconditions,
                cost_model=cost_model,
                confidence=confidence,
            )
        )
        return actions

    def _build_exploit_action(
        self,
        *,
        target_type: str,
        target_id: str,
        scope: ActionScope,
        preconditions: dict[str, bool | int | float | str],
        soft_preconditions: dict[str, float],
        effects: dict[str, EffectEstimate],
        cost_model: CostModel,
        confidence: ActionConfidence,
    ) -> SynthesizedAction:
        action_suffix = _action_suffix(target_type, target_id, scope)
        action_id = f"action_exploit_{action_suffix}_v1"
        policy_id = f"policy_exploit_{target_type}_v1"
        action_preconditions = dict(preconditions)
        if scope is ActionScope.INSTANCE:
            action_preconditions["target_id"] = target_id
        target_binding = (
            TargetBinding(mode="resource_id", resource_id=target_id)
            if scope is ActionScope.INSTANCE
            else TargetBinding(mode="current_target", required_type=target_type)
        )
        display_target = target_type if scope is ActionScope.TEMPLATE else action_suffix
        action = SynthesizedAction(
            schema_version=self._schema_version,
            action_id=action_id,
            display_name=f"Exploit {display_target.replace('_', ' ')}",
            scope=scope,
            lifecycle=ActionLifecycle.CANDIDATE,
            preconditions=action_preconditions,
            soft_preconditions=dict(soft_preconditions),
            effects=dict(effects),
            side_effects={},
            cost_model=cost_model,
            confidence=_copy_confidence(confidence),
            execution_payload=ExecutionPayload(
                type=ExecutorType.SCRIPTED_PRIMITIVE,
                policy_id=policy_id,
                policy_version=1,
                target_binding=target_binding,
            ),
        )

        if is_promotable(action.confidence):
            action.lifecycle = ActionLifecycle.TRUSTED

        return action

    def _build_travel_action(
        self,
        *,
        target_type: str,
        target_id: str,
        scope: ActionScope,
        preconditions: dict[str, bool | int | float | str],
        soft_preconditions: dict[str, float],
        cost_model: CostModel,
        confidence: ActionConfidence,
    ) -> SynthesizedAction:
        if scope is ActionScope.TEMPLATE:
            action_id = f"action_move_to_known_{target_type}_template_v1"
            display_target = target_type
            target_binding = TargetBinding(
                mode="current_target", required_type=target_type
            )
        else:
            action_id = f"action_move_to_{target_id}_v1"
            display_target = target_id
            target_binding = TargetBinding(mode="resource_id", resource_id=target_id)

        action_preconditions = dict(preconditions)
        if scope is ActionScope.INSTANCE:
            action_preconditions["target_id"] = target_id

        action = SynthesizedAction(
            schema_version=self._schema_version,
            action_id=action_id,
            display_name=f"Move to {display_target.replace('_', ' ')}",
            scope=scope,
            lifecycle=ActionLifecycle.CANDIDATE,
            preconditions=action_preconditions,
            soft_preconditions=dict(soft_preconditions),
            effects={},
            side_effects={},
            cost_model=cost_model,
            confidence=_copy_confidence(confidence),
            execution_payload=ExecutionPayload(
                type=ExecutorType.PATHFINDER,
                policy_id="policy_move_to_known_discoverable_v1",
                policy_version=1,
                target_binding=target_binding,
            ),
            symbolic_effects={
                "at_known_target": True,
                "at_discoverable": True,
                "visible_discoverable": True,
            },
        )
        if is_promotable(action.confidence):
            action.lifecycle = ActionLifecycle.TRUSTED
        return action


def _exploit_preconditions(
    inferred: dict[str, bool | int | float | str],
    target_type: str,
) -> dict[str, bool | int | float | str]:
    preconditions: dict[str, bool | int | float | str] = {}
    if inferred.get("at_discoverable") is True:
        preconditions["at_discoverable"] = True
    if target_type != "none":
        preconditions["target_type"] = target_type
    if inferred.get("target_has_resource") is True:
        preconditions["target_has_resource"] = True
    if inferred.get("has_target_location") is True:
        preconditions["has_target_location"] = True
    if inferred.get("at_known_target") is True:
        preconditions["at_known_target"] = True
    return preconditions


def _confidence_for(
    successful: list[Trajectory],
    failed: list[Trajectory],
) -> ActionConfidence:
    total = len(successful) + len(failed)
    succ_rate = len(successful) / total if total else 0.0
    death_ct = sum(1 for t in failed if t.end.needs.health <= 0.0)
    timeout_ct = sum(1 for t in failed if t.cost_ticks > 500)
    return ActionConfidence(
        trials=total,
        successful_trials=len(successful),
        failed_trials=len(failed),
        success_rate=round(succ_rate, 4),
        death_rate=round(death_ct / total, 4) if total else 0.0,
        timeout_rate=round(timeout_ct / total, 4) if total else 0.0,
        death_trials=death_ct,
        timeout_trials=timeout_ct,
    )


def _copy_confidence(confidence: ActionConfidence) -> ActionConfidence:
    return ActionConfidence(
        trials=confidence.trials,
        successful_trials=confidence.successful_trials,
        failed_trials=confidence.failed_trials,
        success_rate=confidence.success_rate,
        death_rate=confidence.death_rate,
        timeout_rate=confidence.timeout_rate,
        death_trials=confidence.death_trials,
        timeout_trials=confidence.timeout_trials,
    )


def _travel_distance_weight(successful: list[Trajectory]) -> float:
    values: list[float] = []
    for trajectory in successful:
        target_x = trajectory.start.symbolic.get("target_known_x")
        target_y = trajectory.start.symbolic.get("target_known_y")
        if not isinstance(target_x, int) or not isinstance(target_y, int):
            continue
        distance = abs(target_x - trajectory.start.x) + abs(
            target_y - trajectory.start.y
        )
        if distance > 0:
            values.append(float(trajectory.cost_ticks) / float(distance))
    if not values:
        return 1.0
    return sum(values) / float(len(values))


def _action_suffix(target_type: str, target_id: str, scope: ActionScope) -> str:
    if scope is ActionScope.TEMPLATE:
        return f"{target_type}_template"
    if target_id.startswith(f"{target_type}_"):
        return target_id
    target_parts = target_id.rsplit("_", maxsplit=1)
    if len(target_parts) == 2 and target_parts[1].isdigit():
        return f"{target_type}_{target_parts[1]}"
    return target_id
