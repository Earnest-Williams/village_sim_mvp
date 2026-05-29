"""Orchestrator: observes trajectories and synthesizes GOAP actions (§4, §10–§17)."""

from __future__ import annotations

from village_sim.orchestrator.action_model import (
    ActionConfidence,
    ActionLifecycle,
    ActionScope,
    CostModel,
    EffectEstimate,
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
    evaluate_fatigue_task,
    evaluate_hunger_task,
    evaluate_thirst_task,
)
from village_sim.orchestrator.induction import (
    average_cost,
    infer_hard_preconditions,
    infer_need_effect,
    infer_soft_preconditions,
)
from village_sim.orchestrator.trajectory import Trajectory

_EVALUATORS = {
    NeedName.HUNGER: evaluate_hunger_task,
    NeedName.THIRST: evaluate_thirst_task,
    NeedName.FATIGUE: evaluate_fatigue_task,
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

        is_success = False
        for need, evaluator in _EVALUATORS.items():
            result: TaskResult = evaluator(trajectory)
            if result.success:
                is_success = True
                break

        if is_success:
            self._clusters.setdefault(key, []).append(trajectory)
        else:
            self._failed.setdefault(key, []).append(trajectory)

    def synthesize_all(self) -> list[SynthesizedAction]:
        """Attempt to synthesize one action per cluster that has enough data."""
        actions: list[SynthesizedAction] = []
        for key, successful in self._clusters.items():
            if len(successful) < 10:
                continue  # insufficient evidence
            failed = self._failed.get(key, [])
            action = self._synthesize_cluster(key, successful, failed)
            if action is not None:
                actions.append(action)
        return actions

    def _synthesize_cluster(
        self,
        cluster_key: str,
        successful: list[Trajectory],
        failed: list[Trajectory],
    ) -> SynthesizedAction | None:
        parts = cluster_key.split(":")
        task_name = parts[0] if len(parts) > 0 else "unknown"
        target_type = parts[1] if len(parts) > 1 else "none"

        # Representative trajectory for payload extraction
        rep = successful[0]
        target_id = str(rep.start.symbolic.get("target_id", "unknown"))

        action_id = f"action_exploit_{target_id}_v1"
        policy_id = f"policy_exploit_{target_type}_v1"

        preconditions = infer_hard_preconditions(successful, failed)
        if not preconditions:
            return None  # cannot build a meaningful action without preconditions

        soft_preconditions = infer_soft_preconditions(successful, failed)

        effects: dict[str, EffectEstimate] = {}
        for need in NeedName:
            estimate = infer_need_effect(successful, need)
            if estimate is not None:
                effects[f"{need.value}_delta"] = estimate

        if not effects:
            return None

        base_ticks = average_cost(successful)
        total = len(successful) + len(failed)
        succ_rate = len(successful) / total if total else 0.0
        death_ct = sum(1 for t in failed if t.end.needs.health <= 0.0)
        timeout_ct = sum(1 for t in failed if t.cost_ticks > 500)

        scope = (
            ActionScope.INSTANCE
            if target_id not in ("none", "unknown")
            else ActionScope.TEMPLATE
        )
        target_binding = (
            TargetBinding(mode="resource_id", resource_id=target_id)
            if scope is ActionScope.INSTANCE
            else TargetBinding(mode="current_target", required_type=target_type)
        )

        action = SynthesizedAction(
            schema_version=self._schema_version,
            action_id=action_id,
            display_name=f"Exploit {target_id.replace('_', ' ')}",
            scope=scope,
            lifecycle=ActionLifecycle.CANDIDATE,
            preconditions=preconditions,
            soft_preconditions=soft_preconditions,
            effects=effects,
            side_effects={},
            cost_model=CostModel(
                base_ticks=round(base_ticks, 1),
                distance_weight=0.0,
                fatigue_weight=1.0,
                night_multiplier=1.25,
            ),
            confidence=ActionConfidence(
                trials=total,
                successful_trials=len(successful),
                failed_trials=len(failed),
                success_rate=round(succ_rate, 4),
                death_rate=round(death_ct / total, 4) if total else 0.0,
                timeout_rate=round(timeout_ct / total, 4) if total else 0.0,
            ),
            execution_payload=ExecutionPayload(
                type=ExecutorType.RL_POLICY,
                policy_id=policy_id,
                policy_version=1,
                target_binding=target_binding,
            ),
        )

        if is_promotable(action.confidence):
            action.lifecycle = ActionLifecycle.TRUSTED

        return action
