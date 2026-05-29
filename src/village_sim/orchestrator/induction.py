"""Precondition and effect inference (§14, §15, §16)."""

from __future__ import annotations

from dataclasses import dataclass

from village_sim.orchestrator.evaluator import NeedName
from village_sim.orchestrator.symbolic import FactValue, SymbolicState
from village_sim.orchestrator.trajectory import Trajectory


# ── Precondition inference (§14) ─────────────────────────────────────────────


def fact_frequency(
    trajectories: list[Trajectory],
    fact: str,
    value: FactValue,
) -> float:
    """Return the fraction of trajectories where fact==value at the start."""
    if not trajectories:
        return 0.0
    matches = sum(
        1 for t in trajectories if t.start.symbolic.get(fact) == value
    )
    return matches / len(trajectories)


def candidate_start_facts(
    trajectories: list[Trajectory],
) -> dict[str, set[FactValue]]:
    """Collect every (fact, value) pair seen at the start of any trajectory."""
    values: dict[str, set[FactValue]] = {}
    for trajectory in trajectories:
        for key, value in trajectory.start.symbolic.items():
            values.setdefault(key, set()).add(value)
    return values


def infer_hard_preconditions(
    successful: list[Trajectory],
    failed: list[Trajectory],
    min_success_freq: float = 0.85,
    min_failure_gap: float = 0.25,
) -> dict[str, FactValue]:
    """Return facts that are nearly always true at the start of successful runs
    but rare (or much rarer) at the start of failed runs.
    """
    preconditions: dict[str, FactValue] = {}
    candidates = candidate_start_facts(successful)

    for fact, values in candidates.items():
        for value in values:
            success_freq = fact_frequency(successful, fact, value)
            failure_freq = fact_frequency(failed, fact, value)
            gap = success_freq - failure_freq
            if success_freq >= min_success_freq and gap >= min_failure_gap:
                preconditions[fact] = value

    return preconditions


def infer_soft_preconditions(
    successful: list[Trajectory],
    failed: list[Trajectory],
    min_success_freq: float = 0.50,
    max_success_freq: float = 0.85,
) -> dict[str, float]:
    """Return facts that help but are not essential; value = success frequency."""
    soft: dict[str, float] = {}
    candidates = candidate_start_facts(successful)

    for fact, values in candidates.items():
        for value in values:
            freq = fact_frequency(successful, fact, value)
            if min_success_freq <= freq < max_success_freq:
                soft[f"{fact}={value}"] = round(freq, 4)

    return soft


# ── Effect inference (§15) ───────────────────────────────────────────────────


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * q)
    return ordered[index]


@dataclass(slots=True)
class EffectEstimate:
    mean: float
    p10: float
    p90: float
    confidence: float


def infer_need_effect(
    successful: list[Trajectory],
    need: NeedName,
    min_abs_delta: float = 0.10,
) -> EffectEstimate | None:
    """Estimate the distribution of need change across successful trajectories."""
    deltas: list[float] = []

    for t in successful:
        before: float = getattr(t.start.needs, need.value)
        after: float = getattr(t.end.needs, need.value)
        delta: float = after - before
        if abs(delta) >= min_abs_delta:
            deltas.append(delta)

    if not deltas:
        return None

    mean = sum(deltas) / len(deltas)
    confidence = len(deltas) / len(successful)

    return EffectEstimate(
        mean=round(mean, 4),
        p10=round(_percentile(deltas, 0.10), 4),
        p90=round(_percentile(deltas, 0.90), 4),
        confidence=round(confidence, 4),
    )


# ── Cost inference (§16) ─────────────────────────────────────────────────────


def average_cost(trajectories: list[Trajectory]) -> float:
    if not trajectories:
        return 0.0
    total = sum(t.cost_ticks for t in trajectories)
    return total / len(trajectories)
