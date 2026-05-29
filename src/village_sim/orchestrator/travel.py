"""Focused recording helpers for known-discoverable travel trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from village_sim.orchestrator.trajectory import StateSnapshot


@dataclass(slots=True)
class TravelAttempt:
    target_id: str
    target_type: str
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    success: bool
    ticks: int


def should_record_travel_segment(
    before: StateSnapshot,
    after: StateSnapshot,
) -> bool:
    """Return true when a segment reaches a remembered target location."""
    target_id = before.symbolic.get("target_id", "none")
    return (
        before.symbolic.get("has_target_location") is True
        and before.symbolic.get("at_known_target") is False
        and after.symbolic.get("at_known_target") is True
        and target_id != "none"
    )
