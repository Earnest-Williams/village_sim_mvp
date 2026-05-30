"""Typed decision tracing for agent policy choices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DecisionSource(StrEnum):
    """Stable sources explaining why a resource-seeking action happened."""

    NONE = "none"
    EXPLORE = "explore"
    VISIBLE_RESOURCE = "visible_resource"
    REMEMBERED_RESOURCE = "remembered_resource"
    SEARCH_NEAR_MEMORY = "search_near_memory"
    CURRENT_TILE_RESOURCE = "current_tile_resource"
    GOAP = "goap"


@dataclass(slots=True)
class DecisionTrace:
    """Explanation for the agent's current resource-seeking decision."""

    source: DecisionSource = DecisionSource.NONE
    target_kind: str = "none"
    target_x: int = -1
    target_y: int = -1
    memory_confidence: float = 0.0
    memory_successful_uses: int = 0
    memory_failed_uses: int = 0
    reason: str = ""
