"""Small helpers for preserving text viewport state across control replacement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextViewState:
    """Serializable view state for a scrollable text-like control."""

    horizontal_scroll: int
    vertical_scroll: int
    insertion_point: int


def clamp_insertion_point(insertion_point: int, text_length: int) -> int:
    """Clamp an insertion point so replacing text does not force the view origin."""

    if insertion_point < 0:
        return 0
    if insertion_point > text_length:
        return text_length
    return insertion_point
