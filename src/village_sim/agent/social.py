"""Vectorized social proximity computations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_social_interactions(
    x_pos: NDArray[np.int32],
    y_pos: NDArray[np.int32],
    active_mask: NDArray[np.bool_],
    radius: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return teacher/learner index pairs within Chebyshev social radius."""

    if radius < 0:
        raise ValueError("social radius must be non-negative")
    if x_pos.shape != y_pos.shape or x_pos.shape != active_mask.shape:
        raise ValueError("x, y, and active arrays must share one shape")

    active_indices: NDArray[np.int64] = np.flatnonzero(active_mask).astype(
        np.int64,
        copy=False,
    )
    if active_indices.size <= 1:
        empty: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        return empty, empty

    active_x: NDArray[np.int32] = x_pos[active_indices]
    active_y: NDArray[np.int32] = y_pos[active_indices]
    dx: NDArray[np.int32] = np.abs(active_x[:, np.newaxis] - active_x[np.newaxis, :])
    dy: NDArray[np.int32] = np.abs(active_y[:, np.newaxis] - active_y[np.newaxis, :])
    distance: NDArray[np.int32] = np.maximum(dx, dy)
    proximity_mask: NDArray[np.bool_] = (distance <= radius) & (distance > 0)
    source_offsets: NDArray[np.int64]
    target_offsets: NDArray[np.int64]
    source_offsets, target_offsets = np.where(proximity_mask)
    return active_indices[source_offsets], active_indices[target_offsets]
