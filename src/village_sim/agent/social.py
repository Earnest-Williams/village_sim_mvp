"""Vectorized social proximity computations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TRADE_NEED_THRESHOLD = np.float32(0.0)
TRADE_SURPLUS_THRESHOLD = np.float32(0.0)
TRADE_ADJACENT_DISTANCE = 1


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


def compute_trade_opportunities(
    buyer_needs: NDArray[np.float32],
    seller_surplus: NDArray[np.float32],
    dist_matrix: NDArray[np.int32],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return buyer/seller index pairs that can trade adjacent surplus."""

    if buyer_needs.ndim != 1 or seller_surplus.ndim != 1:
        raise ValueError("buyer_needs and seller_surplus must be one-dimensional")
    expected_shape = (buyer_needs.shape[0], seller_surplus.shape[0])
    if dist_matrix.shape != expected_shape:
        raise ValueError("dist_matrix shape must be buyer_count by seller_count")

    need_mask: NDArray[np.bool_] = buyer_needs > TRADE_NEED_THRESHOLD
    surplus_mask: NDArray[np.bool_] = seller_surplus > TRADE_SURPLUS_THRESHOLD
    adjacent_mask: NDArray[np.bool_] = (dist_matrix <= TRADE_ADJACENT_DISTANCE) & (
        dist_matrix > 0
    )
    trade_mask: NDArray[np.bool_] = (
        need_mask[:, np.newaxis] & surplus_mask[np.newaxis, :] & adjacent_mask
    )
    buyer_idx: NDArray[np.int64]
    seller_idx: NDArray[np.int64]
    buyer_idx, seller_idx = np.where(trade_mask)
    return buyer_idx.astype(np.int64, copy=False), seller_idx.astype(
        np.int64, copy=False
    )
