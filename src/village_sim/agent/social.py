"""Vectorized social and market proximity algorithms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["compute_social_interactions", "compute_trade_opportunities"]


def compute_social_interactions(
    active_mask: NDArray[np.bool_],
    x_pos: NDArray[np.int32],
    y_pos: NDArray[np.int32],
    vision_radius: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Find ordered active-agent pairs within Chebyshev vision radius."""

    if vision_radius < 0:
        raise ValueError("vision_radius must be non-negative")
    if x_pos.shape != y_pos.shape or x_pos.shape != active_mask.shape:
        raise ValueError("active_mask, x_pos, and y_pos must share one shape")

    active_indices: NDArray[np.int64] = np.flatnonzero(active_mask).astype(
        np.int64,
        copy=False,
    )
    if active_indices.size < 2:
        empty: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        return empty, empty

    active_x: NDArray[np.int32] = x_pos[active_indices]
    active_y: NDArray[np.int32] = y_pos[active_indices]

    dx: NDArray[np.int32] = np.abs(active_x[:, np.newaxis] - active_x[np.newaxis, :])
    dy: NDArray[np.int32] = np.abs(active_y[:, np.newaxis] - active_y[np.newaxis, :])
    distance: NDArray[np.int32] = np.maximum(dx, dy)

    valid_pairs: NDArray[np.bool_] = distance <= vision_radius
    np.fill_diagonal(valid_pairs, False)
    buyer_offsets: NDArray[np.int64]
    seller_offsets: NDArray[np.int64]
    buyer_offsets, seller_offsets = np.where(valid_pairs)

    return active_indices[buyer_offsets], active_indices[seller_offsets]


def compute_trade_opportunities(
    active_mask: NDArray[np.bool_],
    x_pos: NDArray[np.int32],
    y_pos: NDArray[np.int32],
    buyer_needs: NDArray[np.float32],
    seller_surplus: NDArray[np.float32],
    vision_radius: int,
    need_threshold: float = 0.60,
    surplus_threshold: float = 0.40,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    """Identify valid trade pairs and unit trade values without Python loops."""

    if (
        buyer_needs.shape != active_mask.shape
        or seller_surplus.shape != active_mask.shape
    ):
        raise ValueError("need, surplus, and active arrays must share one shape")
    if buyer_needs.ndim != 1:
        raise ValueError("agent state arrays must be one-dimensional")

    buyer_indices: NDArray[np.int64]
    seller_indices: NDArray[np.int64]
    buyer_indices, seller_indices = compute_social_interactions(
        active_mask,
        x_pos,
        y_pos,
        vision_radius,
    )
    if buyer_indices.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    buyer_mask: NDArray[np.bool_] = buyer_needs[buyer_indices] > np.float32(
        need_threshold
    )
    seller_mask: NDArray[np.bool_] = seller_surplus[seller_indices] > np.float32(
        surplus_threshold
    )
    valid_trades: NDArray[np.bool_] = buyer_mask & seller_mask

    final_buyers: NDArray[np.int64] = buyer_indices[valid_trades]
    final_sellers: NDArray[np.int64] = seller_indices[valid_trades]
    trade_values: NDArray[np.float32] = np.full(
        final_buyers.size,
        np.float32(1.0),
        dtype=np.float32,
    )

    return final_buyers, final_sellers, trade_values
