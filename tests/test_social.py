"""Vectorized social interaction tests."""

from __future__ import annotations

import numpy as np

from village_sim.agent.social import (
    compute_social_interactions,
    compute_trade_opportunities,
)


def test_compute_social_interactions_uses_active_chebyshev_radius() -> None:
    x_pos = np.asarray([0, 1, 3, 1], dtype=np.int32)
    y_pos = np.asarray([0, 1, 0, 2], dtype=np.int32)
    active = np.asarray([True, True, False, True], dtype=np.bool_)

    teachers, learners = compute_social_interactions(x_pos, y_pos, active, 1)

    pairs = set(zip(teachers.tolist(), learners.tolist(), strict=True))
    assert pairs == {(0, 1), (1, 0), (1, 3), (3, 1)}


def test_compute_trade_opportunities_filters_adjacent_need_and_surplus() -> None:
    buyer_needs = np.asarray([0.5, 0.0], dtype=np.float32)
    seller_surplus = np.asarray([0.0, 0.7, 0.8], dtype=np.float32)
    dist_matrix = np.asarray(
        [[1, 1, 2], [1, 1, 1]],
        dtype=np.int32,
    )

    buyers, sellers = compute_trade_opportunities(
        buyer_needs, seller_surplus, dist_matrix
    )

    assert buyers.tolist() == [0]
    assert sellers.tolist() == [1]
