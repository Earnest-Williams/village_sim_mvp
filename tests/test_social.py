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

    teachers, learners = compute_social_interactions(active, x_pos, y_pos, 1)

    pairs = set(zip(teachers.tolist(), learners.tolist(), strict=True))
    assert pairs == {(0, 1), (1, 0), (1, 3), (3, 1)}


def test_compute_trade_opportunities_filters_nearby_need_and_surplus() -> None:
    x_pos = np.asarray([0, 1, 2, 5], dtype=np.int32)
    y_pos = np.asarray([0, 0, 0, 0], dtype=np.int32)
    active = np.asarray([True, True, True, True], dtype=np.bool_)
    buyer_needs = np.asarray([0.7, 0.0, 0.8, 0.9], dtype=np.float32)
    seller_surplus = np.asarray([0.0, 0.7, 0.8, 0.9], dtype=np.float32)

    buyers, sellers, values = compute_trade_opportunities(
        active,
        x_pos,
        y_pos,
        buyer_needs,
        seller_surplus,
        1,
    )

    assert buyers.tolist() == [0, 2]
    assert sellers.tolist() == [1, 1]
    assert values.tolist() == [1.0, 1.0]
