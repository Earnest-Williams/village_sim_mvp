"""Dense vectorized economic ledger for town-scale settlement accounting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Economy:
    """Array-backed wealth and debt ledger indexed directly by agent id."""

    def __init__(self, max_agents: int) -> None:
        if max_agents <= 0:
            raise ValueError("max_agents must be positive")
        self.max_agents = max_agents
        self.wealth: NDArray[np.float32] = np.zeros(max_agents, dtype=np.float32)
        self.debt_ledger: NDArray[np.float32] = np.zeros(
            (max_agents, max_agents), dtype=np.float32
        )

    def settle_trades(
        self,
        buyers: NDArray[np.int64],
        sellers: NDArray[np.int64],
        values: NDArray[np.float32],
    ) -> None:
        """Settle many buyer-to-seller payments with vectorized debt fallback."""

        if buyers.shape != sellers.shape or buyers.shape != values.shape:
            raise ValueError("buyers, sellers, and values must share one shape")
        if buyers.ndim != 1:
            raise ValueError("trade arrays must be one-dimensional")
        if buyers.size == 0:
            return
        if np.any(values < np.float32(0.0)):
            raise ValueError("trade values must be non-negative")
        if np.any(buyers < 0) or np.any(sellers < 0):
            raise IndexError("agent indices must be non-negative")
        if np.any(buyers >= self.max_agents) or np.any(sellers >= self.max_agents):
            raise IndexError("agent index exceeds economy capacity")

        np.add.at(self.wealth, sellers, values)
        buyer_funds: NDArray[np.float32] = self.wealth[buyers]
        paid_values: NDArray[np.float32] = np.minimum(buyer_funds, values).astype(
            np.float32,
            copy=False,
        )
        debt_values: NDArray[np.float32] = (values - paid_values).astype(
            np.float32,
            copy=False,
        )
        np.add.at(self.wealth, buyers, -paid_values)
        debt_mask: NDArray[np.bool_] = debt_values > np.float32(0.0)
        np.add.at(
            self.debt_ledger,
            (buyers[debt_mask], sellers[debt_mask]),
            debt_values[debt_mask],
        )
        np.fill_diagonal(self.debt_ledger, np.float32(0.0))

    def net_worth(self) -> NDArray[np.float32]:
        """Return liquid wealth plus receivables minus payables."""

        receivables: NDArray[np.float32] = self.debt_ledger.sum(axis=0)
        payables: NDArray[np.float32] = self.debt_ledger.sum(axis=1)
        return (self.wealth + receivables - payables).astype(np.float32, copy=False)
