"""Dense matrix economy ledger for vectorized trade resolution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["Economy"]


class Economy:
    """Array-backed economic ledger avoiding per-agent trade objects."""

    __slots__ = ("max_agents", "wealth", "debt_ledger", "inventory_surplus")

    def __init__(self, max_agents: int) -> None:
        if max_agents <= 0:
            raise ValueError("max_agents must be positive")
        self.max_agents = max_agents
        self.wealth: NDArray[np.float32] = np.zeros(max_agents, dtype=np.float32)
        self.debt_ledger: NDArray[np.float32] = np.zeros(
            (max_agents, max_agents), dtype=np.float32
        )
        self.inventory_surplus: NDArray[np.float32] = np.zeros(
            max_agents, dtype=np.float32
        )

    def batch_transact(
        self,
        buyer_indices: NDArray[np.int64],
        seller_indices: NDArray[np.int64],
        values: NDArray[np.float32],
    ) -> None:
        """Settle buyer-to-seller trades in batch with debt for unpaid value."""

        if (
            buyer_indices.shape != seller_indices.shape
            or buyer_indices.shape != values.shape
        ):
            raise ValueError(
                "buyer_indices, seller_indices, and values must share one shape"
            )
        if buyer_indices.ndim != 1:
            raise ValueError("trade arrays must be one-dimensional")
        if buyer_indices.size == 0:
            return
        if np.any(values < np.float32(0.0)):
            raise ValueError("trade values must be non-negative")
        if np.any(buyer_indices < 0) or np.any(seller_indices < 0):
            raise IndexError("agent indices must be non-negative")
        if np.any(buyer_indices >= self.max_agents) or np.any(
            seller_indices >= self.max_agents
        ):
            raise IndexError("agent index exceeds economy capacity")

        total_due_by_buyer: NDArray[np.float32] = np.zeros(
            self.max_agents, dtype=np.float32
        )
        np.add.at(total_due_by_buyer, buyer_indices, values)

        buyer_total_due: NDArray[np.float32] = total_due_by_buyer[buyer_indices]
        safe_due: NDArray[np.float32] = np.maximum(buyer_total_due, np.float32(1.0))
        buyer_payment_budget: NDArray[np.float32] = np.minimum(
            self.wealth[buyer_indices], buyer_total_due
        ).astype(np.float32, copy=False)
        paid_values: NDArray[np.float32] = (
            values * (buyer_payment_budget / safe_due)
        ).astype(np.float32, copy=False)
        remaining_debt: NDArray[np.float32] = (values - paid_values).astype(
            np.float32, copy=False
        )

        np.add.at(self.wealth, seller_indices, paid_values)
        total_paid_by_buyer: NDArray[np.float32] = np.zeros(
            self.max_agents, dtype=np.float32
        )
        np.add.at(total_paid_by_buyer, buyer_indices, paid_values)
        self.wealth -= total_paid_by_buyer
        np.maximum(self.wealth, np.float32(0.0), out=self.wealth)

        debt_mask: NDArray[np.bool_] = remaining_debt > np.float32(0.0)
        np.add.at(
            self.debt_ledger,
            (buyer_indices[debt_mask], seller_indices[debt_mask]),
            remaining_debt[debt_mask],
        )
        np.fill_diagonal(self.debt_ledger, np.float32(0.0))

    def settle_trades(
        self,
        buyers: NDArray[np.int64],
        sellers: NDArray[np.int64],
        values: NDArray[np.float32],
    ) -> None:
        """Backward-compatible alias for batch trade settlement."""

        self.batch_transact(buyers, sellers, values)

    def net_worth(self) -> NDArray[np.float32]:
        """Return liquid wealth plus receivables minus payables."""

        receivables: NDArray[np.float32] = self.debt_ledger.sum(axis=0)
        payables: NDArray[np.float32] = self.debt_ledger.sum(axis=1)
        return (self.wealth + receivables - payables).astype(np.float32, copy=False)
