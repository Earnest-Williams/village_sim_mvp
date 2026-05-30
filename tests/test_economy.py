"""Dense economy ledger tests."""

from __future__ import annotations

import numpy as np

from village_sim.sim.economy import Economy


def test_settle_trades_uses_debt_when_buyer_lacks_wealth() -> None:
    economy = Economy(max_agents=3)
    economy.wealth[0] = np.float32(2.0)

    economy.settle_trades(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 2], dtype=np.int64),
        np.asarray([5.0, 4.0], dtype=np.float32),
    )

    assert float(economy.wealth[0]) == 0.0
    assert float(economy.wealth[1]) == 0.0
    assert float(economy.wealth[2]) == 9.0
    assert float(economy.debt_ledger[0, 2]) == 3.0
    assert float(economy.debt_ledger[1, 2]) == 4.0
    assert float(economy.debt_ledger[2, 2]) == 0.0
