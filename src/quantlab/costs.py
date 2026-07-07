"""Transaction cost models: commission and slippage.

Slippage adjusts the fill price against the trader; commission is charged
on top of the fill notional.
"""

from __future__ import annotations

from dataclasses import dataclass


class CostModel:
    """Combines a slippage model and a commission model."""

    def __init__(self, slippage: "FixedSlippage | None" = None,
                 commission: "PerShareCommission | BpsCommission | None" = None):
        self.slippage = slippage or FixedSlippage(0.0)
        self.commission = commission or BpsCommission(0.0)

    def fill_price(self, reference_price: float, side: int) -> float:
        """side is +1 for buy, -1 for sell."""
        return self.slippage.fill_price(reference_price, side)

    def commission_for(self, shares: float, fill_price: float) -> float:
        return self.commission.charge(shares, fill_price)

    @classmethod
    def zero(cls) -> "CostModel":
        return cls()

    @classmethod
    def realistic(cls) -> "CostModel":
        """5 bps slippage, 1 bp commission: a reasonable default for liquid equities."""
        return cls(slippage=FixedSlippage(0.0005), commission=BpsCommission(0.0001))


@dataclass
class FixedSlippage:
    """Fill price moves against the trader by a fixed fraction of price."""

    rate: float

    def fill_price(self, reference_price: float, side: int) -> float:
        if side not in (1, -1):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        return reference_price * (1.0 + side * self.rate)


@dataclass
class PerShareCommission:
    """Fixed fee per share traded, with an optional minimum per order."""

    per_share: float
    minimum: float = 0.0

    def charge(self, shares: float, fill_price: float) -> float:
        return max(abs(shares) * self.per_share, self.minimum if shares else 0.0)


@dataclass
class BpsCommission:
    """Commission as a fraction of traded notional."""

    rate: float

    def charge(self, shares: float, fill_price: float) -> float:
        return abs(shares) * fill_price * self.rate
