"""The backtesting engine.

Execution model (deliberately conservative, no lookahead):

1. At the close of bar t, the strategy sees history up to and including t
   and emits target weights.
2. Those targets are filled at the *open of bar t+1*, adjusted for slippage,
   with commission charged on the traded notional.
3. Equity is marked to market at every close.

Fractional shares are allowed; this is a research engine, not an OMS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.costs import CostModel
from quantlab.metrics import summary
from quantlab.strategy import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    initial_capital: float

    def summary(self) -> pd.Series:
        return summary(self.returns)


@dataclass
class Backtester:
    prices: dict[str, pd.DataFrame]
    strategy: Strategy
    initial_capital: float = 100_000.0
    costs: CostModel = field(default_factory=CostModel.zero)
    rebalance_every: int = 1

    def run(self) -> BacktestResult:
        symbols = sorted(self.prices)
        if not symbols:
            raise ValueError("prices is empty")

        # Align all symbols on their common dates.
        index = self.prices[symbols[0]].index
        for sym in symbols[1:]:
            index = index.intersection(self.prices[sym].index)
        if len(index) < 2:
            raise ValueError("need at least two common bars across all symbols")
        aligned = {sym: self.prices[sym].loc[index] for sym in symbols}

        opens = pd.DataFrame({s: aligned[s]["open"] for s in symbols})
        closes = pd.DataFrame({s: aligned[s]["close"] for s in symbols})

        cash = self.initial_capital
        positions = {s: 0.0 for s in symbols}
        pending_targets: dict[str, float] | None = None

        equity = np.empty(len(index))
        weights = np.zeros((len(index), len(symbols)))
        trade_rows: list[dict] = []

        for t in range(len(index)):
            date = index[t]

            # 1. Fill orders decided at the previous close.
            if pending_targets is not None:
                cash = self._execute(
                    pending_targets, positions, cash,
                    opens.iloc[t], date, trade_rows,
                )
                pending_targets = None

            # 2. Mark to market at the close.
            close_row = closes.iloc[t]
            value = cash + sum(positions[s] * close_row[s] for s in symbols)
            equity[t] = value
            if value > 0:
                for j, s in enumerate(symbols):
                    weights[t, j] = positions[s] * close_row[s] / value

            # 3. Ask the strategy for new targets (visible history: bars 0..t).
            if t >= self.strategy.warmup and t % self.rebalance_every == 0 and t < len(index) - 1:
                history = {s: aligned[s].iloc[: t + 1] for s in symbols}
                targets = self.strategy.target_weights(history)
                self._validate_targets(targets, symbols)
                pending_targets = targets

        equity_s = pd.Series(equity, index=index, name="equity")
        returns = equity_s.pct_change().fillna(0.0)
        returns.name = "returns"
        weights_df = pd.DataFrame(weights, index=index, columns=symbols)
        trades_df = pd.DataFrame(
            trade_rows, columns=["date", "symbol", "shares", "fill_price", "commission"]
        )
        return BacktestResult(
            equity=equity_s,
            returns=returns,
            weights=weights_df,
            trades=trades_df,
            initial_capital=self.initial_capital,
        )

    def _execute(
        self,
        targets: dict[str, float],
        positions: dict[str, float],
        cash: float,
        open_row: pd.Series,
        date,
        trade_rows: list[dict],
    ) -> float:
        symbols = sorted(positions)
        # Conservative sizing: value the book at sell-side fill prices, so a
        # full rotation (sell everything, buy everything) can never overspend
        # cash because of slippage.
        value = cash + sum(
            positions[s] * self.costs.fill_price(float(open_row[s]), -1)
            for s in symbols
        )

        deltas: list[tuple[str, float, float]] = []
        for s in symbols:
            ref = float(open_row[s])
            desired_notional = targets.get(s, 0.0) * value
            naive_delta = desired_notional / ref - positions[s]
            if abs(naive_delta) * ref <= 1e-9:
                continue
            side = 1 if naive_delta > 0 else -1
            fill = self.costs.fill_price(ref, side)
            # Buys are sized at the (higher) buy fill; sells at the reference
            # open, so the post-trade holding matches the target notional.
            delta = desired_notional / fill - positions[s] if side == 1 else naive_delta
            deltas.append((s, delta, fill))
        # Sells first so the freed cash can fund the buys.
        deltas.sort(key=lambda x: x[1])

        for s, delta, fill in deltas:
            commission = self.costs.commission_for(delta, fill)
            cash -= delta * fill + commission
            positions[s] += delta
            trade_rows.append(
                {"date": date, "symbol": s, "shares": delta,
                 "fill_price": fill, "commission": commission}
            )
        return cash

    @staticmethod
    def _validate_targets(targets: dict[str, float], symbols: list[str]) -> None:
        unknown = set(targets) - set(symbols)
        if unknown:
            raise ValueError(f"strategy returned weights for unknown symbols: {unknown}")
        gross = sum(abs(w) for w in targets.values())
        if gross > 1.0 + 1e-9:
            raise ValueError(f"gross target weight {gross:.4f} exceeds 1.0 (no leverage)")
