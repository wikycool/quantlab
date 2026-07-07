"""Strategy interface and reference implementations.

A strategy receives price history up to and including the current bar and
returns target portfolio weights. It never sees the future: the engine
guarantees that fills happen at the *next* bar's open.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    #: number of bars the strategy needs before it starts emitting weights
    warmup: int = 0

    @abstractmethod
    def target_weights(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Return target weights (fraction of equity) per symbol.

        Weights may sum to less than 1 (remainder held in cash).
        Symbols omitted from the dict are treated as weight 0.
        """


class BuyAndHold(Strategy):
    """Equal-weight everything on the first eligible bar, then never trade."""

    def __init__(self) -> None:
        self._weights: dict[str, float] | None = None

    def target_weights(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        if self._weights is None:
            n = len(history)
            self._weights = {sym: 1.0 / n for sym in history}
        return self._weights


class Momentum(Strategy):
    """Cross-sectional momentum: long the top_k assets by trailing return."""

    def __init__(self, lookback: int = 60, top_k: int = 1) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback
        self.top_k = top_k
        self.warmup = lookback

    def target_weights(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for sym, df in history.items():
            closes = df["close"]
            scores[sym] = closes.iloc[-1] / closes.iloc[-self.lookback] - 1.0
        winners = sorted(scores, key=scores.get, reverse=True)[: self.top_k]
        return {sym: 1.0 / len(winners) for sym in winners}


class MeanReversion(Strategy):
    """Long assets trading below their rolling mean by more than z_entry sigmas."""

    def __init__(self, lookback: int = 20, z_entry: float = 1.0) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback
        self.z_entry = z_entry
        self.warmup = lookback

    def target_weights(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        longs: list[str] = []
        for sym, df in history.items():
            window = df["close"].iloc[-self.lookback :]
            mean, std = window.mean(), window.std()
            if std == 0:
                continue
            z = (window.iloc[-1] - mean) / std
            if z < -self.z_entry:
                longs.append(sym)
        if not longs:
            return {}
        return {sym: 1.0 / len(longs) for sym in longs}
