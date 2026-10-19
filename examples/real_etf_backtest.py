"""Backtest the reference strategies on real daily ETF data, with and without costs.

Downloads split/dividend-adjusted daily bars from Yahoo Finance (via yfinance)
into a local cache directory; no data is committed to the repo.

    pip install -e . yfinance
    python examples/real_etf_backtest.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

from quantlab import Backtester, BuyAndHold, CostModel, MeanReversion, Momentum, load_csv

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START, END = "2010-01-01", "2026-01-01"
CACHE = Path(__file__).parent / "data"


def fetch(ticker: str) -> pd.DataFrame:
    path = CACHE / f"{ticker}.csv"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df.index.name = "date"
        df.to_csv(path)
    return load_csv(str(path))


prices = {t: fetch(t) for t in TICKERS}
strategies = {
    "buy_and_hold": BuyAndHold,
    "momentum_60d": lambda: Momentum(lookback=60, top_k=1),
    "mean_rev_20d": lambda: MeanReversion(lookback=20, z_entry=1.0),
}
costs = {"zero": CostModel.zero(), "realistic": CostModel.realistic()}

rows = []
for name, make in strategies.items():
    for cost_name, cost in costs.items():
        result = Backtester(prices, make(), initial_capital=100_000, costs=cost).run()
        s = result.summary()
        rows.append({
            "strategy": name, "costs": cost_name,
            "total_return": s["total_return"], "cagr": s["cagr"],
            "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"],
            "trades": len(result.trades),
        })

print(pd.DataFrame(rows).round(3).to_string(index=False))
