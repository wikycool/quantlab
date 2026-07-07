"""quantlab: event-driven backtesting with realistic costs.

Design principles:
- No lookahead: strategies only ever see data up to the current bar,
  and orders decided at bar t are filled at the open of bar t+1.
- Costs are first-class: commission and slippage models are explicit
  inputs, not afterthoughts.
- Results are plain pandas objects, easy to analyze and plot.
"""

from quantlab.backtest import Backtester, BacktestResult
from quantlab.costs import CostModel, FixedSlippage, PerShareCommission, BpsCommission
from quantlab.data import synthetic_prices, load_csv
from quantlab.metrics import summary, sharpe, sortino, max_drawdown, cagr, calmar
from quantlab.strategy import Strategy, BuyAndHold, Momentum, MeanReversion

__all__ = [
    "Backtester",
    "BacktestResult",
    "CostModel",
    "FixedSlippage",
    "PerShareCommission",
    "BpsCommission",
    "synthetic_prices",
    "load_csv",
    "summary",
    "sharpe",
    "sortino",
    "max_drawdown",
    "cagr",
    "calmar",
    "Strategy",
    "BuyAndHold",
    "Momentum",
    "MeanReversion",
]

__version__ = "0.1.0"
