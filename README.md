# quantlab

An event-driven backtesting engine for strategy research, built around one rule: **the engine should make it structurally impossible to cheat.**

Most retail backtests are quietly broken in two ways — lookahead bias (the strategy sees prices it couldn't have known) and free execution (fills at close, zero costs). quantlab makes both impossible by construction:

- Strategies only ever receive history up to the current bar.
- Orders decided at the close of bar *t* are filled at the **open of bar t+1**, through explicit slippage and commission models.
- Leverage above gross 1.0 is rejected — if your strategy only works levered, you should know that.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python examples/momentum_vs_meanrev.py
```

```python
from quantlab import Backtester, Momentum, CostModel, synthetic_prices

prices = synthetic_prices(n_periods=504, symbols=("AAA", "BBB", "CCC"))
result = Backtester(
    prices,
    Momentum(lookback=60, top_k=1),
    initial_capital=100_000,
    costs=CostModel.realistic(),  # 5 bps slippage, 1 bp commission
).run()

print(result.summary())
# total_return, cagr, annual_vol, sharpe, sortino, max_drawdown, calmar
```

## Writing a strategy

Implement one method. The engine handles execution, costs, and accounting.

```python
from quantlab import Strategy

class GoldenCross(Strategy):
    warmup = 200

    def target_weights(self, history):
        weights = {}
        for symbol, df in history.items():
            fast = df["close"].iloc[-50:].mean()
            slow = df["close"].iloc[-200:].mean()
            if fast > slow:
                weights[symbol] = 1.0
        n = len(weights)
        return {s: 1.0 / n for s in weights} if n else {}
```

`history` maps each symbol to its OHLCV DataFrame up to and including the current bar. Return target weights as fractions of equity; anything unallocated stays in cash.

## Design notes

- **Execution model.** Decision at close of *t*, fill at open of *t+1*. This is the most conservative assumption available from daily bars and eliminates the most common source of inflated backtest Sharpe.
- **Costs are first-class.** `CostModel` composes a slippage model (`FixedSlippage`) with a commission model (`BpsCommission`, `PerShareCommission`). `CostModel.realistic()` is a sane default for liquid equities; `CostModel.zero()` exists for controlled experiments, not for reporting results.
- **Plain pandas out.** `BacktestResult` holds the equity curve, per-bar returns, realized weights, and a full trade log as ordinary Series/DataFrames.
- **Fractional shares.** This is a research engine, not an order management system, so position sizing is exact.

## What it doesn't do (yet)

Shorting and leverage, intrabar fills, borrow costs, and multi-currency support are deliberately out of scope for v0.1. The roadmap prioritizes a walk-forward validation harness and parameter sensitivity reports, because overfitting is a bigger enemy than missing features.

## Testing

The test suite covers metric correctness against hand-computed values, cost model arithmetic, fill timing (fills verifiably land at next-bar opens), warmup behavior, and rejection of leveraged or malformed strategies.

```bash
pytest -v
```

## License

MIT
