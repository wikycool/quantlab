"""Compare momentum vs mean reversion vs buy-and-hold on synthetic data.

Run from the repo root:
    pip install -e .
    python examples/momentum_vs_meanrev.py
"""

import pandas as pd

from quantlab import (
    Backtester,
    BuyAndHold,
    CostModel,
    MeanReversion,
    Momentum,
    synthetic_prices,
)

prices = synthetic_prices(n_periods=756, symbols=("AAA", "BBB", "CCC", "DDD"), seed=11)
costs = CostModel.realistic()

strategies = {
    "buy_and_hold": BuyAndHold(),
    "momentum_60d": Momentum(lookback=60, top_k=2),
    "mean_rev_20d": MeanReversion(lookback=20, z_entry=1.0),
}

results = {}
for name, strat in strategies.items():
    result = Backtester(prices, strat, initial_capital=100_000, costs=costs,
                        rebalance_every=5).run()
    results[name] = result.summary()
    print(f"{name}: {len(result.trades)} trades, "
          f"final equity {result.equity.iloc[-1]:,.0f}")

print()
print(pd.DataFrame(results).round(3).to_string())
