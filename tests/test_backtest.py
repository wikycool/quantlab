import numpy as np
import pandas as pd
import pytest

from quantlab import (
    Backtester,
    BuyAndHold,
    CostModel,
    FixedSlippage,
    MeanReversion,
    Momentum,
    Strategy,
    synthetic_prices,
)


@pytest.fixture
def prices():
    return synthetic_prices(n_periods=252, symbols=("AAA", "BBB", "CCC"), seed=42)


def test_buy_and_hold_zero_costs_matches_asset_return():
    prices = synthetic_prices(n_periods=100, symbols=("AAA",), seed=1)
    bt = Backtester(prices, BuyAndHold(), initial_capital=10_000.0,
                    costs=CostModel.zero())
    result = bt.run()
    df = prices["AAA"]
    # Decision at bar 0 close, fill at bar 1 open, fully invested after that.
    expected = 10_000.0 * df["close"].iloc[-1] / df["open"].iloc[1]
    assert result.equity.iloc[-1] == pytest.approx(expected)


def test_fills_happen_at_next_bar_open(prices):
    bt = Backtester(prices, BuyAndHold(), costs=CostModel.zero())
    result = bt.run()
    first_trades = result.trades[result.trades["date"] == result.trades["date"].min()]
    fill_date = first_trades["date"].iloc[0]
    # BuyAndHold decides at bar 0, so all first fills are at bar 1's open.
    assert fill_date == result.equity.index[1]
    for _, row in first_trades.iterrows():
        assert row["fill_price"] == pytest.approx(
            prices[row["symbol"]].loc[fill_date, "open"]
        )


def test_costs_strictly_reduce_final_equity(prices):
    free = Backtester(prices, Momentum(lookback=20, top_k=1),
                      costs=CostModel.zero()).run()
    costly = Backtester(prices, Momentum(lookback=20, top_k=1),
                        costs=CostModel.realistic()).run()
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert (costly.trades["commission"] > 0).all()


def test_no_trades_before_warmup(prices):
    strat = Momentum(lookback=50, top_k=2)
    result = Backtester(prices, strat).run()
    if not result.trades.empty:
        first_fill = result.trades["date"].min()
        # First decision at bar index `warmup`, fill one bar later.
        assert first_fill >= result.equity.index[strat.warmup + 1]
    # Equity is flat at initial capital during warmup.
    assert (result.equity.iloc[: strat.warmup] == result.initial_capital).all()


def test_leverage_is_rejected(prices):
    class Leveraged(Strategy):
        def target_weights(self, history):
            return {sym: 1.0 for sym in history}  # gross 3.0

    with pytest.raises(ValueError, match="exceeds 1.0"):
        Backtester(prices, Leveraged()).run()


def test_unknown_symbol_is_rejected(prices):
    class Confused(Strategy):
        def target_weights(self, history):
            return {"ZZZ": 1.0}

    with pytest.raises(ValueError, match="unknown symbols"):
        Backtester(prices, Confused()).run()


def test_mean_reversion_runs_end_to_end(prices):
    result = Backtester(prices, MeanReversion(lookback=20, z_entry=1.0),
                        costs=CostModel(slippage=FixedSlippage(0.0005))).run()
    assert np.isfinite(result.equity).all()
    assert (result.equity > 0).all()
    # Realized weights never exceed gross 1 (plus tiny numerical noise).
    assert (result.weights.abs().sum(axis=1) <= 1.0 + 1e-6).all()


def test_weights_index_matches_equity_index(prices):
    result = Backtester(prices, Momentum(lookback=30)).run()
    pd.testing.assert_index_equal(result.weights.index, result.equity.index)
    assert result.returns.iloc[0] == 0.0
