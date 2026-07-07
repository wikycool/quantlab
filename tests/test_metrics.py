import numpy as np
import pandas as pd
import pytest

from quantlab.metrics import (
    cagr,
    calmar,
    max_drawdown,
    sharpe,
    sortino,
    summary,
    total_return,
)


def test_total_return_compounds():
    r = pd.Series([0.10, 0.10])
    assert total_return(r) == pytest.approx(0.21)


def test_cagr_of_constant_daily_return():
    r = pd.Series([0.001] * 252)
    assert cagr(r) == pytest.approx(1.001**252 - 1)


def test_max_drawdown_known_path():
    # equity: 1.10 -> 0.55, peak 1.10 => drawdown -50%
    r = pd.Series([0.10, -0.50])
    assert max_drawdown(r) == pytest.approx(-0.50)


def test_max_drawdown_monotonic_up_is_zero():
    r = pd.Series([0.01] * 50)
    assert max_drawdown(r) == pytest.approx(0.0)


def test_sharpe_matches_manual_computation():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    expected = r.mean() / r.std() * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected)


def test_sharpe_zero_vol_returns_zero():
    r = pd.Series([0.01] * 10)
    assert sharpe(r) == 0.0


def test_sortino_no_downside_is_inf_when_positive():
    r = pd.Series([0.01, 0.02, 0.005])
    assert sortino(r) == float("inf")


def test_calmar_sign():
    r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.01] * 20)
    mdd = max_drawdown(r)
    assert calmar(r) == pytest.approx(cagr(r) / abs(mdd))


def test_summary_has_all_fields():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 300))
    s = summary(r)
    for key in ["total_return", "cagr", "annual_vol", "sharpe", "sortino",
                "max_drawdown", "calmar"]:
        assert key in s.index
