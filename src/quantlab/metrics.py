"""Performance metrics computed from a series of periodic returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / n) - 1.0


def sharpe(returns: pd.Series, risk_free_rate: float = 0.0,
           periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio. risk_free_rate is annual."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    # Guard against exact-zero and floating-point-noise volatility alike.
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, risk_free_rate: float = 0.0,
            periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sortino ratio (downside deviation in the denominator)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev == 0:
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, returned as a negative number."""
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def calmar(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / abs(mdd)


def summary(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> pd.Series:
    """One-stop summary of the usual suspects."""
    return pd.Series(
        {
            "total_return": total_return(returns),
            "cagr": cagr(returns, periods_per_year),
            "annual_vol": float(returns.std() * np.sqrt(periods_per_year)),
            "sharpe": sharpe(returns, periods_per_year=periods_per_year),
            "sortino": sortino(returns, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(returns),
            "calmar": calmar(returns, periods_per_year),
        }
    )
