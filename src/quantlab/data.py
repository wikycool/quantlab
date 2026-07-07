"""Data loading and synthetic data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def synthetic_prices(
    n_periods: int = 504,
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    seed: int = 7,
    annual_drift: float = 0.06,
    annual_vol: float = 0.20,
    start: str = "2020-01-01",
) -> dict[str, pd.DataFrame]:
    """Generate correlated GBM OHLCV bars for testing and examples.

    Returns a dict mapping symbol -> DataFrame with columns
    open/high/low/close/volume, indexed by business day.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(symbols)
    dt = 1.0 / 252.0

    # Mild common factor so cross-sectional strategies have structure to find.
    common = rng.standard_normal(n_periods)
    idio = rng.standard_normal((n_periods, n_assets))
    shocks = 0.4 * common[:, None] + np.sqrt(1 - 0.4**2) * idio

    drifts = annual_drift + rng.uniform(-0.04, 0.04, size=n_assets)
    vols = annual_vol + rng.uniform(-0.05, 0.05, size=n_assets)

    log_returns = (drifts - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * shocks
    closes = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    index = pd.bdate_range(start=start, periods=n_periods)
    out: dict[str, pd.DataFrame] = {}
    for j, sym in enumerate(symbols):
        close = closes[:, j]
        open_ = np.empty_like(close)
        open_[0] = 100.0
        # Next open gaps slightly from previous close.
        gaps = 1.0 + rng.normal(0, 0.001, size=n_periods - 1)
        open_[1:] = close[:-1] * gaps
        intrabar = np.abs(rng.normal(0, 0.004, size=n_periods))
        high = np.maximum(open_, close) * (1 + intrabar)
        low = np.minimum(open_, close) * (1 - intrabar)
        volume = rng.integers(50_000, 500_000, size=n_periods).astype(float)
        out[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
    return out


def load_csv(path: str) -> pd.DataFrame:
    """Load a single-symbol OHLCV CSV with a date column or date index.

    Column names are lowercased; a 'date' column, if present, becomes the index.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV at {path} is missing columns: {missing}")
    return df[OHLCV_COLUMNS].sort_index()
