"""Deterministic OHLCV fixtures for offline unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def long_history():
    rng = np.random.default_rng(7)
    n = 4000
    base = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2,
        "Close": base + rng.normal(0, 0.3, n),
        "Volume": rng.integers(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def _make_df(closes: list[float], start: datetime, freq: str = "h",
             volume: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    closes_arr = np.array(closes, dtype=float)
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    highs = np.maximum(opens, closes_arr) + 0.5
    lows = np.minimum(opens, closes_arr) - 0.5
    if volume is None:
        volume = [1000.0] * n
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes_arr,
            "Volume": np.array(volume, dtype=float),
        },
        index=idx,
    )


@pytest.fixture
def uptrend_df() -> pd.DataFrame:
    """100 bars trending up 2000 → 2200."""
    closes = list(np.linspace(2000.0, 2200.0, 100))
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def downtrend_df() -> pd.DataFrame:
    closes = list(np.linspace(2200.0, 2000.0, 100))
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def flat_df() -> pd.DataFrame:
    closes = [2100.0] * 100
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def short_df() -> pd.DataFrame:
    """Only 10 bars — many indicators should return neutral."""
    closes = list(np.linspace(2000.0, 2050.0, 10))
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def df_with_fvg() -> pd.DataFrame:
    """100 bars with a clear bullish FVG injected at index 50."""
    closes = list(np.linspace(2000.0, 2100.0, 100))
    df = _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))
    df.iloc[48, df.columns.get_loc("High")] = 2040.0
    df.iloc[49, df.columns.get_loc("High")] = 2060.0
    df.iloc[49, df.columns.get_loc("Low")] = 2050.0
    df.iloc[50, df.columns.get_loc("Low")] = 2055.0
    return df


@pytest.fixture
def df_with_volume_none() -> pd.DataFrame:
    """Mimics Twelve Data spot response with NaN Volume."""
    closes = list(np.linspace(2000.0, 2100.0, 100))
    df = _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))
    df["Volume"] = np.nan
    return df


@pytest.fixture
def all_tfs(uptrend_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mock fetch_all_timeframes return value."""
    return {tf: uptrend_df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}
