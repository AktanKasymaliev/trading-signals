"""Stationary feature builder for Path F.

Replaces absolute price levels (close_*, raw ema*_h1) with normalised
ratios and distances expressed in ATR units, plus multi-horizon returns.
All outputs are stationary across regime/price-level shifts.

Public API:
  - STATIONARY_FEATURES: ordered column list
  - build_stationary_features(tfs) -> (DataFrame, complete: bool)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


STATIONARY_FEATURES: list[str] = [
    "close_vs_ema8_atr",
    "close_vs_ema21_atr",
    "close_vs_ema50_atr",
    "close_vs_ema200_atr",
    "ema8_vs_ema21_atr",
    "ema21_vs_ema50_atr",
    "ema50_vs_ema200_atr",
    "return_m15_1",
    "return_m15_3",
    "return_m15_5",
    "return_h1_1",
    "return_h1_3",
    "return_h4_1",
    "atr_percentile_h1",
    "range_vs_atr_m15",
    "distance_to_recent_high_atr",
    "distance_to_recent_low_atr",
]


def _atr(df: pd.DataFrame | None, period: int = 14) -> float:
    """Wilder-style ATR (rolling-mean variant) on the last `period` bars.
    Returns 0.0 if data is too short or NaN."""
    if df is None or df.empty or len(df) < period + 1:
        return 0.0
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.0


def _ema_last(close: pd.Series, span: int) -> float:
    return float(close.ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])


def _safe_div(num: float, den: float) -> float:
    if den == 0 or pd.isna(den) or pd.isna(num):
        return 0.0
    return float(num / den)


def _return(df: pd.DataFrame | None, bars: int) -> float:
    if df is None or df.empty or len(df) <= bars:
        return 0.0
    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-1 - bars])
    return _safe_div(cur - prev, prev)


def _atr_percentile(h1: pd.DataFrame, period: int = 14,
                    lookback: int = 100) -> float:
    """Rank of latest ATR among the last `lookback` ATR values (0..1)."""
    if h1 is None or len(h1) < period + lookback:
        return 0.0
    high = h1["High"].astype(float)
    low = h1["Low"].astype(float)
    close = h1["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean().dropna()
    if len(atr_series) < lookback:
        return 0.0
    window = atr_series.tail(lookback)
    rank = window.rank(pct=True).iloc[-1]
    return float(rank) if not pd.isna(rank) else 0.0


def _zero_row() -> pd.DataFrame:
    return pd.DataFrame([[0.0] * len(STATIONARY_FEATURES)],
                        columns=STATIONARY_FEATURES)


def build_stationary_features(
    tfs: dict[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, bool]:
    """Build the 17-column stationary feature row from a multi-TF dict.

    Returns `(df, complete)`. `complete` is True iff enough history was
    available to compute every feature without falling back to a default.
    On the False branch we still return a one-row DataFrame with the
    correct columns (zero-filled where data was missing) so downstream
    code can rely on a stable shape.
    """
    if not tfs or "H1" not in tfs or tfs["H1"] is None or tfs["H1"].empty:
        return _zero_row(), False

    h1 = tfs["H1"].copy(deep=True)
    m15 = tfs.get("M15")
    h4 = tfs.get("H4")

    if len(h1) < 210:
        return _zero_row(), False

    close_h1 = h1["Close"].astype(float)
    ema8 = _ema_last(close_h1, 8)
    ema21 = _ema_last(close_h1, 21)
    ema50 = _ema_last(close_h1, 50)
    ema200 = _ema_last(close_h1, 200)
    close_now = float(close_h1.iloc[-1])
    atr_h1 = _atr(h1, period=14)
    if atr_h1 == 0.0:
        return _zero_row(), False

    recent_high = float(h1["High"].tail(20).max())
    recent_low = float(h1["Low"].tail(20).min())

    range_m15 = 0.0
    if m15 is not None and not m15.empty:
        range_m15 = float(m15["High"].iloc[-1] - m15["Low"].iloc[-1])

    row = {
        "close_vs_ema8_atr":   _safe_div(close_now - ema8, atr_h1),
        "close_vs_ema21_atr":  _safe_div(close_now - ema21, atr_h1),
        "close_vs_ema50_atr":  _safe_div(close_now - ema50, atr_h1),
        "close_vs_ema200_atr": _safe_div(close_now - ema200, atr_h1),
        "ema8_vs_ema21_atr":   _safe_div(ema8 - ema21, atr_h1),
        "ema21_vs_ema50_atr":  _safe_div(ema21 - ema50, atr_h1),
        "ema50_vs_ema200_atr": _safe_div(ema50 - ema200, atr_h1),
        "return_m15_1":  _return(m15, 1),
        "return_m15_3":  _return(m15, 3),
        "return_m15_5":  _return(m15, 5),
        "return_h1_1":   _return(h1, 1),
        "return_h1_3":   _return(h1, 3),
        "return_h4_1":   _return(h4, 1),
        "atr_percentile_h1": _atr_percentile(h1),
        "range_vs_atr_m15":  _safe_div(range_m15, atr_h1),
        "distance_to_recent_high_atr": _safe_div(close_now - recent_high, atr_h1),
        "distance_to_recent_low_atr":  _safe_div(close_now - recent_low, atr_h1),
    }
    df = pd.DataFrame([[row[name] for name in STATIONARY_FEATURES]],
                      columns=STATIONARY_FEATURES)
    complete = bool(np.isfinite(df.values).all())
    return df, complete
