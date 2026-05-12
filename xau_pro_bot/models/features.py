"""Deterministic feature preparation for optional AI inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot.indicators import classic
from xau_pro_bot.indicators.smc import premium_discount
from xau_pro_bot.indicators.wyckoff import detect_wyckoff


REQUIRED_AI_FEATURES = [
    "close_m15",
    "close_h1",
    "close_h4",
    "close_d1",
    "return_m15_1",
    "return_m15_3",
    "return_m15_5",
    "return_h1_1",
    "return_h1_3",
    "return_h4_1",
    "atr_h1",
    "atr_m15",
    "rsi_h1",
    "rsi_m15",
    "ema8_h1",
    "ema21_h1",
    "ema50_h1",
    "ema200_h1",
    "ema8_above_ema21_h1",
    "ema21_above_ema50_h1",
    "ema50_above_ema200_h1",
    "price_above_ema50_h1",
    "price_above_ema200_h1",
    "h1_range_pct",
    "m15_range_pct",
    "pd_zone_h4_encoded",
    "wyckoff_bias_h4_encoded",
    "hour_utc",
    "day_of_week",
]


def _copy_df(tfs: dict[str, pd.DataFrame], tf: str) -> pd.DataFrame:
    df = tfs.get(tf)
    if df is None:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return df.copy(deep=True)


def _with_classic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy(deep=True)
    try:
        return classic.add_classic(df.copy(deep=True))
    except Exception:
        return df.copy(deep=True)


def _last_float(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    if df.empty or col not in df.columns:
        return default
    value = df[col].iloc[-1]
    return float(value) if not pd.isna(value) else default


def _return(df: pd.DataFrame, bars: int) -> float:
    if df.empty or "Close" not in df.columns or len(df) <= bars:
        return 0.0
    current = float(df["Close"].iloc[-1])
    previous = float(df["Close"].iloc[-1 - bars])
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return 0.0
    return (current - previous) / previous


def _above(left: float, right: float) -> int:
    if pd.isna(left) or pd.isna(right):
        return 0
    return 1 if left > right else -1


def _range_pct(df: pd.DataFrame) -> float:
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    high = _last_float(df, "High")
    low = _last_float(df, "Low")
    close = _last_float(df, "Close")
    if pd.isna(high) or pd.isna(low) or pd.isna(close) or close == 0:
        return 0.0
    return (high - low) / close


def _zone_code(zone: str | None) -> int:
    if zone == "discount":
        return 1
    if zone == "premium":
        return -1
    return 0


def _bias_code(bias: str | None) -> int:
    if bias == "bull":
        return 1
    if bias == "bear":
        return -1
    return 0


def _timestamp_features(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty or df.index.empty:
        return 0, 0
    ts = pd.Timestamp(df.index[-1])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    return int(ts.hour), int(ts.dayofweek)


def build_ai_features(tfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build exactly one deterministic feature row for AI inference."""
    m15 = _with_classic(_copy_df(tfs, "M15"))
    h1 = _with_classic(_copy_df(tfs, "H1"))
    h4 = _with_classic(_copy_df(tfs, "H4"))
    d1 = _with_classic(_copy_df(tfs, "D1"))

    pd_code = 0
    try:
        pd_code = _zone_code(premium_discount(h4, lookback=50).get("zone"))
    except Exception:
        pd_code = 0

    wy_code = 0
    try:
        wy_code = _bias_code(detect_wyckoff(h4).get("bias"))
    except Exception:
        wy_code = 0

    hour, day = _timestamp_features(m15 if not m15.empty else h1)

    close_h1 = _last_float(h1, "Close")
    ema8_h1 = _last_float(h1, "EMA_8")
    ema21_h1 = _last_float(h1, "EMA_21")
    ema50_h1 = _last_float(h1, "EMA_50")
    ema200_h1 = _last_float(h1, "EMA_200")

    row: dict[str, Any] = {
        "close_m15": _last_float(m15, "Close"),
        "close_h1": close_h1,
        "close_h4": _last_float(h4, "Close"),
        "close_d1": _last_float(d1, "Close"),
        "return_m15_1": _return(m15, 1),
        "return_m15_3": _return(m15, 3),
        "return_m15_5": _return(m15, 5),
        "return_h1_1": _return(h1, 1),
        "return_h1_3": _return(h1, 3),
        "return_h4_1": _return(h4, 1),
        "atr_h1": _last_float(h1, "ATR_14"),
        "atr_m15": _last_float(m15, "ATR_14"),
        "rsi_h1": _last_float(h1, "RSI_14"),
        "rsi_m15": _last_float(m15, "RSI_14"),
        "ema8_h1": ema8_h1,
        "ema21_h1": ema21_h1,
        "ema50_h1": ema50_h1,
        "ema200_h1": ema200_h1,
        "ema8_above_ema21_h1": _above(ema8_h1, ema21_h1),
        "ema21_above_ema50_h1": _above(ema21_h1, ema50_h1),
        "ema50_above_ema200_h1": _above(ema50_h1, ema200_h1),
        "price_above_ema50_h1": _above(close_h1, ema50_h1),
        "price_above_ema200_h1": _above(close_h1, ema200_h1),
        "h1_range_pct": _range_pct(h1),
        "m15_range_pct": _range_pct(m15),
        "pd_zone_h4_encoded": pd_code,
        "wyckoff_bias_h4_encoded": wy_code,
        "hour_utc": hour,
        "day_of_week": day,
    }
    return pd.DataFrame(
        [[row[name] for name in REQUIRED_AI_FEATURES]],
        columns=REQUIRED_AI_FEATURES,
    )
