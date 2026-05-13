"""Feature builder matching JonusNattapong/xauusd-trading-ai-smc-v2 contract.

Verified against model.feature_names_in_ (21 features) and X_features.csv
encoding (alphabetical LabelEncoder: bearish=0, bullish=1, none=2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SMC_V2_FEATURES = [
    "Close", "High", "Low", "Open",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_middle", "BB_lower",
    "FVG_Size", "FVG_Type", "OB_Type",
    "Close_lag1", "Close_lag2", "Close_lag3",
]

_MIN_BARS_FOR_COMPLETE = 60  # SMA_50 + lookback for SMC features

# Label encoding from X_features.csv (alphabetical):
_FVG_ENCODE_NONE = 2
_FVG_ENCODE_BULLISH = 1
_FVG_ENCODE_BEARISH = 0
_OB_ENCODE_NONE = 2
_OB_ENCODE_BULLISH = 1
_OB_ENCODE_BEARISH = 0


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    middle = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = middle + mult * std
    lower = middle - mult * std
    return upper, middle, lower


def _last_fvg(df: pd.DataFrame, lookback: int = 50) -> tuple[float, int]:
    """Return (FVG_Size, FVG_Type_encoded).

    FVG (3-candle imbalance):
      Bullish at bar i: low[i] > high[i-2] → gap_size = low[i] - high[i-2]
      Bearish at bar i: high[i] < low[i-2] → gap_size = low[i-2] - high[i]
    """
    if len(df) < 3:
        return 0.0, _FVG_ENCODE_NONE
    window = df.iloc[-lookback:] if len(df) > lookback else df
    high = window["High"].values
    low = window["Low"].values
    for i in range(len(window) - 1, 1, -1):
        if low[i] > high[i - 2]:
            return float(low[i] - high[i - 2]), _FVG_ENCODE_BULLISH
        if high[i] < low[i - 2]:
            return float(low[i - 2] - high[i]), _FVG_ENCODE_BEARISH
    return 0.0, _FVG_ENCODE_NONE


def _last_ob_encoded(df: pd.DataFrame, lookback: int = 50) -> int:
    """Most recent Order Block direction.

    Heuristic: a candle whose body is opposite to a >=1.5*body move in the
    next 3 bars is treated as an OB in the move's direction.
    """
    if len(df) < 5:
        return _OB_ENCODE_NONE
    window = df.iloc[-lookback:] if len(df) > lookback else df
    o = window["Open"].values
    c = window["Close"].values
    for i in range(len(window) - 4, -1, -1):
        body = abs(c[i] - o[i])
        if body <= 0:
            continue
        move = c[i + 3] - c[i]
        if c[i] < o[i] and move > 1.5 * body:
            return _OB_ENCODE_BULLISH
        if c[i] > o[i] and move < -1.5 * body:
            return _OB_ENCODE_BEARISH
    return _OB_ENCODE_NONE


def build_smc_v2_features(
    tfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    """Build a 1-row DataFrame of the 21 SMC v2 features from M15 bars."""
    m15 = tfs.get("M15")
    if m15 is None or m15.empty:
        empty = pd.DataFrame(
            [[0.0] * len(REQUIRED_SMC_V2_FEATURES)],
            columns=REQUIRED_SMC_V2_FEATURES,
        )
        return empty, False

    df = m15.copy(deep=True)
    complete = len(df) >= _MIN_BARS_FOR_COMPLETE

    close = df["Close"]
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    rsi = _rsi(close, 14)
    macd, macd_signal, macd_hist = _macd(close)
    bb_upper, bb_middle, bb_lower = _bollinger(close)
    fvg_size, fvg_type = _last_fvg(df)
    ob_type = _last_ob_encoded(df)

    last_close = float(close.iloc[-1])
    lag1 = float(close.iloc[-2]) if len(close) >= 2 else last_close
    lag2 = float(close.iloc[-3]) if len(close) >= 3 else last_close
    lag3 = float(close.iloc[-4]) if len(close) >= 4 else last_close

    row: dict[str, Any] = {
        "Close": last_close,
        "High": float(df["High"].iloc[-1]),
        "Low": float(df["Low"].iloc[-1]),
        "Open": float(df["Open"].iloc[-1]),
        "SMA_20": float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else last_close,
        "SMA_50": float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else last_close,
        "EMA_12": float(ema_12.iloc[-1]),
        "EMA_26": float(ema_26.iloc[-1]),
        "RSI": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
        "MACD": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
        "MACD_signal": float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
        "MACD_hist": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0.0,
        "BB_upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else last_close,
        "BB_middle": float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else last_close,
        "BB_lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else last_close,
        "FVG_Size": fvg_size,
        "FVG_Type": fvg_type,
        "OB_Type": ob_type,
        "Close_lag1": lag1,
        "Close_lag2": lag2,
        "Close_lag3": lag3,
    }
    out = pd.DataFrame(
        [[row[name] for name in REQUIRED_SMC_V2_FEATURES]],
        columns=REQUIRED_SMC_V2_FEATURES,
    )
    return out.fillna(0.0), complete
