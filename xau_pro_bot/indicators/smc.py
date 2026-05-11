"""Smart Money Concepts: BOS, CHOCH, Premium/Discount, Stop Hunt."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _swings(df: pd.DataFrame, swing_len: int) -> tuple[list[float], list[float]]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(swing_len, len(df) - swing_len):
        window_h = highs[i - swing_len:i + swing_len + 1]
        window_l = lows[i - swing_len:i + swing_len + 1]
        if highs[i] == window_h.max():
            swing_highs.append(float(highs[i]))
        if lows[i] == window_l.min():
            swing_lows.append(float(lows[i]))
    return swing_highs, swing_lows


def detect_structure(df: pd.DataFrame, swing_len: int = 5) -> dict[str, Any]:
    if len(df) < swing_len * 4:
        return {"last_event": None, "prev_swing_high": None, "prev_swing_low": None}

    swing_highs, swing_lows = _swings(df, swing_len)
    if not swing_highs or not swing_lows:
        return {"last_event": None, "prev_swing_high": None, "prev_swing_low": None}

    last_close = float(df["Close"].iloc[-1])
    prev_swing_high = swing_highs[-1]
    prev_swing_low = swing_lows[-1]

    recent = df["Close"].iloc[-20:].mean()
    earlier = df["Close"].iloc[-40:-20].mean() if len(df) >= 40 else recent
    in_uptrend = recent > earlier

    event: str | None = None
    if last_close > prev_swing_high:
        event = "BOS_bull" if in_uptrend else "CHOCH_bull"
    elif last_close < prev_swing_low:
        event = "BOS_bear" if not in_uptrend else "CHOCH_bear"

    return {
        "last_event": event,
        "prev_swing_high": prev_swing_high,
        "prev_swing_low": prev_swing_low,
    }


def premium_discount(df: pd.DataFrame, lookback: int = 50) -> dict[str, Any]:
    if len(df) < lookback:
        return {"zone": "neutral", "pct_of_range": 50.0,
                "equilibrium": None, "range_high": None, "range_low": None}
    window = df.tail(lookback)
    range_high = float(window["High"].max())
    range_low = float(window["Low"].min())
    equilibrium = (range_high + range_low) / 2
    last_price = float(df["Close"].iloc[-1])
    span = max(range_high - range_low, 1e-9)
    pct = (last_price - range_low) / span * 100
    if last_price > equilibrium:
        zone = "premium"
    elif last_price < equilibrium:
        zone = "discount"
    else:
        zone = "neutral"
    return {
        "zone": zone, "pct_of_range": float(pct),
        "equilibrium": float(equilibrium),
        "range_high": range_high, "range_low": range_low,
    }


def detect_stop_hunt(df: pd.DataFrame, atr: float) -> dict[str, Any]:
    """Detect stop hunt: wick > 2*body AND wick > 0.5*atr."""
    if len(df) < 5 or atr is None or atr <= 0 or np.isnan(atr):
        return {"bull_hunt": False, "bear_hunt": False, "level_hunted": None}

    last3 = df.tail(3)
    if len(df) >= 23:
        swing_low = float(df["Low"].iloc[:-3].tail(20).min())
        swing_high = float(df["High"].iloc[:-3].tail(20).max())
    else:
        swing_low = float(df["Low"].min())
        swing_high = float(df["High"].max())

    bull_hunt = False
    bear_hunt = False
    level_hunted: float | None = None

    for _, row in last3.iterrows():
        body = abs(row["Close"] - row["Open"])
        lower_wick = min(row["Open"], row["Close"]) - row["Low"]
        upper_wick = row["High"] - max(row["Open"], row["Close"])

        if (lower_wick > 2 * body and lower_wick > 0.5 * atr
                and row["Low"] < swing_low and row["Close"] > swing_low):
            bull_hunt = True
            level_hunted = swing_low

        if (upper_wick > 2 * body and upper_wick > 0.5 * atr
                and row["High"] > swing_high and row["Close"] < swing_high):
            bear_hunt = True
            level_hunted = swing_high

    return {"bull_hunt": bull_hunt, "bear_hunt": bear_hunt,
            "level_hunted": level_hunted}


if __name__ == "__main__":
    pass
