"""Swing setups: 1000-pip and 500-pip Fibonacci retracement entries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.indicators import classic


def _d1_trend(d1_df: pd.DataFrame) -> str | None:
    enriched = classic.add_classic(d1_df)
    last = enriched.iloc[-1]
    e50, e200 = last.get("EMA_50", np.nan), last.get("EMA_200", np.nan)
    if pd.isna(e50) or pd.isna(e200):
        return None
    return "bull" if e50 > e200 else "bear"


def find_swing_setup(d1_df: pd.DataFrame, h4_df: pd.DataFrame) -> dict[str, Any] | None:
    if len(d1_df) < 200:
        return None
    window = d1_df.tail(200)
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    full_range_usd = swing_high - swing_low
    range_pips = full_range_usd / config.XAU_PIP_VALUE
    if range_pips < 500:
        return None

    trend = _d1_trend(d1_df)
    if trend is None:
        return None
    direction = "BUY" if trend == "bull" else "SELL"

    if range_pips >= 1000:
        setup_type = "1000pip"
        fib = 0.20
        sl_buffer_pips = 50
    else:
        setup_type = "500pip"
        fib = 0.236
        sl_buffer_pips = 30

    sl_buffer = sl_buffer_pips * config.XAU_PIP_VALUE

    if direction == "BUY":
        entry = swing_high - fib * full_range_usd
        tp = swing_high
        sl = swing_low - sl_buffer
    else:
        entry = swing_low + fib * full_range_usd
        tp = swing_low
        sl = swing_high + sl_buffer

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < 2.0:
        return None

    return {
        "type": setup_type,
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "range_pips": round(range_pips, 1),
        "rr": round(rr, 2),
    }
