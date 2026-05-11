"""M15 scalp setup: EMA cross + RSI extreme + BB touch + volume."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.indicators import classic
from xau_pro_bot.indicators.ict import get_killzone


def _h4_trend_bias(h4_df: pd.DataFrame) -> str | None:
    enriched = classic.add_classic(h4_df)
    last = enriched.iloc[-1]
    e50, e200 = last.get("EMA_50", np.nan), last.get("EMA_200", np.nan)
    if pd.isna(e50) or pd.isna(e200):
        return None
    return "bull" if e50 > e200 else "bear"


def scalp_signal(m15_df: pd.DataFrame, h1_df: pd.DataFrame,
                 h4_df: pd.DataFrame) -> dict[str, Any] | None:
    kz = get_killzone()
    if kz not in config.PRIORITY_KILLZONES:
        return None

    if len(m15_df) < 50:
        return None

    enriched = classic.add_classic(m15_df)
    if len(enriched) < 3:
        return None

    last = enriched.iloc[-1]
    prev = enriched.iloc[-2]
    pprev = enriched.iloc[-3]

    direction: str | None = None
    conditions_met: list[str] = []

    cross_bull = (pprev["EMA_8"] < pprev["EMA_21"]
                  and last["EMA_8"] > last["EMA_21"])
    cross_bear = (pprev["EMA_8"] > pprev["EMA_21"]
                  and last["EMA_8"] < last["EMA_21"])
    if cross_bull:
        direction = "BUY"; conditions_met.append("EMA cross bull")
    elif cross_bear:
        direction = "SELL"; conditions_met.append("EMA cross bear")
    if direction is None:
        return None

    rsi = last["RSI_14"]
    if direction == "BUY" and not pd.isna(rsi) and rsi < 35:
        conditions_met.append(f"RSI {rsi:.0f} OS")
    elif direction == "SELL" and not pd.isna(rsi) and rsi > 65:
        conditions_met.append(f"RSI {rsi:.0f} OB")

    close = float(last["Close"])
    bbl = last["BBL_20_2.0"]
    bbu = last["BBU_20_2.0"]
    if direction == "BUY" and not pd.isna(bbl) and close <= bbl * 1.001:
        conditions_met.append("BB lower")
    elif direction == "SELL" and not pd.isna(bbu) and close >= bbu * 0.999:
        conditions_met.append("BB upper")

    vol_ratio = last.get("vol_ratio", np.nan)
    if not pd.isna(vol_ratio) and vol_ratio > 1.3:
        conditions_met.append(f"Vol {vol_ratio:.1f}x")

    if len(conditions_met) < 3:
        return None

    atr_m15 = last["ATR_14"]
    if pd.isna(atr_m15) or atr_m15 <= 0:
        return None

    h4_trend = _h4_trend_bias(h4_df)
    counter_trend = ((h4_trend == "bear" and direction == "BUY")
                     or (h4_trend == "bull" and direction == "SELL"))

    if direction == "BUY":
        sl = close - atr_m15 * 1.0
        tp1 = close + atr_m15 * 1.5
        tp2 = close + atr_m15 * 2.5
    else:
        sl = close + atr_m15 * 1.0
        tp1 = close - atr_m15 * 1.5
        tp2 = close - atr_m15 * 2.5

    return {
        "active": True,
        "direction": direction,
        "entry": round(close, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "conditions_met": conditions_met,
        "counter_trend": counter_trend,
        "killzone": kz,
        "atr_m15": float(atr_m15),
    }
