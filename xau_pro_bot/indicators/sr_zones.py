"""S/R zones: historical key levels, psychological round levels, and zone scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot.indicators import classic
from xau_pro_bot.indicators.sr_levels import swing_highs_lows


_TOUCH_TOLERANCE = 0.003  # 0.3%


def find_psychological_levels(price: float, span: float = 200.0) -> list[float]:
    """Return psychological round levels within +/- span of price (every $50)."""
    low = price - span
    high = price + span
    levels: set[float] = set()
    base = int(low // 50) * 50
    while base <= high:
        levels.add(float(base))
        base += 50
    return sorted(levels)


def _count_touches(prices: np.ndarray, level: float) -> int:
    diff = np.abs(prices - level) / max(abs(level), 1.0)
    return int(np.sum(diff <= _TOUCH_TOLERANCE))


def _zone_strength(touches: int, recency_pct: float, tf_bonus: int) -> int:
    raw = min(touches, 5) * 15 + int(recency_pct * 10) + tf_bonus
    return max(0, min(100, raw))


def _build_zone(level: float, touches: int, atr_h4: float,
                tf_bonus: int, recency_pct: float, kind: str) -> dict[str, Any]:
    width = max(atr_h4 * 0.5, 0.5)
    return {
        "level": float(level),
        "zone_top": float(level + width),
        "zone_bot": float(level - width),
        "touches": touches,
        "strength": _zone_strength(touches, recency_pct, tf_bonus),
        "type": kind,
    }


def find_sr_zones(h4_df: pd.DataFrame, d1_df: pd.DataFrame,
                  current_price: float) -> dict[str, Any]:
    atr_h4 = 1.0
    enriched_h4 = classic.add_classic(h4_df)
    last_atr = enriched_h4["ATR_14"].iloc[-1] if "ATR_14" in enriched_h4 else np.nan
    if not pd.isna(last_atr):
        atr_h4 = float(last_atr)

    candidates: list[dict[str, Any]] = []

    d1_window = d1_df.tail(365)
    if len(d1_window) >= 20:
        sh, sl = swing_highs_lows(d1_window, window=5)
        all_prices = np.concatenate(
            [d1_window["High"].to_numpy(), d1_window["Low"].to_numpy()])
        unique_swings = list({round(x, 2) for x in (sh + sl)})
        close_np = d1_window["Close"].to_numpy()
        for lvl in unique_swings:
            touches = _count_touches(all_prices, lvl)
            if touches >= 2:
                idxs = [
                    i for i, p in enumerate(close_np)
                    if abs(p - lvl) / max(abs(lvl), 1) <= _TOUCH_TOLERANCE
                ]
                last_touch_idx = max(idxs) if idxs else 0
                recency_pct = last_touch_idx / max(len(close_np) - 1, 1)
                kind = "MAJOR" if touches >= 3 else "MINOR"
                candidates.append(_build_zone(
                    lvl, touches, atr_h4, tf_bonus=20,
                    recency_pct=recency_pct, kind=kind))

    for lvl in find_psychological_levels(current_price, span=200.0):
        candidates.append(_build_zone(
            level=lvl, touches=1, atr_h4=atr_h4,
            tf_bonus=8, recency_pct=0.5, kind="PSYCHOLOGICAL"))

    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for z in candidates:
        if z["level"] > current_price:
            resistance.append(z)
        elif z["level"] < current_price:
            support.append(z)

    resistance.sort(key=lambda z: z["level"])
    support.sort(key=lambda z: z["level"], reverse=True)

    at_resistance = any(z["zone_bot"] <= current_price <= z["zone_top"]
                        for z in resistance[:3])
    at_support = any(z["zone_bot"] <= current_price <= z["zone_top"]
                     for z in support[:3])

    return {
        "resistance_zones": resistance[:6],
        "support_zones": support[:6],
        "at_resistance": at_resistance,
        "at_support": at_support,
        "nearest_resistance": resistance[0]["level"] if resistance else None,
        "nearest_support": support[0]["level"] if support else None,
        "atr_h4": atr_h4,
    }
