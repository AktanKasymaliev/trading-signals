"""Support/Resistance helpers and swing utilities."""

from __future__ import annotations

import pandas as pd


def swing_highs_lows(df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    sh: list[float] = []
    sl: list[float] = []
    for i in range(window, len(df) - window):
        if highs[i] == highs[i - window:i + window + 1].max():
            sh.append(float(highs[i]))
        if lows[i] == lows[i - window:i + window + 1].min():
            sl.append(float(lows[i]))
    return sh, sl


def nearest_above(price: float, levels: list[float]) -> float | None:
    above = [lv for lv in levels if lv > price]
    return min(above) if above else None


def nearest_below(price: float, levels: list[float]) -> float | None:
    below = [lv for lv in levels if lv < price]
    return max(below) if below else None
