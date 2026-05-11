"""Wyckoff phase detection — soft bias only (max ±5 in engine)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config


def detect_wyckoff(df: pd.DataFrame) -> dict[str, Any]:
    n = config.WYCKOFF_BARS
    if len(df) < n:
        return {"phase": "neutral", "bias": "neutral", "strength": 0}

    window = df.tail(n)
    closes = window["Close"].to_numpy()
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()

    tr_high = float(highs.max())
    tr_low = float(lows.min())
    span = max(tr_high - tr_low, 1e-9)
    last_price = float(closes[-1])
    pos = (last_price - tr_low) / span  # 0..1

    x = np.arange(n)
    slope = np.polyfit(x, closes, 1)[0]
    slope_norm = slope * n / span

    phase = "neutral"
    bias = "neutral"

    if slope_norm > 0.6 and pos > 0.7:
        phase, bias = "markup", "bull"
    elif slope_norm < -0.6 and pos < 0.3:
        phase, bias = "markdown", "bear"
    elif pos < 0.3 and abs(slope_norm) < 0.5:
        phase, bias = "accumulation", "bull"
    elif pos > 0.7 and abs(slope_norm) < 0.5:
        phase, bias = "distribution", "bear"

    strength = int(min(100, abs(slope_norm) * 50 + abs(pos - 0.5) * 100))
    return {"phase": phase, "bias": bias, "strength": strength}


if __name__ == "__main__":
    pass
