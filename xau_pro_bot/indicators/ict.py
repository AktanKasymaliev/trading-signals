"""ICT concepts: OTE, FVG, Order Blocks, Liquidity, Killzones."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from xau_pro_bot import config


_NY = ZoneInfo(config.TIMEZONE)


def _neutral_ote() -> dict[str, Any]:
    return {
        "in_ote": False, "ote_low": None, "ote_high": None,
        "swing_high": None, "swing_low": None, "direction": None,
    }


def find_ote(df: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    if len(df) < lookback + 1:
        return _neutral_ote()
    window = df.tail(lookback)
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    if swing_high == swing_low:
        return _neutral_ote()
    last_close = float(df["Close"].iloc[-1])
    mid = (swing_high + swing_low) / 2
    if last_close >= mid:
        direction = "bull"
        ote_low = swing_low + 0.62 * (swing_high - swing_low)
        ote_high = swing_low + 0.79 * (swing_high - swing_low)
    else:
        direction = "bear"
        ote_low = swing_high - 0.79 * (swing_high - swing_low)
        ote_high = swing_high - 0.62 * (swing_high - swing_low)
    in_ote = ote_low <= last_close <= ote_high
    return {
        "in_ote": bool(in_ote),
        "ote_low": float(ote_low),
        "ote_high": float(ote_high),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "direction": direction,
    }


def find_fvg(df: pd.DataFrame, max_gaps: int = 5) -> list[dict[str, Any]]:
    """Return unfilled FVGs (most recent first), max `max_gaps`."""
    if len(df) < 3:
        return []
    gaps: list[dict[str, Any]] = []
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    last_idx = len(df) - 1
    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            top = float(lows[i])
            bottom = float(highs[i - 2])
            future_lows = lows[i + 1:]
            if len(future_lows) == 0 or future_lows.min() > bottom:
                gaps.append({
                    "type": "bull", "top": top, "bottom": bottom,
                    "midpoint": (top + bottom) / 2, "age_bars": last_idx - i,
                })
        if highs[i] < lows[i - 2]:
            top = float(lows[i - 2])
            bottom = float(highs[i])
            future_highs = highs[i + 1:]
            if len(future_highs) == 0 or future_highs.max() < top:
                gaps.append({
                    "type": "bear", "top": top, "bottom": bottom,
                    "midpoint": (top + bottom) / 2, "age_bars": last_idx - i,
                })
    gaps.sort(key=lambda g: g["age_bars"])
    return gaps[:max_gaps]


def find_order_blocks(df: pd.DataFrame, lookback: int = 50) -> list[dict[str, Any]]:
    if len(df) < lookback + 2:
        return []
    obs: list[dict[str, Any]] = []
    window = df.tail(lookback).reset_index(drop=False)
    opens = window["Open"].to_numpy()
    closes = window["Close"].to_numpy()
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()

    for i in range(1, len(window)):
        if closes[i] > opens[i] * 1.003:
            for j in range(i - 1, -1, -1):
                if closes[j] < opens[j]:
                    tested = bool(lows[j + 1:].min() <= highs[j]) if i > j + 1 else False
                    obs.append({
                        "type": "bull",
                        "high": float(highs[j]),
                        "low": float(lows[j]),
                        "mid": float((highs[j] + lows[j]) / 2),
                        "index": int(j),
                        "tested": tested,
                    })
                    break
        if closes[i] < opens[i] * 0.997:
            for j in range(i - 1, -1, -1):
                if closes[j] > opens[j]:
                    tested = bool(highs[j + 1:].max() >= lows[j]) if i > j + 1 else False
                    obs.append({
                        "type": "bear",
                        "high": float(highs[j]),
                        "low": float(lows[j]),
                        "mid": float((highs[j] + lows[j]) / 2),
                        "index": int(j),
                        "tested": tested,
                    })
                    break
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for ob in obs[::-1]:
        if ob["index"] in seen:
            continue
        seen.add(ob["index"])
        unique.append(ob)
    return unique[:10]


def find_liquidity(df: pd.DataFrame, tolerance: float = 0.002,
                   lookback: int = 30) -> dict[str, list[float]]:
    if len(df) < lookback:
        return {"buy_side": [], "sell_side": []}
    window = df.tail(lookback)
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()

    def cluster(values, ref) -> list[float]:
        clusters: list[float] = []
        used = [False] * len(values)
        for i, v in enumerate(values):
            if used[i]:
                continue
            group = [v]
            used[i] = True
            for j in range(i + 1, len(values)):
                if used[j]:
                    continue
                if abs(values[j] - v) / max(abs(ref), 1) <= tolerance:
                    group.append(values[j])
                    used[j] = True
            if len(group) >= 2:
                clusters.append(float(sum(group) / len(group)))
        return clusters

    ref = float(df["Close"].iloc[-1])
    return {
        "buy_side":  cluster(list(highs), ref),
        "sell_side": cluster(list(lows), ref),
    }


def get_killzone(now: datetime | None = None) -> str | None:
    if now is None:
        now = datetime.now(_NY)
    if now.tzinfo is None:
        raise ValueError("get_killzone requires timezone-aware datetime")
    now_ny = now.astimezone(_NY).time()
    for name, (start, end) in config.KILLZONES_NY.items():
        if start <= now_ny <= end:
            return name
    return None


if __name__ == "__main__":
    print("killzone now:", get_killzone())
