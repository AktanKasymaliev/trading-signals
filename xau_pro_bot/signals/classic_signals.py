"""Classic TA scoring contributions on H1 (already enriched by add_classic)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_classic(h1_df, m15_df) -> tuple[float, float, list[str]]:
    bull = bear = 0.0
    reasons: list[str] = []

    last = h1_df.iloc[-1]
    prev = h1_df.iloc[-2] if len(h1_df) >= 2 else last

    rsi = last.get("RSI_14", np.nan)
    if not pd.isna(rsi):
        if rsi < 30:
            bull += 8
            reasons.append(f"RSI H1 oversold ({rsi:.1f})")
        elif rsi > 70:
            bear += 8
            reasons.append(f"RSI H1 overbought ({rsi:.1f})")
        elif 40 <= rsi <= 60:
            bull -= 8
            bear -= 8

    macd = last.get("MACD_12_26_9", np.nan)
    macd_s = last.get("MACDs_12_26_9", np.nan)
    prev_macd = prev.get("MACD_12_26_9", np.nan)
    prev_macd_s = prev.get("MACDs_12_26_9", np.nan)
    if not any(pd.isna(x) for x in (macd, macd_s, prev_macd, prev_macd_s)):
        if prev_macd < prev_macd_s and macd > macd_s:
            bull += 6
            reasons.append("MACD H1 bull cross")
        elif prev_macd > prev_macd_s and macd < macd_s:
            bear += 6
            reasons.append("MACD H1 bear cross")

    k = last.get("STOCHk_14_3_3", np.nan)
    d = last.get("STOCHd_14_3_3", np.nan)
    pk = prev.get("STOCHk_14_3_3", np.nan)
    pd_val = prev.get("STOCHd_14_3_3", np.nan)
    if not any(pd.isna(x) for x in (k, d, pk, pd_val)):
        if pk < pd_val and k > d and k < 30:
            bull += 6
            reasons.append("Stoch H1 bull cross OS")
        elif pk > pd_val and k < d and k > 70:
            bear += 6
            reasons.append("Stoch H1 bear cross OB")

    bbl = last.get("BBL_20_2.0", np.nan)
    bbu = last.get("BBU_20_2.0", np.nan)
    close = float(last["Close"])
    if not pd.isna(bbl) and close <= bbl:
        bull += 5
        reasons.append("BB lower rejection")
    if not pd.isna(bbu) and close >= bbu:
        bear += 5
        reasons.append("BB upper rejection")

    vol_ratio = last.get("vol_ratio", np.nan)
    if not pd.isna(vol_ratio):
        if vol_ratio > 1.5:
            bull += 5
            bear += 5
            reasons.append(f"Volume {vol_ratio:.1f}x avg")
        elif vol_ratio < 0.6:
            bull -= 6
            bear -= 6

    if len(m15_df) >= 3 and "EMA_8" in m15_df.columns and "EMA_21" in m15_df.columns:
        a, b = m15_df.iloc[-1], m15_df.iloc[-2]
        if b["EMA_8"] < b["EMA_21"] and a["EMA_8"] > a["EMA_21"]:
            bull += 4
            reasons.append("M15 EMA8>EMA21 cross")
        elif b["EMA_8"] > b["EMA_21"] and a["EMA_8"] < a["EMA_21"]:
            bear += 4
            reasons.append("M15 EMA8<EMA21 cross")

    s1 = last.get("s1", np.nan)
    r1 = last.get("r1", np.nan)
    if not pd.isna(s1) and abs(close - s1) / max(close, 1) < 0.002:
        bull += 3
        reasons.append(f"Pivot S1 {s1:.2f}")
    if not pd.isna(r1) and abs(close - r1) / max(close, 1) < 0.002:
        bear += 3
        reasons.append(f"Pivot R1 {r1:.2f}")

    return bull, bear, reasons
