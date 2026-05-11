"""Classic TA indicators: EMA, RSI, MACD, BB, ATR, Stoch, Volume, Pivots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.indicators import pandas_ta as ta  # patched/aliased package import


def add_classic(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich an OHLCV DataFrame with classic indicators (returns a copy)."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["EMA_8"] = ta.ema(close, length=8)
    out["EMA_21"] = ta.ema(close, length=21)
    out["EMA_50"] = ta.ema(close, length=50)
    out["EMA_200"] = ta.ema(close, length=200)

    out["RSI_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["MACD_12_26_9"] = macd.iloc[:, 0]
        out["MACDh_12_26_9"] = macd.iloc[:, 1]
        out["MACDs_12_26_9"] = macd.iloc[:, 2]
    else:
        for c in ("MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"):
            out[c] = np.nan

    stoch = ta.stoch(high, low, close, k=14, d=3)
    if stoch is not None and not stoch.empty:
        out["STOCHk_14_3_3"] = stoch.iloc[:, 0]
        out["STOCHd_14_3_3"] = stoch.iloc[:, 1]
    else:
        out["STOCHk_14_3_3"] = np.nan
        out["STOCHd_14_3_3"] = np.nan

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        out["BBL_20_2.0"] = bb.iloc[:, 0]
        out["BBM_20_2.0"] = bb.iloc[:, 1]
        out["BBU_20_2.0"] = bb.iloc[:, 2]
    else:
        for c in ("BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"):
            out[c] = np.nan

    out["ATR_14"] = ta.atr(high, low, close, length=14)

    if "Volume" in out and not out["Volume"].isna().all():
        vol_avg = out["Volume"].rolling(20).mean()
        out["vol_ratio"] = out["Volume"] / vol_avg
    else:
        out["vol_ratio"] = np.nan

    prev = out.shift(1)
    out["pivot"] = (prev["High"] + prev["Low"] + prev["Close"]) / 3
    out["r1"] = 2 * out["pivot"] - prev["Low"]
    out["s1"] = 2 * out["pivot"] - prev["High"]
    out["r2"] = out["pivot"] + (prev["High"] - prev["Low"])
    out["s2"] = out["pivot"] - (prev["High"] - prev["Low"])

    return out


if __name__ == "__main__":
    from datetime import datetime, timezone
    n = 250
    df = pd.DataFrame({
        "Open": np.linspace(2000, 2100, n),
        "High": np.linspace(2005, 2105, n),
        "Low": np.linspace(1995, 2095, n),
        "Close": np.linspace(2000, 2100, n),
        "Volume": [1000.0] * n,
    }, index=pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc),
                           periods=n, freq="h"))
    enriched = add_classic(df)
    print(enriched.tail(3).T)
