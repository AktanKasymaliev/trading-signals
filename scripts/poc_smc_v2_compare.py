"""PoC: head-to-head compare-ai using M15-base history.

Bypasses the H1-only CLI: builds history from raw M15 CSV, resamples to all
higher TFs (H1, H4, D1, W1), then calls compare_backtests directly.

Run:
    AI_ENABLED=true \\
    AI_MODEL_ID=JonusNattapong/xauusd-trading-ai-smc-v2 \\
    AI_MODEL_REVISION=d1ee87d058bf714af1b6f4b3979646dd0024b726 \\
    AI_MODEL_FILENAME=trading_model_15m.pkl \\
    AI_FEATURE_SET=smc_v2 \\
    AI_CACHE_DIR=./models_cache \\
    PYTHONPATH=. .venv/bin/python scripts/poc_smc_v2_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from xau_pro_bot.backtest import compare_backtests


AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}


def _resample(base: pd.DataFrame, rule: str) -> pd.DataFrame:
    return base.resample(rule).agg(AGG).dropna()


def _load_m15(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def main() -> int:
    m15 = _load_m15(Path("./data_xauusd_m15.csv"))
    print(f"M15 bars: {len(m15)} | range: {m15.index.min()} → {m15.index.max()}")

    h1 = _resample(m15, "1h")
    h4 = _resample(m15, "4h")
    d1 = _resample(m15, "1D")
    w1 = _resample(m15, "1W")
    history = {"M15": m15, "H1": h1, "H4": h4, "D1": d1, "W1": w1}
    print(f"H1 bars: {len(h1)}, H4: {len(h4)}, D1: {len(d1)}, W1: {len(w1)}")

    comparison = compare_backtests(
        history=history,
        timeout_bars=48,
        step=4,
        stream="intraday",
    )
    b, a, s = comparison["baseline"], comparison["ai"], comparison["summary"]
    print()
    print("=== BASELINE (deterministic only) ===")
    print(f"Signals:      {b.signals_generated}")
    print(f"Wins/Loss:    {b.wins} / {b.losses} (timeouts {b.timeouts})")
    print(f"Win rate:     {b.win_rate:.1%}")
    print(f"Expectancy:   {b.expectancy:.2f} R")
    print(f"Profit f.:    {b.profit_factor:.2f}")
    print(f"Max DD:       {b.max_drawdown:.2f} R")
    print(f"Avg RR:       {b.average_rr:.2f}")
    print()
    print("=== BASELINE + AI (SMC v2) ===")
    print(f"Signals:      {a.signals_generated}")
    print(f"Blocked AI:   {a.blocked_signals}")
    print(f"Wins/Loss:    {a.wins} / {a.losses} (timeouts {a.timeouts})")
    print(f"Win rate:     {a.win_rate:.1%}")
    print(f"Expectancy:   {a.expectancy:.2f} R")
    print(f"Profit f.:    {a.profit_factor:.2f}")
    print(f"Max DD:       {a.max_drawdown:.2f} R")
    print(f"Avg RR:       {a.average_rr:.2f}")
    print()
    print("=== DELTA ===")
    print(f"Trade delta:       {s['trade_delta']}")
    print(f"Win-rate delta:    {s['win_rate_delta']:.1%}")
    print(f"Expectancy delta:  {s['expectancy_delta']:.2f} R")
    print(f"Profit factor Δ:   {s['profit_factor_delta']:.2f}")
    print(f"Blocked signals:   {s['blocked_signals']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
