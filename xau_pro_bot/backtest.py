"""Walk-forward backtester for MasterSignalEngine.

Usage:
    python -m xau_pro_bot.backtest --csv path/to/history.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from xau_pro_bot.signals.engine import MasterSignalEngine


@dataclass
class BacktestResult:
    signals_generated: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    pnl_r: list[float] = field(default_factory=list)
    per_tier: dict[str, dict[str, int]] = field(
        default_factory=lambda: {t: {"n": 0, "w": 0, "l": 0}
                                  for t in ("STRONG", "NORMAL", "WEAK")})

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def expectancy(self) -> float:
        return float(np.mean(self.pnl_r)) if self.pnl_r else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(x for x in self.pnl_r if x > 0)
        losses = -sum(x for x in self.pnl_r if x < 0)
        if losses > 0:
            return gains / losses
        return float("inf") if gains > 0 else 0.0


def load_csv_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _resample(h1: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return h1.resample(rule).agg(agg).dropna()


def _outcome(future: pd.DataFrame, entry: float, sl: float,
             tp: float, direction: str, timeout_bars: int) -> tuple[str, float]:
    risk = abs(entry - sl)
    if risk <= 0:
        return "timeout", 0.0
    bars = future.iloc[:timeout_bars]
    for _, row in bars.iterrows():
        if direction == "BUY":
            if row["Low"] <= sl:
                return "loss", -1.0
            if row["High"] >= tp:
                return "win", abs(tp - entry) / risk
        else:
            if row["High"] >= sl:
                return "loss", -1.0
            if row["Low"] <= tp:
                return "win", abs(entry - tp) / risk
    return "timeout", 0.0


def run_backtest(history: dict[str, pd.DataFrame],
                 timeout_bars: int = 48,
                 step: int = 4) -> BacktestResult:
    eng = MasterSignalEngine()
    res = BacktestResult()
    h1 = history["H1"]
    if len(h1) < 250:
        return res

    for i in range(250, len(h1) - timeout_bars, step):
        cutoff = h1.index[i]
        slice_data: dict[str, pd.DataFrame] = {}
        for tf, df in history.items():
            slice_data[tf] = df.loc[:cutoff].tail(720)
        try:
            sig = eng.analyze(slice_data)
        except Exception:
            continue
        if sig["tier"] == "NO_SIGNAL" or sig.get("tp1") is None:
            continue
        res.signals_generated += 1
        target = sig.get("tp2") or sig["tp1"]
        future = h1.iloc[i + 1:]
        outcome, r = _outcome(future, sig["entry"], sig["sl"],
                              target, sig["direction"], timeout_bars)
        if outcome == "win":
            res.wins += 1
            res.per_tier[sig["tier"]]["n"] += 1
            res.per_tier[sig["tier"]]["w"] += 1
        elif outcome == "loss":
            res.losses += 1
            res.per_tier[sig["tier"]]["n"] += 1
            res.per_tier[sig["tier"]]["l"] += 1
        else:
            res.timeouts += 1
        res.pnl_r.append(r)
    return res


def _cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--timeout-bars", type=int, default=48)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--export", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    h1 = load_csv_history(Path(args.csv))
    history = {
        "H1": h1,
        "M15": _resample(h1, "15min"),
        "H4": _resample(h1, "4h"),
        "D1": _resample(h1, "1D"),
        "W1": _resample(h1, "1W"),
    }
    res = run_backtest(history, timeout_bars=args.timeout_bars, step=args.step)
    print(f"Signals:    {res.signals_generated}")
    print(f"Wins/Loss:  {res.wins} / {res.losses} (timeouts {res.timeouts})")
    print(f"Win rate:   {res.win_rate:.1%}")
    print(f"Expectancy: {res.expectancy:.2f} R")
    print(f"Profit f.:  {res.profit_factor:.2f}")
    print("By tier:")
    for tier, st in res.per_tier.items():
        if st["n"]:
            wr = st["w"] / st["n"]
            print(f"  {tier}: n={st['n']} wr={wr:.1%}")
    if args.export:
        pd.DataFrame({"R": res.pnl_r}).to_csv(args.export, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
