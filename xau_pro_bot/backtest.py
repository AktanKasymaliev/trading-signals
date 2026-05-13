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
from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot.models.hf_model import HFTradingModel
from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.router import StreamRouter


@dataclass
class BacktestResult:
    signals_generated: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    blocked_signals: int = 0
    pnl_r: list[float] = field(default_factory=list)
    rr_values: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
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

    @property
    def average_rr(self) -> float:
        return float(np.mean(self.rr_values)) if self.rr_values else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            max_dd = min(max_dd, value - peak)
        return abs(max_dd)


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


def _build_analyzer(stream: str, use_ai: bool = False,
                    ai_model: Any | None = None,
                    ai_model_id: str = "",
                    ai_model_type: str = "sklearn",
                    ai_model_revision: str = "",
                    filter_model: Any | None = None,
                    hybrid_thresholds: Any | None = None):
    router = StreamRouter()
    if stream not in router.analyzers:
        raise ValueError(f"Unknown stream: {stream}")
    if stream != "intraday" or (not use_ai and filter_model is None):
        return router.analyzers[stream]
    model = ai_model
    if model is None and ai_model_id:
        model = HFTradingModel(
            model_id=ai_model_id,
            model_type=ai_model_type,
            revision=ai_model_revision,
        )
    return MasterSignalEngine(ai_enabled=use_ai, ai_model=model,
                              filter_model=filter_model,
                              hybrid_thresholds=hybrid_thresholds)


def run_backtest(history: dict[str, pd.DataFrame],
                 timeout_bars: int = 48,
                 step: int = 4,
                 stream: str = "intraday",
                 use_ai: bool = False,
                 ai_model: Any | None = None,
                 ai_model_id: str = "",
                 ai_model_type: str = "sklearn",
                 ai_model_revision: str = "",
                 filter_model: Any | None = None,
                 hybrid_thresholds: Any | None = None) -> BacktestResult:
    analyzer = _build_analyzer(
        stream=stream,
        use_ai=use_ai,
        ai_model=ai_model,
        ai_model_id=ai_model_id,
        ai_model_type=ai_model_type,
        ai_model_revision=ai_model_revision,
        filter_model=filter_model,
        hybrid_thresholds=hybrid_thresholds,
    )
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
            sig = analyzer.analyze(slice_data)
        except Exception:
            continue
        if sig is None:
            continue
        if sig["tier"] == "NO_SIGNAL":
            if sig.get("ai_blocked"):
                res.blocked_signals += 1
            continue
        if sig.get("tp1") is None:
            continue
        res.signals_generated += 1
        if sig.get("rr") is not None:
            res.rr_values.append(float(sig["rr"]))
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
        previous = res.equity_curve[-1] if res.equity_curve else 0.0
        res.equity_curve.append(previous + r)
    return res


def compare_backtests(history: dict[str, pd.DataFrame],
                      timeout_bars: int = 48,
                      step: int = 4,
                      stream: str = "intraday",
                      ai_model: Any | None = None,
                      ai_model_id: str = "",
                      ai_model_type: str = "sklearn",
                      ai_model_revision: str = "",
                      filter_model: Any | None = None,
                      hybrid_thresholds: Any | None = None) -> dict[str, Any]:
    baseline = run_backtest(
        history=history,
        timeout_bars=timeout_bars,
        step=step,
        stream=stream,
        use_ai=False,
    )
    ai = run_backtest(
        history=history,
        timeout_bars=timeout_bars,
        step=step,
        stream=stream,
        use_ai=True,
        ai_model=ai_model,
        ai_model_id=ai_model_id,
        ai_model_type=ai_model_type,
        ai_model_revision=ai_model_revision,
        filter_model=filter_model,
        hybrid_thresholds=hybrid_thresholds,
    )
    return {
        "baseline": baseline,
        "ai": ai,
        "summary": {
            "trade_delta": ai.signals_generated - baseline.signals_generated,
            "win_rate_delta": ai.win_rate - baseline.win_rate,
            "expectancy_delta": ai.expectancy - baseline.expectancy,
            "profit_factor_delta": ai.profit_factor - baseline.profit_factor,
            "blocked_signals": ai.blocked_signals,
        },
    }


def _cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--timeout-bars", type=int, default=48)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--stream", default="intraday",
                   choices=["intraday", "swing", "scalp", "all"])
    p.add_argument("--export", default=None)
    p.add_argument("--use-ai", action="store_true")
    p.add_argument("--ai-model-id", default="")
    p.add_argument("--ai-model-type", default="sklearn",
                   choices=["sklearn", "transformers", "custom"])
    p.add_argument("--ai-model-revision", default="")
    p.add_argument("--compare-ai", action="store_true")
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
    streams = ["intraday", "swing", "scalp"] if args.stream == "all" else [args.stream]
    default_timeouts = {"intraday": args.timeout_bars, "swing": 336, "scalp": 8}
    export_rows: list[dict[str, float | str]] = []

    def print_result(label: str, res: BacktestResult) -> None:
        print(f"\n=== {label} ===")
        print(f"Signals:      {res.signals_generated}")
        print(f"Blocked AI:   {res.blocked_signals}")
        print(f"Wins/Loss:    {res.wins} / {res.losses} (timeouts {res.timeouts})")
        print(f"Win rate:     {res.win_rate:.1%}")
        print(f"Expectancy:   {res.expectancy:.2f} R")
        print(f"Profit f.:    {res.profit_factor:.2f}")
        print(f"Max DD:       {res.max_drawdown:.2f} R")
        print(f"Avg RR:       {res.average_rr:.2f}")

    for s in streams:
        timeout = default_timeouts[s]
        if args.compare_ai:
            comparison = compare_backtests(
                history=history,
                timeout_bars=timeout,
                step=args.step,
                stream=s,
                ai_model_id=args.ai_model_id,
                ai_model_type=args.ai_model_type,
                ai_model_revision=args.ai_model_revision,
            )
            print(f"\n=== Stream: {s} comparison ===")
            print_result("baseline", comparison["baseline"])
            print_result("baseline + AI", comparison["ai"])
            summary = comparison["summary"]
            print("\nSummary:")
            print(f"Trade delta:       {summary['trade_delta']}")
            print(f"Win-rate delta:    {summary['win_rate_delta']:.1%}")
            print(f"Expectancy delta:  {summary['expectancy_delta']:.2f} R")
            print(f"Profit factor Δ:   {summary['profit_factor_delta']:.2f}")
            print(f"Blocked signals:   {summary['blocked_signals']}")
            export_rows.extend({"stream": s, "mode": "baseline", "R": r}
                               for r in comparison["baseline"].pnl_r)
            export_rows.extend({"stream": s, "mode": "ai", "R": r}
                               for r in comparison["ai"].pnl_r)
            continue

        res = run_backtest(
            history,
            timeout_bars=timeout,
            step=args.step,
            stream=s,
            use_ai=args.use_ai,
            ai_model_id=args.ai_model_id,
            ai_model_type=args.ai_model_type,
            ai_model_revision=args.ai_model_revision,
        )
        print_result(f"Stream: {s}", res)
        export_rows.extend({"stream": s, "mode": "ai" if args.use_ai else "baseline", "R": r}
                           for r in res.pnl_r)

    if args.export:
        pd.DataFrame(export_rows).to_csv(args.export, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
