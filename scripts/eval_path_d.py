"""Path D evaluator.

Modes (all run on the same history, same step / timeout):

  AI modes:
    A baseline only (AI off)
    B Path C directional (existing)
    E Path D filter (threshold chosen on validation, applied to test)
    F Hybrid (Path D filter + tier-aware thresholds)

  Non-AI baselines (prove AI > simple tier rules):
    H baseline without WEAK
    I baseline STRONG only
    J baseline STRONG + NORMAL only

Threshold sweep runs the filter on the VALIDATION slice across
{0.50..0.75}; the best by (PF, kept_trades) — subject to a 25%
min-kept floor — is applied ONCE to the test slice.

Writes `docs/reports/path_d_trade_outcome_results.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from xau_pro_bot.backtest import BacktestResult, run_backtest
from xau_pro_bot.models.hf_model import HFTradingModel
from xau_pro_bot.models.trade_filter_model import TradeFilterModel
from xau_pro_bot.signals.hybrid_policy import HybridThresholds


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def _load(csv: Path) -> dict[str, pd.DataFrame]:
    m15 = pd.read_csv(csv)
    m15["datetime"] = pd.to_datetime(m15["datetime"], utc=True)
    m15 = m15.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }



def _result_summary(r: BacktestResult) -> dict:
    return {
        "trades": r.signals_generated,
        "blocked": r.blocked_signals,
        "wins": r.wins, "losses": r.losses,
        "wr": round(r.win_rate, 4),
        "expectancy": round(r.expectancy, 4),
        "pf": round(r.profit_factor, 4),
        "avg_rr": round(r.average_rr, 4),
        "max_dd": round(r.max_drawdown, 4),
    }


def tier_filter_result(r: BacktestResult, keep: set[str]) -> BacktestResult:
    """Synthesize a 'baseline-without-tier-X' result from per_tier counters."""
    out = BacktestResult()
    out.per_tier = {t: {"n": 0, "w": 0, "l": 0} for t in keep}
    for tier, cnt in r.per_tier.items():
        if tier in keep:
            out.signals_generated += cnt["n"]
            out.wins += cnt["w"]
            out.losses += cnt["l"]
            out.per_tier[tier] = dict(cnt)
    out.rr_values = []
    return out


def pick_best_threshold(sweep: dict[float, dict], min_kept: int) -> float | None:
    eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
    if not eligible:
        return None
    return sorted(eligible.items(),
                  key=lambda kv: (kv[1]["pf"], kv[1]["kept"]),
                  reverse=True)[0][0]



def run_all_modes(history, *, path_c_local: str | None,
                  path_d_filter: str | None,
                  val_split=(0.70, 0.85)) -> dict:
    h1 = history["H1"]
    n = len(h1)
    t_val = h1.index[int(n * val_split[0])]
    t_test = h1.index[int(n * val_split[1])]

    base_kwargs = dict(timeout_bars=48, step=4, stream="intraday")

    results: dict = {}

    # A baseline on the test window (full history given, walk_from filters)
    a = run_backtest(history, walk_from=t_test, **base_kwargs)
    results["A_baseline"] = _result_summary(a)

    # H/I/J non-AI tier filters
    results["H_no_weak"]            = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))
    results["I_strong_only"]        = _result_summary(tier_filter_result(a, {"STRONG"}))
    results["J_strong_normal_only"] = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))

    # B Path C
    if path_c_local and Path(path_c_local).exists():
        ai = HFTradingModel(model_id="", model_type="sklearn", local_path=path_c_local)
        b = run_backtest(history, ai_model=ai, use_ai=True,
                         walk_from=t_test, **base_kwargs)
        results["B_path_c"] = _result_summary(b)

    # E Path D filter — pick threshold on validation window only
    chosen_threshold = None
    sweep: dict = {}
    if path_d_filter and Path(path_d_filter).exists():
        min_kept = max(1, int(a.signals_generated * 0.25))
        sweep = {}
        for t in THRESHOLDS:
            flt = TradeFilterModel(local_path=path_d_filter, threshold=float(t))
            r = run_backtest(history, filter_model=flt,
                             walk_from=t_val, walk_to=t_test, **base_kwargs)
            sweep[t] = {
                "pf": float(r.profit_factor),
                "expectancy": float(r.expectancy),
                "wr": float(r.win_rate),
                "kept": int(r.signals_generated),
                "blocked": int(r.blocked_signals),
            }
        chosen_threshold = pick_best_threshold(sweep, min_kept=min_kept)
        if chosen_threshold is not None:
            flt = TradeFilterModel(local_path=path_d_filter,
                                   threshold=float(chosen_threshold))
            e = run_backtest(history, filter_model=flt,
                             walk_from=t_test, **base_kwargs)
            results["E_path_d_filter"] = _result_summary(e)

            thr = HybridThresholds(weak=0.70, normal=float(chosen_threshold),
                                   strong_block=0.80)
            f = run_backtest(history, filter_model=flt,
                             hybrid_thresholds=thr,
                             walk_from=t_test, **base_kwargs)
            results["F_hybrid"] = _result_summary(f)

    return {
        "results": results,
        "threshold_sweep": sweep,
        "chosen_threshold": chosen_threshold,
        "test_window": (str(t_test), str(h1.index[-1])),
        "val_window":  (str(t_val), str(t_test)),
    }


def _md_table(summary: dict[str, dict]) -> str:
    cols = ["trades", "blocked", "wins", "losses", "wr",
            "expectancy", "pf", "avg_rr", "max_dd"]
    header = "| mode | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for k, v in summary.items():
        rows.append("| " + k + " | " + " | ".join(str(v.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def write_report(payload: dict, out_path: Path,
                  metrics_json_path: Path | None = None) -> None:
    res = payload["results"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Path D — Trade Outcome Results",
        "",
        f"**Test window:** {payload['test_window'][0]} -> {payload['test_window'][1]}",
        f"**Validation window:** {payload['val_window'][0]} -> {payload['val_window'][1]}",
        f"**Chosen filter threshold (from validation):** {payload['chosen_threshold']}",
        "",
        "## Modes",
        "",
        _md_table(res),
        "",
        "## Filter Threshold Sweep (validation)",
        "",
    ]
    sweep = payload["threshold_sweep"]
    if sweep:
        lines.append("| threshold | kept | blocked | wr | expectancy | pf |")
        lines.append("|---|---|---|---|---|---|")
        for t, m in sorted(sweep.items()):
            lines.append(f"| {t:.2f} | {m['kept']} | {m['blocked']} | "
                         f"{m['wr']:.3f} | {m['expectancy']:.3f} | {m['pf']:.3f} |")
    else:
        lines.append("_(no sweep — filter not provided)_")
    base_trades = res.get("A_baseline", {}).get("trades", 0)
    lines += [
        "",
        "## Acceptance check",
        "",
        f"- Min trade floor (25% of baseline test trades): {int(base_trades * 0.25)}",
        "",
        "## Notes",
        "",
        "- Path C: forward-return labels (legacy).",
        "- Path D: TP/SL outcomes on M15, time-split 70/15/15, threshold picked on validation only.",
        "- Acceptance: PF > Path C **and** kept_trades >= 25% baseline, else 'do not deploy'.",
    ]
    if metrics_json_path is not None and metrics_json_path.exists():
        try:
            j = json.loads(metrics_json_path.read_text())
            lines += [
                "",
                "## Training metrics",
                "",
                "```json",
                json.dumps({k: v for k, v in j.items() if k != "reports"}, indent=2),
                "```",
            ]
        except Exception:
            pass
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--path-c", default="./models_cache/path_c_lgb.joblib")
    ap.add_argument("--path-d-filter",
                     default="./models_cache/path_d_trade_outcome_lgb.joblib")
    ap.add_argument("--report", default="docs/reports/path_d_trade_outcome_results.md")
    ap.add_argument("--metrics-json", default="models_cache/path_d_metrics.json")
    args = ap.parse_args()

    history = _load(Path(args.csv))
    payload = run_all_modes(
        history,
        path_c_local=args.path_c if Path(args.path_c).exists() else None,
        path_d_filter=args.path_d_filter if Path(args.path_d_filter).exists() else None,
    )
    write_report(payload, Path(args.report), Path(args.metrics_json))
    print(json.dumps(payload["results"], indent=2))
    print(f"\nThreshold sweep: {payload['threshold_sweep']}")
    print(f"Chosen threshold: {payload['chosen_threshold']}")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
