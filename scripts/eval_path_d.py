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
import sys
from pathlib import Path

import pandas as pd

from xau_pro_bot.backtest import BacktestResult, run_backtest


def _check_macro_csvs(dxy_csv: str | None, us10y_csv: str | None) -> bool:
    """Return True iff both macro CSVs are usable. Prints NO_MACRO_DATA on stderr
    when not. Used by Path F L3_path_e_stationary_macro mode.

    Contract: NEVER silently fall back; if macro CSVs are missing, the caller
    skips the L3 row and surfaces the reason in the results table.
    """
    ok = True
    if dxy_csv is None and us10y_csv is None:
        print("NO_MACRO_DATA: no --dxy-csv / --us10y-csv supplied", file=sys.stderr)
        return False
    if dxy_csv and not Path(dxy_csv).exists():
        print(f"NO_MACRO_DATA: dxy={dxy_csv} (file not found)", file=sys.stderr)
        ok = False
    if us10y_csv and not Path(us10y_csv).exists():
        print(f"NO_MACRO_DATA: us10y={us10y_csv} (file not found)", file=sys.stderr)
        ok = False
    if dxy_csv is None:
        print("NO_MACRO_DATA: --dxy-csv not supplied", file=sys.stderr)
        ok = False
    if us10y_csv is None:
        print("NO_MACRO_DATA: --us10y-csv not supplied", file=sys.stderr)
        ok = False
    return ok
from xau_pro_bot.models.hf_model import HFTradingModel
from xau_pro_bot.models.trade_filter_model import TradeFilterModel
from xau_pro_bot.signals.hybrid_policy import HybridThresholds


THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
EXPECTED_R_THRESHOLDS = (0.00, 0.03, 0.05, 0.10, 0.15)


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
    summary: dict = {
        "trades": r.signals_generated,
        "blocked": r.blocked_signals,
        "wins": r.wins, "losses": r.losses,
        "wr": round(r.win_rate, 4),
        "expectancy": round(r.expectancy, 4),
        "pf": round(r.profit_factor, 4),
        "avg_rr": round(r.average_rr, 4),
        "max_dd": round(r.max_drawdown, 4),
    }
    if r.per_tier:
        summary["by_tier"] = {
            t: {"n": cnt["n"], "w": cnt["w"], "l": cnt["l"]}
            for t, cnt in r.per_tier.items()
        }
    return summary


def tier_filter_result(r: BacktestResult, keep: set[str]) -> BacktestResult:
    """Synthesize a 'baseline-without-tier-X' result from per_tier counters.

    Carries rr_values, pnl_r, and equity_curve for the kept tiers so that
    PF / Expectancy / MaxDD on the synthesized result are honest. The old
    implementation only populated rr_values, which left H_no_weak /
    I_strong_only / J_strong_normal_only with PF=0 / Expectancy=0 despite
    non-zero trade counts.
    """
    out = BacktestResult()
    out.per_tier = {t: {"n": 0, "w": 0, "l": 0, "rr": []} for t in keep}
    for tier, cnt in r.per_tier.items():
        if tier in keep:
            out.signals_generated += cnt["n"]
            out.wins += cnt["w"]
            out.losses += cnt["l"]
            tier_rr = list(cnt.get("rr", []))
            out.per_tier[tier] = {**cnt, "rr": tier_rr}
            out.rr_values.extend(tier_rr)
    out.pnl_r = list(out.rr_values)
    running = 0.0
    out.equity_curve = []
    for r_value in out.pnl_r:
        running += r_value
        out.equity_curve.append(running)
    return out


def pick_best_threshold(sweep: dict[float, dict], *, min_kept: int) -> float | None:
    """Return threshold with highest PF among entries where kept >= min_kept.

    Tie-break by lower threshold value. Returns None (NO-GO) when no
    threshold meets min_kept or when the sweep is empty. Callers must
    treat None as a hard veto and skip test-slice evaluation.
    """
    if not sweep:
        return None
    eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
    if not eligible:
        return None
    return sorted(
        eligible.items(),
        key=lambda kv: (kv[1]["pf"], -kv[0]),
        reverse=True,
    )[0][0]



def run_all_modes(history, *, path_c_local: str | None,
                  path_d_filter: str | None,
                  path_d_filter_calibrated: str | None = None,
                  path_e: str | None = None,
                  path_c_stationary: str | None = None,
                  path_e_stationary: str | None = None,
                  path_e_stationary_macro: str | None = None,
                  dxy_csv: str | None = None,
                  us10y_csv: str | None = None,
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

    # G/H/I/J non-AI tier baselines
    results["G_baseline_all"]       = _result_summary(a)
    results["H_no_weak"]            = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))
    results["I_strong_only"]        = _result_summary(tier_filter_result(a, {"STRONG"}))
    results["J_strong_normal_only"] = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))

    # B Path C
    if path_c_local and Path(path_c_local).exists():
        ai = HFTradingModel(model_id="", model_type="sklearn", local_path=path_c_local)
        b = run_backtest(history, ai_model=ai, use_ai=True,
                         walk_from=t_test, **base_kwargs)
        results["B_path_c"] = _result_summary(b)

    min_kept = max(1, int(a.signals_generated * 0.25))

    # E Path D filter — pick threshold on validation window only
    chosen_threshold = None
    sweep: dict = {}
    if path_d_filter and Path(path_d_filter).exists():
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
                "max_dd": float(r.max_drawdown),
                "avg_rr": float(r.average_rr),
                "per_tier": dict(r.per_tier),
            }
        chosen_threshold = pick_best_threshold(sweep, min_kept=min_kept)
        if chosen_threshold is not None:
            flt = TradeFilterModel(local_path=path_d_filter,
                                   threshold=float(chosen_threshold))
            e = run_backtest(history, filter_model=flt,
                             walk_from=t_test, **base_kwargs)
            results["E_path_d_filter"] = _result_summary(e)

            thr_default = HybridThresholds(weak=0.70, normal=float(chosen_threshold), strong_block=0.80)
            results["F_hybrid_default"] = _result_summary(run_backtest(
                history, filter_model=flt, hybrid_thresholds=thr_default,
                walk_from=t_test, **base_kwargs))

            # WEAK never keeps: weak=2.0 is unreachable since good_prob ∈ [0,1]
            thr_no_weak = HybridThresholds(weak=2.0, normal=float(chosen_threshold), strong_block=0.80)
            results["F_hybrid_no_weak"] = _result_summary(run_backtest(
                history, filter_model=flt, hybrid_thresholds=thr_no_weak,
                walk_from=t_test, **base_kwargs))

            # Only STRONG kept (NORMAL also blocked via unreachable threshold)
            thr_strong_only = HybridThresholds(weak=2.0, normal=2.0, strong_block=0.80)
            results["F_hybrid_strong_only"] = _result_summary(run_backtest(
                history, filter_model=flt, hybrid_thresholds=thr_strong_only,
                walk_from=t_test, **base_kwargs))

            # NORMAL + STRONG only (same as no_weak by definition, but kept for clarity)
            results["F_hybrid_normal_strong"] = results["F_hybrid_no_weak"]
        else:
            results["E_path_d_filter"] = {
                "trades": 0, "pf": 0.0, "expectancy": 0.0,
                "no_go": True, "reason": "no_threshold_meets_min_kept",
            }

    # K Path D filter calibrated — separate model, same sweep/selection logic
    if path_d_filter_calibrated and Path(path_d_filter_calibrated).exists():
        sweep_cal: dict = {}
        for t in THRESHOLDS:
            flt = TradeFilterModel(local_path=path_d_filter_calibrated, threshold=float(t))
            r = run_backtest(history, filter_model=flt,
                             walk_from=t_val, walk_to=t_test, **base_kwargs)
            sweep_cal[t] = {
                "pf": float(r.profit_factor),
                "expectancy": float(r.expectancy),
                "wr": float(r.win_rate),
                "kept": int(r.signals_generated),
                "blocked": int(r.blocked_signals),
                "max_dd": float(r.max_drawdown),
                "avg_rr": float(r.average_rr),
                "per_tier": dict(r.per_tier),
            }
        chosen_cal = pick_best_threshold(sweep_cal, min_kept=min_kept)
        if chosen_cal is not None:
            flt = TradeFilterModel(local_path=path_d_filter_calibrated,
                                   threshold=float(chosen_cal))
            k = run_backtest(history, filter_model=flt,
                             walk_from=t_test, **base_kwargs)
            results["K_path_d_filter_calibrated"] = _result_summary(k)
            results.setdefault("threshold_sweeps", {})["K_path_d_filter_calibrated"] = sweep_cal
            results.setdefault("chosen_thresholds", {})["K_path_d_filter_calibrated"] = float(chosen_cal)
        else:
            results["K_path_d_filter_calibrated"] = {
                "trades": 0, "pf": 0.0, "expectancy": 0.0,
                "no_go": True, "reason": "no_threshold_meets_min_kept",
            }

    # L Path E — expected-R filter; predicted_R sweep on validation, best applied to test
    chosen_er = None
    sweep_er: dict = {}
    if path_e and Path(path_e).exists():
        from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel
        for t in EXPECTED_R_THRESHOLDS:
            flt = ExpectedRFilterModel(local_path=path_e, threshold=float(t))
            r = run_backtest(history, filter_model=flt,
                             walk_from=t_val, walk_to=t_test, **base_kwargs)
            sweep_er[t] = {
                "pf": float(r.profit_factor),
                "expectancy": float(r.expectancy),
                "wr": float(r.win_rate),
                "kept": int(r.signals_generated),
                "blocked": int(r.blocked_signals),
                "max_dd": float(r.max_drawdown),
                "avg_rr": float(r.average_rr),
            }
        chosen_er = pick_best_threshold(sweep_er, min_kept=min_kept)
        if chosen_er is not None:
            flt = ExpectedRFilterModel(local_path=path_e,
                                       threshold=float(chosen_er))
            l_res = run_backtest(history, filter_model=flt,
                                 walk_from=t_test, **base_kwargs)
            results["L_path_e_expected_r"] = _result_summary(l_res)
        else:
            results["L_path_e_expected_r"] = {
                "trades": 0, "pf": 0.0, "expectancy": 0.0,
                "no_go": True, "reason": "no_threshold_meets_min_kept",
            }

    # B2 Path C stationary — same as B_path_c but with the stationary-tagged artifact.
    # Engine dispatches features through the model's feature_set tag at predict time.
    if path_c_stationary and Path(path_c_stationary).exists():
        ai_b2 = HFTradingModel(model_id="", model_type="sklearn",
                               local_path=path_c_stationary)
        b2 = run_backtest(history, ai_model=ai_b2, use_ai=True,
                          walk_from=t_test, **base_kwargs)
        results["B2_path_c_stationary"] = _result_summary(b2)

    # L2 Path E stationary — expected-R sweep on the stationary feature set.
    sweep_l2: dict = {}
    chosen_l2 = None
    if path_e_stationary and Path(path_e_stationary).exists():
        from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel
        for t in EXPECTED_R_THRESHOLDS:
            flt = ExpectedRFilterModel(local_path=path_e_stationary, threshold=float(t))
            r = run_backtest(history, filter_model=flt,
                             walk_from=t_val, walk_to=t_test, **base_kwargs)
            sweep_l2[t] = {
                "pf": float(r.profit_factor),
                "expectancy": float(r.expectancy),
                "wr": float(r.win_rate),
                "kept": int(r.signals_generated),
                "blocked": int(r.blocked_signals),
                "max_dd": float(r.max_drawdown),
                "avg_rr": float(r.average_rr),
            }
        chosen_l2 = pick_best_threshold(sweep_l2, min_kept=min_kept)
        if chosen_l2 is not None:
            flt = ExpectedRFilterModel(local_path=path_e_stationary,
                                       threshold=float(chosen_l2))
            l2_res = run_backtest(history, filter_model=flt,
                                  walk_from=t_test, **base_kwargs)
            results["L2_path_e_stationary"] = _result_summary(l2_res)
            results.setdefault("threshold_sweeps", {})["L2_path_e_stationary"] = sweep_l2
            results.setdefault("chosen_thresholds", {})["L2_path_e_stationary"] = float(chosen_l2)
        else:
            results["L2_path_e_stationary"] = {
                "trades": 0, "pf": 0.0, "expectancy": 0.0,
                "no_go": True, "reason": "no_threshold_meets_min_kept",
            }

    # L3 Path E stationary + macro — requires both DXY and US10Y CSVs.
    sweep_l3: dict = {}
    chosen_l3 = None
    if path_e_stationary_macro and Path(path_e_stationary_macro).exists():
        if not _check_macro_csvs(dxy_csv, us10y_csv):
            results["L3_path_e_stationary_macro"] = {
                "trades": 0, "pf": 0.0, "expectancy": 0.0,
                "skipped": True, "reason": "NO_MACRO_DATA",
            }
        else:
            from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel
            for t in EXPECTED_R_THRESHOLDS:
                flt = ExpectedRFilterModel(local_path=path_e_stationary_macro,
                                           threshold=float(t))
                r = run_backtest(history, filter_model=flt,
                                 walk_from=t_val, walk_to=t_test, **base_kwargs)
                sweep_l3[t] = {
                    "pf": float(r.profit_factor),
                    "expectancy": float(r.expectancy),
                    "wr": float(r.win_rate),
                    "kept": int(r.signals_generated),
                    "blocked": int(r.blocked_signals),
                    "max_dd": float(r.max_drawdown),
                    "avg_rr": float(r.average_rr),
                }
            chosen_l3 = pick_best_threshold(sweep_l3, min_kept=min_kept)
            if chosen_l3 is not None:
                flt = ExpectedRFilterModel(local_path=path_e_stationary_macro,
                                           threshold=float(chosen_l3))
                l3_res = run_backtest(history, filter_model=flt,
                                      walk_from=t_test, **base_kwargs)
                results["L3_path_e_stationary_macro"] = _result_summary(l3_res)
                results.setdefault("threshold_sweeps", {})["L3_path_e_stationary_macro"] = sweep_l3
                results.setdefault("chosen_thresholds", {})["L3_path_e_stationary_macro"] = float(chosen_l3)
            else:
                results["L3_path_e_stationary_macro"] = {
                    "trades": 0, "pf": 0.0, "expectancy": 0.0,
                    "no_go": True, "reason": "no_threshold_meets_min_kept",
                }

    return {
        "results": results,
        "threshold_sweep": sweep,
        "chosen_threshold": chosen_threshold,
        "expected_r_sweep": sweep_er,
        "chosen_expected_r_threshold": chosen_er,
        "test_window": (str(t_test), str(h1.index[-1])),
        "val_window":  (str(t_val), str(t_test)),
    }


_NON_MODE_KEYS = {"threshold_sweeps", "chosen_thresholds"}


def _md_table(summary: dict[str, dict]) -> str:
    cols = ["trades", "blocked", "wins", "losses", "wr",
            "expectancy", "pf", "avg_rr", "max_dd"]
    header = "| mode | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for k, v in summary.items():
        if k in _NON_MODE_KEYS:
            continue
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
        lines.append("| th | kept | blocked | PF | Expectancy | WR | MaxDD | AvgRR |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for t, m in sorted(sweep.items()):
            lines.append(
                f"| {t:.2f} | {m['kept']} | {m['blocked']} | "
                f"{m['pf']:.3f} | {m['expectancy']:.3f} | {m['wr']:.3f} | "
                f"{m.get('max_dd', 0.0):.3f} | {m.get('avg_rr', 0.0):.3f} |"
            )
    else:
        lines.append("_(no sweep — filter not provided)_")

    # K calibrated sweep (optional)
    cal_sweeps = res.get("threshold_sweeps", {})
    cal_thresholds = res.get("chosen_thresholds", {})
    if "K_path_d_filter_calibrated" in cal_sweeps:
        sweep_k = cal_sweeps["K_path_d_filter_calibrated"]
        chosen_k = cal_thresholds.get("K_path_d_filter_calibrated")
        lines += [
            "",
            "## K Calibrated Filter — Threshold Sweep (validation)",
            "",
            f"**Chosen threshold:** {chosen_k}",
            "",
            "| th | kept | blocked | PF | Expectancy | WR | MaxDD | AvgRR |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, m in sorted(sweep_k.items()):
            lines.append(
                f"| {t:.2f} | {m['kept']} | {m['blocked']} | "
                f"{m['pf']:.3f} | {m['expectancy']:.3f} | {m['wr']:.3f} | "
                f"{m.get('max_dd', 0.0):.3f} | {m.get('avg_rr', 0.0):.3f} |"
            )

    er_sweep = payload.get("expected_r_sweep") or {}
    if er_sweep:
        chosen_er = payload.get("chosen_expected_r_threshold")
        lines += [
            "",
            "## L Path E (expected_R) — Threshold Sweep (validation)",
            "",
            f"**Chosen threshold (predicted_R >):** {chosen_er}",
            "",
            "| th | kept | blocked | PF | Expectancy | WR | MaxDD | AvgRR |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, m in sorted(er_sweep.items()):
            lines.append(
                f"| {t:.2f} | {m['kept']} | {m['blocked']} | "
                f"{m['pf']:.3f} | {m['expectancy']:.3f} | {m['wr']:.3f} | "
                f"{m.get('max_dd', 0.0):.3f} | {m.get('avg_rr', 0.0):.3f} |"
            )

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
    ap.add_argument("--path-d-filter-calibrated", default=None)
    ap.add_argument("--path-e", default=None,
                    help="Path E expected_R joblib bundle.")
    ap.add_argument("--path-c-stationary", default=None,
                    help="Path F: Path C trained on the stationary feature set (B2).")
    ap.add_argument("--path-e-stationary", default=None,
                    help="Path F: Path E trained on the stationary feature set (L2).")
    ap.add_argument("--path-e-stationary-macro", default=None,
                    help="Path F: Path E trained on stationary+macro feature set (L3).")
    ap.add_argument("--dxy-csv", default=None,
                    help="DXY CSV (required for L3_path_e_stationary_macro).")
    ap.add_argument("--us10y-csv", default=None,
                    help="US10Y CSV (required for L3_path_e_stationary_macro).")
    ap.add_argument("--report", default="docs/reports/path_d_trade_outcome_results.md")
    ap.add_argument("--metrics-json", default="models_cache/path_d_metrics.json")
    args = ap.parse_args()

    history = _load(Path(args.csv))
    cal_path = args.path_d_filter_calibrated
    payload = run_all_modes(
        history,
        path_c_local=args.path_c if Path(args.path_c).exists() else None,
        path_d_filter=args.path_d_filter if Path(args.path_d_filter).exists() else None,
        path_d_filter_calibrated=cal_path if cal_path and Path(cal_path).exists() else None,
        path_e=args.path_e if args.path_e and Path(args.path_e).exists() else None,
        path_c_stationary=args.path_c_stationary,
        path_e_stationary=args.path_e_stationary,
        path_e_stationary_macro=args.path_e_stationary_macro,
        dxy_csv=args.dxy_csv,
        us10y_csv=args.us10y_csv,
    )
    write_report(payload, Path(args.report), Path(args.metrics_json))
    print(json.dumps(payload["results"], indent=2))
    print(f"\nThreshold sweep: {payload['threshold_sweep']}")
    print(f"Chosen threshold: {payload['chosen_threshold']}")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
