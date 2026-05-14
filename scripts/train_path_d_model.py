"""Train Path D models from a long M15 CSV.

Run all three artifacts in one go:

    PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \\
        --csv ./data_long_m15.csv --out-dir ./models_cache

Outputs:
    models_cache/path_d_directional_a1_lgb.joblib
    models_cache/path_d_directional_a2_lgb.joblib
    models_cache/path_d_trade_outcome_lgb.joblib
    models_cache/path_d_dataset.parquet
    models_cache/path_d_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from xau_pro_bot.models.label_policy import LabelPolicy
from xau_pro_bot.models.path_d_harvest import HarvestConfig, harvest_path_d_samples
from xau_pro_bot.models.train_path_d import (
    save_model, train_directional, train_filter, train_filter_calibrated,
)

_AUDIT_CONFIGS = [
    ("step_h1=4",          HarvestConfig(step_h1=4)),
    ("step_h1=1",          HarvestConfig(step_h1=1)),
    ("step_h1=1,step_m15=2", HarvestConfig(step_h1=1, step_m15=2)),
]

_OUTCOME_COLS = ["TP", "SL", "UNRESOLVED", "SAME_CANDLE_SL_FIRST"]


def _run_audit(history: dict, configs: list) -> list[dict]:
    """Return list of audit rows for the given configs. Pure function — no I/O."""
    rows: list[dict] = []
    for label, cfg in configs:
        df = harvest_path_d_samples(history, cfg)
        if df.empty:
            rows.append({
                "config": label,
                "rows": 0,
                "baseline": 0,
                "synthetic": 0,
                **{c: "0.0%" for c in _OUTCOME_COLS},
            })
            continue

        total = len(df)
        baseline = int(df["baseline_sample"].sum())
        synthetic = int(df["is_synthetic"].sum())
        vc = df["outcome_class"].value_counts()
        pct = {c: f"{vc.get(c, 0) / total * 100:.1f}%" for c in _OUTCOME_COLS}
        rows.append({
            "config": label,
            "rows": total,
            "baseline": baseline,
            "synthetic": synthetic,
            **pct,
        })
    return rows


def _acceptance_guard(metrics: dict, *, min_kept_pct: float = 0.05) -> None:
    """Raise SystemExit if the trained filter is operationally useless.

    Iteration 2 invariant: a model that predicts BAD for everything, or keeps
    fewer than `min_kept_pct` of test trades, is not a viable trade filter
    regardless of its accuracy on the BAD-majority class.
    """
    if metrics.get("predicts_only_bad"):
        raise SystemExit("acceptance guard: model predicts BAD for every test sample")
    cm = metrics.get("confusion_matrix")
    if cm:
        kept_pred = sum(row[1] for row in cm)
        total = sum(sum(row) for row in cm)
        if total > 0 and (kept_pred / total) < min_kept_pct:
            raise SystemExit(
                f"acceptance guard: kept_pct={kept_pred/total:.3f} < {min_kept_pct}")


def _load_history(csv: Path) -> dict[str, pd.DataFrame]:
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


def _run_label_policy_sweep(df: pd.DataFrame, out_dir: Path) -> dict:
    """Train filter once per LabelPolicy and write policy_sweep.json.

    Returns the per-policy results dict (also written to disk).
    Does NOT save per-policy joblib artifacts.
    """
    results: dict = {}
    for policy in LabelPolicy:
        pval = policy.value
        try:
            _model, metrics = train_filter(df, policy=pval)
        except Exception as exc:  # noqa: BLE001
            logging.warning("policy=%s failed: %s", pval, exc)
            results[pval] = {"error": str(exc)}
            continue

        # Determine class_balance (fraction labelled GOOD=1) from confusion matrix
        cm = metrics.get("confusion_matrix") or []
        if cm:
            total_positives = sum(cm[1]) if len(cm) > 1 else 0
            total = sum(sum(row) for row in cm)
            class_balance = float(total_positives / total) if total > 0 else 0.0
        else:
            class_balance = 0.0

        predicts_only_bad = bool(
            all(row[1] == 0 for row in cm) if cm and len(cm[0]) > 1 else True
        )

        degenerate = False
        try:
            _acceptance_guard(
                {"predicts_only_bad": predicts_only_bad, "confusion_matrix": cm},
                min_kept_pct=0.05,
            )
        except SystemExit:
            degenerate = True

        results[pval] = {
            "n": int(metrics.get("n_test", 0) + metrics.get("n_train", 0) + metrics.get("n_val", 0)),
            "class_balance": class_balance,
            "good_prob_stats": {},
            "precision": float(metrics.get("precision_macro", 0.0)),
            "recall": float(metrics.get("recall_macro", 0.0)),
            "confusion_matrix": cm,
            "predicts_only_bad": predicts_only_bad,
            "degenerate": degenerate,
        }

    out_path = out_dir / "path_d_filter_policy_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))
    logging.info("Policy sweep written to %s", out_path)
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--step-h1", type=int, default=4)
    ap.add_argument("--timeout-m15", type=int, default=192)
    ap.add_argument("--synth-stride", type=int, default=8)
    ap.add_argument("--allow-degenerate", action="store_true", default=False,
                    help="Downgrade acceptance guard from SystemExit to a warning")
    ap.add_argument("--audit-only", action="store_true",
                    help="Print sample counts for several harvest configs and exit.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Also train an isotonic-calibrated filter and save it.")
    ap.add_argument("--label-policy-sweep", action="store_true",
                    help="Train filter once per label policy and emit policy_sweep.json.")
    args = ap.parse_args()

    history = _load_history(Path(args.csv))
    print(f"Loaded M15: {len(history['M15'])} bars")

    if args.audit_only:
        audit_rows = _run_audit(history, _AUDIT_CONFIGS)
        # Header
        col_w = 26
        header = f"{'config':<{col_w}} {'rows':>7} {'baseline':>9} {'synthetic':>10}  " + \
                 "  ".join(f"{c:>22}" for c in _OUTCOME_COLS)
        print(header)
        print("-" * len(header))
        for r in audit_rows:
            line = (
                f"{r['config']:<{col_w}} {r['rows']:>7} {r['baseline']:>9} "
                f"{r['synthetic']:>10}  " +
                "  ".join(f"{r[c]:>22}" for c in _OUTCOME_COLS)
            )
            print(line)
        print()
        print("Dataset sources:")
        print("  training source:           data_long_m15.csv (canonical)")
        print("  robustness source:         data_xauusd_15m.csv (GC=F, DO NOT merge into training)")
        print("  evaluation-only candidate: data_xauusd_m15.csv (~2025-07-21+, document only)")
        sys.exit(0)

    if args.out_dir is None:
        ap.error("--out-dir is required unless --audit-only is set")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = HarvestConfig(step_h1=args.step_h1, timeout_m15=args.timeout_m15,
                        include_synthetic=True, synth_stride=args.synth_stride)
    df = harvest_path_d_samples(history, cfg)
    print(f"Dataset: rows={len(df)}, baseline={int(df['baseline_sample'].sum())}, "
          f"synthetic={int(df['is_synthetic'].sum())}")
    if len(df) < 200:
        print("Not enough samples - aborting.")
        return 1

    if args.label_policy_sweep:
        print("Running label policy sweep...")
        _run_label_policy_sweep(df, out_dir)
        print("Sweep done. Artifacts:", sorted(p.name for p in out_dir.glob("path_d_*")))
        return 0

    df.to_parquet(out_dir / "path_d_dataset.parquet")

    outcome_dist = df["outcome_class"].value_counts(normalize=True).to_dict()
    print("Outcome distribution:", outcome_dist)

    print("Training Directional A1...")
    m_a1, met_a1 = train_directional(df, variant="A1")
    save_model(m_a1, met_a1["feature_cols"], out_dir / "path_d_directional_a1_lgb.joblib")

    print("Training Directional A2...")
    m_a2, met_a2 = train_directional(df, variant="A2")
    save_model(m_a2, met_a2["feature_cols"], out_dir / "path_d_directional_a2_lgb.joblib")

    print("Training Filter...")
    m_f, met_f = train_filter(df)
    save_model(m_f, met_f["feature_cols"], out_dir / "path_d_trade_outcome_lgb.joblib")
    try:
        _acceptance_guard(met_f)
    except SystemExit as exc:
        if args.allow_degenerate:
            print(f"warning: {exc}")
        else:
            raise

    met_cal: dict | None = None
    if args.calibrate:
        print("Training Calibrated Filter...")
        m_cal, met_cal = train_filter_calibrated(df)
        save_model(m_cal, met_cal["feature_cols"],
                   out_dir / "path_d_trade_outcome_calibrated.joblib")
        try:
            _acceptance_guard(met_cal)
        except SystemExit as exc:
            if args.allow_degenerate:
                print(f"warning (calibrated): {exc}")
            else:
                raise

    metrics = {
        "outcome_distribution": outcome_dist,
        "directional_a1": {k: v for k, v in met_a1.items() if k not in ("report", "feature_cols")},
        "directional_a2": {k: v for k, v in met_a2.items() if k not in ("report", "feature_cols")},
        "filter":         {k: v for k, v in met_f.items()  if k not in ("report", "feature_cols")},
        "reports": {
            "directional_a1": met_a1["report"],
            "directional_a2": met_a2["report"],
            "filter":         met_f["report"],
        },
    }
    if met_cal is not None:
        metrics["filter_calibrated"] = {
            k: v for k, v in met_cal.items() if k not in ("report", "feature_cols")
        }
        if "report" in met_cal:
            metrics["reports"]["filter_calibrated"] = met_cal["report"]
    (out_dir / "path_d_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Done. Artifacts:", sorted(p.name for p in out_dir.glob("path_d_*")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
