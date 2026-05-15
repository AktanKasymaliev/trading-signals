"""Train Path C LightGBM model on local M15 history.

Run:
    PYTHONPATH=. .venv/bin/python scripts/train_path_c_model.py \\
        --csv ./data_long_m15.csv \\
        --out ./models_cache/path_c_lgb.joblib
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from xau_pro_bot.models.train_lightgbm import (
    build_training_dataset, save_model, train_lightgbm,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.003)
    p.add_argument("--feature-set", choices=["legacy", "stationary"],
                   default="legacy",
                   help="Path F: 'stationary' uses build_stationary_features (B2 retrain).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    m15 = pd.read_csv(args.csv)
    m15["datetime"] = pd.to_datetime(m15["datetime"], utc=True)
    m15 = m15.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    history = {
        "M15": m15,
        "H1": m15.resample("1h").agg(agg).dropna(),
        "H4": m15.resample("4h").agg(agg).dropna(),
        "D1": m15.resample("1D").agg(agg).dropna(),
        "W1": m15.resample("1W").agg(agg).dropna(),
    }
    print(f"Loaded M15: {len(m15)} bars ({m15.index.min()} → {m15.index.max()})")
    print(f"Building dataset (step={args.step}, horizon={args.horizon}, "
          f"threshold={args.threshold})...")
    X, y = build_training_dataset(history, step=args.step, horizon=args.horizon,
                                   threshold=args.threshold,
                                   feature_set=args.feature_set)
    print(f"Dataset: X={X.shape}, y class distribution={y.value_counts().to_dict()}")
    if len(X) < 100:
        print("Not enough samples to train.")
        return 1

    print("Training LightGBM...")
    model, metrics = train_lightgbm(X, y)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print("\nClassification report:\n" + metrics["report"])

    save_model(model, args.out,
               feature_cols=list(X.columns),
               feature_set=args.feature_set)
    print(f"Model saved to {args.out} (feature_set={args.feature_set})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
