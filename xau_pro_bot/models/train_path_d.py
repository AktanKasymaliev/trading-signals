"""Path D trainer: time-split + LightGBM with conservative anti-overfit params.

Trains 3 artifacts:
- directional_a1 (baseline-only samples, 3 classes BUY/SELL/NO_TRADE)
- directional_a2 (baseline + synthetic, 3 classes)
- filter         (baseline-only samples, 2 classes GOOD/BAD)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_NON_FEATURE_COLS = {
    "entry", "sl", "tp_used", "direction", "tier",
    "outcome_class", "final_R", "mfe_R", "mae_R", "bars_to_outcome",
    "label_directional", "label_filter", "baseline_sample",
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _NON_FEATURE_COLS and df[c].dtype.kind in "fiub"]


def split_time_70_15_15(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    n = len(df)
    i_tr = int(n * 0.70)
    i_va = i_tr + int(n * 0.15)
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]


def _lgb_params(num_class: int | None) -> dict:
    p = dict(
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_data_in_leaf=120,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",
        n_estimators=600,
        n_jobs=-1, verbose=-1,
        random_state=42,
    )
    if num_class is None:
        p["objective"] = "binary"
    else:
        p["objective"] = "multiclass"
        p["num_class"] = num_class
    return p


def _fit_lgb(X_tr, y_tr, X_va, y_va, params: dict):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(40)])
    return model


def _metrics(model, X_te, y_te) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                  precision_recall_fscore_support)
    pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "report": classification_report(y_te, pred, zero_division=0),
    }


def train_directional(df: pd.DataFrame, *, variant: Literal["A1", "A2"]):
    data = df if variant == "A2" else df[df["baseline_sample"]]
    data = data.dropna(subset=["label_directional"])
    tr, va, te = split_time_70_15_15(data)
    fcols = _feature_cols(data)
    X_tr, y_tr = tr[fcols], tr["label_directional"].astype(int)
    X_va, y_va = va[fcols], va["label_directional"].astype(int)
    X_te, y_te = te[fcols], te["label_directional"].astype(int)
    model = _fit_lgb(X_tr, y_tr, X_va, y_va, _lgb_params(num_class=3))
    m = _metrics(model, X_te, y_te)
    m.update({"n_train": len(tr), "n_val": len(va), "n_test": len(te),
              "feature_cols": fcols, "variant": variant})
    return model, m


def train_filter(df: pd.DataFrame):
    data = df[df["baseline_sample"]].dropna(subset=["label_filter"])
    tr, va, te = split_time_70_15_15(data)
    fcols = _feature_cols(data)
    X_tr, y_tr = tr[fcols], tr["label_filter"].astype(int)
    X_va, y_va = va[fcols], va["label_filter"].astype(int)
    X_te, y_te = te[fcols], te["label_filter"].astype(int)
    model = _fit_lgb(X_tr, y_tr, X_va, y_va, _lgb_params(num_class=None))
    m = _metrics(model, X_te, y_te)
    m.update({"n_train": len(tr), "n_val": len(va), "n_test": len(te),
              "feature_cols": fcols})
    return model, m


def save_model(model, feature_cols: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)
