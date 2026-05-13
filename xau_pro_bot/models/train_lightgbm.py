"""LightGBM trainer for the AI signal layer.

Builds a 3-class classifier (BUY=1, NO_TRADE=0, SELL=-1) using our internal
29-feature builder and forward-return labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xau_pro_bot.models.features import build_ai_features

log = logging.getLogger(__name__)


def label_forward_returns(close: pd.Series, horizon: int = 16,
                          threshold: float = 0.003) -> pd.Series:
    """Forward-return labels: +1 if return > threshold over `horizon` bars,
    -1 if < -threshold, 0 otherwise. Last `horizon` bars get NaN."""
    fwd = close.shift(-horizon)
    ret = (fwd - close) / close
    labels = pd.Series(0.0, index=close.index, dtype="float64")
    labels[ret > threshold] = 1.0
    labels[ret < -threshold] = -1.0
    labels[ret.isna()] = np.nan
    return labels


def build_training_dataset(history: dict[str, pd.DataFrame], *,
                           step: int = 8, horizon: int = 16,
                           threshold: float = 0.003) -> tuple[pd.DataFrame, pd.Series]:
    """Walk M15 bars, build features at each cutoff, label by forward return."""
    m15 = history["M15"]
    labels_full = label_forward_returns(m15["Close"], horizon=horizon, threshold=threshold)

    feature_rows: list[pd.DataFrame] = []
    label_values: list[int] = []
    indices: list[pd.Timestamp] = []

    start = max(800, horizon)
    for i in range(start, len(m15) - horizon, step):
        cutoff = m15.index[i]
        slice_data = {
            "M15": m15.iloc[max(0, i - 720):i + 1],
            "H1":  history["H1"].loc[:cutoff].tail(720),
            "H4":  history["H4"].loc[:cutoff].tail(720),
            "D1":  history["D1"].loc[:cutoff].tail(720),
            "W1":  history["W1"].loc[:cutoff].tail(720),
        }
        try:
            features, complete = build_ai_features(slice_data)
        except Exception:
            continue
        if not complete:
            continue
        y_val = labels_full.iloc[i]
        if pd.isna(y_val):
            continue
        feature_rows.append(features)
        label_values.append(int(y_val))
        indices.append(cutoff)

    if not feature_rows:
        return pd.DataFrame(), pd.Series(dtype="int64")

    X = pd.concat(feature_rows, ignore_index=True)
    X.index = pd.Index(indices, name="datetime")
    y = pd.Series(label_values, index=X.index, name="label", dtype="int64")
    return X, y


def train_lightgbm(X: pd.DataFrame, y: pd.Series, *,
                   test_size: float = 0.2,
                   params: dict | None = None):
    """Time-based 80/20 train/test split, train LGB classifier, return (model, metrics)."""
    import lightgbm as lgb
    from sklearn.metrics import (accuracy_score, classification_report,
                                  precision_recall_fscore_support)

    n = len(X)
    cut = int(n * (1 - test_size))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]

    default_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(30)])

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="macro",
                                                  zero_division=0)
    report = classification_report(y_te, y_pred, zero_division=0)
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "report": report,
    }
    return model, metrics


def save_model(model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
