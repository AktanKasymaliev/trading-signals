from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.train_lightgbm import (
    build_training_dataset,
    label_forward_returns,
)


def test_label_forward_returns_classifies_correctly():
    closes = pd.Series([100.0] * 17, index=pd.date_range("2026-01-01", periods=17, freq="15min", tz="UTC"))
    closes_buy = closes.copy(); closes_buy.iloc[16] = 100.5
    labels_buy = label_forward_returns(closes_buy, horizon=16, threshold=0.003)
    assert labels_buy.iloc[0] == 1

    closes_sell = closes.copy(); closes_sell.iloc[16] = 99.5
    labels_sell = label_forward_returns(closes_sell, horizon=16, threshold=0.003)
    assert labels_sell.iloc[0] == -1

    labels_neutral = label_forward_returns(closes, horizon=16, threshold=0.003)
    assert labels_neutral.iloc[0] == 0


def test_label_forward_returns_nan_at_tail():
    closes = pd.Series([100.0] * 20, index=pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC"))
    labels = label_forward_returns(closes, horizon=16, threshold=0.003)
    assert pd.isna(labels.iloc[-1])


def test_build_training_dataset_returns_X_y_with_finite_features():
    np.random.seed(0)
    n = 1200
    base = 2000.0 + np.cumsum(np.random.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2,
        "Close": base + np.random.normal(0, 0.3, n),
        "Volume": np.random.randint(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    history = {
        "M15": m15,
        "H1": m15.resample("1h").agg(agg).dropna(),
        "H4": m15.resample("4h").agg(agg).dropna(),
        "D1": m15.resample("1D").agg(agg).dropna(),
        "W1": m15.resample("1W").agg(agg).dropna(),
    }
    X, y = build_training_dataset(history, step=8, horizon=16, threshold=0.003)
    assert len(X) == len(y)
    assert len(X) > 0
    assert np.isfinite(X.values).all()
    assert set(y.unique()).issubset({-1, 0, 1})
