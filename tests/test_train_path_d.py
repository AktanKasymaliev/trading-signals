from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.train_path_d import (
    split_time_70_15_15,
    train_directional,
    train_filter,
)


def _synthetic_dataset(n=600, seed=0, with_synth=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    feats = {f"f{i}": rng.normal(0, 1, n) for i in range(20)}
    df = pd.DataFrame(feats, index=idx)
    df["is_synthetic"] = (rng.random(n) < 0.3).astype(int) if with_synth else 0
    df["baseline_sample"] = df["is_synthetic"] == 0
    df["label_directional"] = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
    df["label_filter"] = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    return df


def test_split_70_15_15_time_based_preserves_order():
    df = _synthetic_dataset()
    tr, va, te = split_time_70_15_15(df)
    assert tr.index.max() < va.index.min()
    assert va.index.max() < te.index.min()
    n = len(df)
    assert len(tr) == int(n * 0.70)
    assert len(va) == int(n * 0.15)
    assert len(te) == n - len(tr) - len(va)


def test_train_directional_a1_uses_baseline_only_rows():
    df = _synthetic_dataset(with_synth=True)
    model, metrics = train_directional(df, variant="A1")
    assert metrics["n_train"] + metrics["n_val"] + metrics["n_test"] <= (df["baseline_sample"]).sum()
    assert set(model.classes_).issubset({-1, 0, 1})


def test_train_directional_a2_includes_synthetic_rows():
    df = _synthetic_dataset(with_synth=True)
    _, metrics_a1 = train_directional(df, variant="A1")
    _, metrics_a2 = train_directional(df, variant="A2")
    a1_total = metrics_a1["n_train"] + metrics_a1["n_val"] + metrics_a1["n_test"]
    a2_total = metrics_a2["n_train"] + metrics_a2["n_val"] + metrics_a2["n_test"]
    assert a2_total >= a1_total


def test_train_filter_binary_classes():
    df = _synthetic_dataset()
    model, metrics = train_filter(df)
    assert set(model.classes_).issubset({0, 1})
    assert metrics["n_train"] > 0 and metrics["n_test"] > 0
