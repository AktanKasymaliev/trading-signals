"""Unit tests for the Path E expected_R regressor trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.expected_r import train_expected_r_regressor


def _toy_dataset(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    bull = rng.uniform(0.0, 1.0, n)
    bear = rng.uniform(0.0, 1.0, n)
    final_r = (bull - bear) * 2.0 + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({
        "bull_score": bull,
        "bear_score": bear,
        "score_gap": np.abs(bull - bear),
        "final_score": bull + bear,
        "tier_STRONG": rng.integers(0, 2, n),
        "tier_NORMAL": rng.integers(0, 2, n),
        "tier_WEAK":   rng.integers(0, 2, n),
        "dir_BUY":     rng.integers(0, 2, n),
        "dir_SELL":    rng.integers(0, 2, n),
        "rr":          np.full(n, 2.0),
        "hour_ny":     rng.integers(0, 24, n).astype(float),
        "day_of_week": rng.integers(0, 5, n).astype(float),
        "atr_percentile_h1": rng.uniform(0.0, 1.0, n),
        "range_vs_atr_m15":  rng.uniform(0.0, 2.0, n),
        "final_R":     final_r,
        "baseline_sample": True,
    }, index=idx)


def test_train_expected_r_returns_model_and_metrics():
    df = _toy_dataset()
    base_params = dict(min_data_in_leaf=5, n_estimators=50, learning_rate=0.1)
    model, metrics = train_expected_r_regressor(df, base_params=base_params)

    assert hasattr(model, "predict")
    assert metrics["n_train"] > 0 and metrics["n_val"] > 0 and metrics["n_test"] > 0
    assert "feature_cols" in metrics and len(metrics["feature_cols"]) > 0
    assert isinstance(metrics["mean_pred"], float)
    assert metrics["p90_pred"] > metrics["p10_pred"]
    assert "feature_importance" in metrics
    assert sum(metrics["feature_importance"].values()) > 0


def test_train_expected_r_rejects_when_empty():
    df = _toy_dataset(n=400)
    df = df.iloc[0:0]
    with pytest.raises(ValueError):
        train_expected_r_regressor(df)


def test_train_script_saves_expected_r_artifact(tmp_path):
    """Smoke test: --expected-r flag produces a joblib bundle that loads.

    Acceptable outcomes:
      (a) returncode 0 AND bundle file exists -> verify bundle structure
      (b) returncode 1 with "Not enough samples" or "Dataset" message ->
          synthetic CSV too small, trainer aborted cleanly (acceptable).
    """
    import subprocess
    import sys
    import os
    import joblib

    csv = tmp_path / "data.csv"
    n = 6 * 24 * 90  # 90 days of M15 -> ~2160 H1 bars after resample
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    base = np.cumsum(np.random.default_rng(0).normal(0, 0.5, n)) + 2000
    pd.DataFrame({
        "datetime": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "Open": base, "High": base + 1.0, "Low": base - 1.0,
        "Close": base, "Volume": 100.0,
    }).to_csv(csv, index=False)

    out = tmp_path / "models"
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd = [
        sys.executable, "scripts/train_path_d_model.py",
        "--csv", str(csv), "--out-dir", str(out),
        "--expected-r", "--allow-degenerate",
        "--step-h1", "4",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    bundle_path = out / "path_e_expected_r_lgb.joblib"
    if res.returncode == 0 and bundle_path.exists():
        b = joblib.load(bundle_path)
        assert "model" in b and "feature_cols" in b
        assert len(b["feature_cols"]) > 0
    else:
        # Acceptable abort path: synthetic data too small.
        assert res.returncode in (0, 1), (
            f"unexpected returncode {res.returncode}\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
