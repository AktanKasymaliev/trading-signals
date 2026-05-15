"""Path F end-to-end smoke: synthetic features → mini-LightGBM →
ExpectedRFilterModel align/predict. Must finish well under 30s.
Catches integration breakage in CI without needing real market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_path_f_pipeline_smoke(tmp_path):
    pytest.importorskip("lightgbm")
    import lightgbm as lgb
    import joblib

    from xau_pro_bot.models.features_stationary import (
        STATIONARY_FEATURES, build_stationary_features,
    )
    from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(200, len(STATIONARY_FEATURES))),
                     columns=STATIONARY_FEATURES)
    y = X["close_vs_ema50_atr"] * 0.4 + rng.normal(scale=0.1, size=200)

    model = lgb.LGBMRegressor(n_estimators=30, num_leaves=7,
                              min_data_in_leaf=10, verbose=-1)
    model.fit(X, y)
    bundle_path = tmp_path / "smoke_stationary.joblib"
    joblib.dump({
        "model": model,
        "feature_cols": list(STATIONARY_FEATURES),
        "feature_set": "stationary",
    }, bundle_path)

    f = ExpectedRFilterModel(str(bundle_path), threshold=0.0)
    feats, _ = build_stationary_features(_synthetic_tfs())
    out = f.predict(feats)
    assert out["error"] is None
    assert isinstance(out["predicted_r"], float)
    assert out["decision"].value in {"KEEP", "BLOCK"}
    assert f.feature_set == "stationary"


def _synthetic_tfs() -> dict[str, pd.DataFrame]:
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.linspace(1800.0, 2000.0, n)
    base = pd.DataFrame({
        "Open":   closes - 0.5,
        "High":   closes + 1.0,
        "Low":    closes - 1.0,
        "Close":  closes,
        "Volume": np.ones(n),
    }, index=idx)
    m15_idx = pd.date_range(idx[0], idx[-1], freq="15min", tz="UTC")
    m15_closes = np.linspace(closes[0], closes[-1], len(m15_idx))
    m15 = pd.DataFrame({
        "Open":   m15_closes - 0.2,
        "High":   m15_closes + 0.5,
        "Low":    m15_closes - 0.5,
        "Close":  m15_closes,
        "Volume": np.ones(len(m15_idx)),
    }, index=m15_idx)
    h4 = base.resample("4h").agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last",
                                  "Volume": "sum"}).dropna()
    d1 = base.resample("1D").agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last",
                                  "Volume": "sum"}).dropna()
    return {"M15": m15, "H1": base, "H4": h4, "D1": d1}
