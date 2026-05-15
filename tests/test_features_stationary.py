"""Unit tests for the stationary feature builder used by Path F."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _PicklableConstantRegressor:
    """Module-level so joblib/pickle can round-trip it during artifact tests."""
    def predict(self, X):
        return np.full(len(X), 0.42)


class _PicklableConstantClassifier:
    def predict(self, X):
        return [1] * len(X)


@pytest.fixture()
def synthetic_tfs() -> dict[str, pd.DataFrame]:
    """Synthetic multi-timeframe history with a clear uptrend."""
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.linspace(1800.5, 2100.5, n)
    base = pd.DataFrame({
        "Open":   closes - 0.5,
        "High":   closes + 1.0,
        "Low":    closes - 1.0,
        "Close":  closes,
        "Volume": np.ones(n),
    }, index=idx)
    m15_idx = pd.date_range(idx[0], idx[-1] + pd.Timedelta("45min"),
                            freq="15min", tz="UTC")
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


def test_stationary_features_export_expected_columns():
    from xau_pro_bot.models.features_stationary import STATIONARY_FEATURES
    expected = {
        "close_vs_ema8_atr", "close_vs_ema21_atr", "close_vs_ema50_atr",
        "close_vs_ema200_atr",
        "ema8_vs_ema21_atr", "ema21_vs_ema50_atr", "ema50_vs_ema200_atr",
        "return_m15_1", "return_m15_3", "return_m15_5",
        "return_h1_1", "return_h1_3", "return_h4_1",
        "atr_percentile_h1", "range_vs_atr_m15",
        "distance_to_recent_high_atr", "distance_to_recent_low_atr",
    }
    assert set(STATIONARY_FEATURES) == expected
    assert len(STATIONARY_FEATURES) == 17


def test_build_stationary_features_returns_one_row(synthetic_tfs):
    from xau_pro_bot.models.features_stationary import (
        STATIONARY_FEATURES, build_stationary_features,
    )
    df, complete = build_stationary_features(synthetic_tfs)
    assert complete is True
    assert list(df.columns) == STATIONARY_FEATURES
    assert len(df) == 1


def test_stationary_features_no_raw_price_columns(synthetic_tfs):
    """Path F contract: no absolute-price column ever leaves the builder."""
    from xau_pro_bot.models.features_stationary import build_stationary_features
    import re
    df, _ = build_stationary_features(synthetic_tfs)
    forbidden = re.compile(r"^(close_(m15|h1|h4|d1)|ema(8|21|50|200)_h1)$")
    leaking = [c for c in df.columns if forbidden.match(c)]
    assert leaking == [], f"forbidden raw-price columns leaked: {leaking}"


def test_close_vs_ema_signs_track_trend(synthetic_tfs):
    """In a clean uptrend, close should sit above slower EMAs (positive
    close_vs_emaN_atr) — sanity check that feature direction is right."""
    from xau_pro_bot.models.features_stationary import build_stationary_features
    df, _ = build_stationary_features(synthetic_tfs)
    assert df["close_vs_ema200_atr"].iloc[0] > 0
    assert df["close_vs_ema50_atr"].iloc[0] > 0


def test_build_stationary_features_short_history_marks_incomplete():
    """Empty / too-short data must NOT crash; return complete=False
    and a zero-filled row with the right columns."""
    from xau_pro_bot.models.features_stationary import (
        STATIONARY_FEATURES, build_stationary_features,
    )
    df, complete = build_stationary_features({})
    assert complete is False
    assert list(df.columns) == STATIONARY_FEATURES
    assert len(df) == 1
    assert (df.iloc[0].fillna(0.0) == 0.0).all()


def test_harvester_config_accepts_feature_set_stationary():
    """HarvestConfig.feature_set must accept 'stationary' for Path F."""
    from xau_pro_bot.models.path_d_harvest import HarvestConfig
    cfg = HarvestConfig(feature_set="stationary")
    assert cfg.feature_set == "stationary"
    default = HarvestConfig()
    assert default.feature_set == "legacy"


def test_expected_r_filter_model_reads_feature_set_tag(tmp_path):
    """A model bundle tagged feature_set='stationary' must surface that
    tag on the filter wrapper so eval can dispatch features correctly."""
    import joblib
    import numpy as np

    from xau_pro_bot.models.features_stationary import STATIONARY_FEATURES
    from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel

    bundle_path = tmp_path / "fake_stationary.joblib"
    joblib.dump({
        "model": _PicklableConstantRegressor(),
        "feature_cols": list(STATIONARY_FEATURES),
        "feature_set": "stationary",
    }, bundle_path)

    f = ExpectedRFilterModel(str(bundle_path), threshold=0.05)
    df = pd.DataFrame([{c: 1.0 for c in STATIONARY_FEATURES}])
    out = f.predict(df)
    assert out["error"] is None
    assert out["predicted_r"] == pytest.approx(0.42)
    assert f.feature_set == "stationary"


def test_hf_trading_model_unwraps_path_f_dict_bundle(tmp_path):
    """HFTradingModel must accept Path F dict bundles AND legacy raw models."""
    import joblib

    from xau_pro_bot.models.hf_model import HFTradingModel

    legacy_path = tmp_path / "legacy.joblib"
    joblib.dump(_PicklableConstantClassifier(), legacy_path)
    legacy = HFTradingModel(model_id="", model_type="sklearn",
                            local_path=str(legacy_path))
    legacy._load_sklearn()
    assert legacy.feature_set == "legacy"

    new_path = tmp_path / "new_stationary.joblib"
    joblib.dump({"model": _PicklableConstantClassifier(),
                 "feature_cols": ["a", "b"],
                 "feature_set": "stationary"}, new_path)
    new = HFTradingModel(model_id="", model_type="sklearn",
                         local_path=str(new_path))
    new._load_sklearn()
    assert new.feature_set == "stationary"
    assert new.feature_cols == ["a", "b"]
