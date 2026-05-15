"""Tests for ExpectedRFilterModel adapter."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.expected_r_filter_model import (
    ExpectedRFilterModel, ExpectedRDecision,
)


class _StubRegressor:
    """Minimal sklearn-like regressor returning a fixed predicted_R."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, X):
        return np.full(len(X), self._value)


@pytest.fixture
def bundle_path(tmp_path: Path) -> Path:
    path = tmp_path / "expected_r.joblib"
    joblib.dump(
        {"model": _StubRegressor(0.07), "feature_cols": ["bull_score", "rr"]},
        path,
    )
    return path


def test_keeps_when_predicted_r_above_threshold(bundle_path: Path):
    flt = ExpectedRFilterModel(local_path=str(bundle_path), threshold=0.05)
    feats = pd.DataFrame([{"bull_score": 0.6, "rr": 2.0}])
    out = flt.predict(feats)
    assert out["predicted_r"] == pytest.approx(0.07)
    assert out["decision"] == ExpectedRDecision.KEEP
    assert out["threshold_used"] == 0.05
    assert out["error"] is None


def test_blocks_when_predicted_r_below_threshold(bundle_path: Path):
    flt = ExpectedRFilterModel(local_path=str(bundle_path), threshold=0.10)
    feats = pd.DataFrame([{"bull_score": 0.6, "rr": 2.0}])
    out = flt.predict(feats)
    assert out["decision"] == ExpectedRDecision.BLOCK


def test_missing_columns_are_zero_filled(bundle_path: Path):
    flt = ExpectedRFilterModel(local_path=str(bundle_path), threshold=0.05)
    feats = pd.DataFrame([{"bull_score": 0.6}])  # `rr` missing
    out = flt.predict(feats)
    assert out["error"] is None
    assert out["decision"] == ExpectedRDecision.KEEP


def test_load_failure_returns_neutral_keep(tmp_path: Path):
    flt = ExpectedRFilterModel(local_path=str(tmp_path / "missing.joblib"),
                                threshold=0.05)
    out = flt.predict(pd.DataFrame([{"bull_score": 0.6}]))
    assert out["decision"] == ExpectedRDecision.KEEP
    assert out["error"] is not None
    assert out["predicted_r"] is None
