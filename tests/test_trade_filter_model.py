from __future__ import annotations

import joblib
import pandas as pd
import pytest

from xau_pro_bot.models.trade_filter_model import (
    FilterDecision,
    TradeFilterModel,
)


class _StubBinary:
    classes_ = [0, 1]

    def __init__(self, good_prob: float):
        self._p = good_prob

    def predict_proba(self, X):
        return [[1 - self._p, self._p]]


def _dump(tmp_path, good_prob: float, fcols=("f0",)):
    p = tmp_path / "f.joblib"
    joblib.dump({"model": _StubBinary(good_prob), "feature_cols": list(fcols)}, p)
    return p


def test_keep_when_good_prob_above_threshold(tmp_path):
    p = _dump(tmp_path, good_prob=0.80)
    m = TradeFilterModel(local_path=str(p), threshold=0.55)
    pred = m.predict(pd.DataFrame([{"f0": 0.1}]))
    assert pred["good_prob"] == pytest.approx(0.80)
    assert pred["bad_prob"]  == pytest.approx(0.20)
    assert pred["decision"]  == FilterDecision.KEEP
    assert pred["threshold_used"] == 0.55


def test_block_when_good_prob_below_threshold(tmp_path):
    p = _dump(tmp_path, good_prob=0.30)
    m = TradeFilterModel(local_path=str(p), threshold=0.55)
    pred = m.predict(pd.DataFrame([{"f0": 0.0}]))
    assert pred["decision"] == FilterDecision.BLOCK


def test_missing_features_filled_with_zero(tmp_path):
    p = _dump(tmp_path, good_prob=0.7, fcols=("f0", "f1", "f2"))
    m = TradeFilterModel(local_path=str(p), threshold=0.5)
    pred = m.predict(pd.DataFrame([{"f0": 1.0}]))
    assert pred["decision"] == FilterDecision.KEEP


def test_load_failure_yields_neutral_keep(tmp_path):
    m = TradeFilterModel(local_path=str(tmp_path / "missing.joblib"),
                          threshold=0.55)
    pred = m.predict(pd.DataFrame([{"f0": 0.0}]))
    assert pred["decision"] == FilterDecision.KEEP
    assert pred["error"] is not None
