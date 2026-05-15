from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.config import load_ai_config
from xau_pro_bot.models.trade_filter_model import TradeFilterModel
from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.hybrid_policy import HybridThresholds


class _StubFilter:
    classes_ = [0, 1]
    def __init__(self, good): self._g = good
    def predict_proba(self, X): return [[1 - self._g, self._g]]


@pytest.fixture
def history():
    rng = np.random.default_rng(3)
    n = 500
    base = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2,
        "Close": base + rng.normal(0, 0.3, n),
        "Volume": rng.integers(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def test_config_includes_path_d_keys(monkeypatch):
    monkeypatch.setenv("AI_PATH_D_FILTER_PATH", "/tmp/f.joblib")
    monkeypatch.setenv("AI_HYBRID_MODE", "filter")
    monkeypatch.setenv("AI_FILTER_THRESHOLD_NORMAL", "0.60")
    cfg = load_ai_config()
    assert cfg["path_d_filter_path"] == "/tmp/f.joblib"
    assert cfg["hybrid_mode"] == "filter"
    assert cfg["filter_threshold_normal"] == 0.60


def test_filter_block_marks_signal_as_blocked(tmp_path, history):
    fp = tmp_path / "f.joblib"
    joblib.dump({"model": _StubFilter(good=0.10),
                 "feature_cols": ["dummy"]}, fp)
    filt = TradeFilterModel(local_path=str(fp), threshold=0.55)
    eng = MasterSignalEngine(filter_model=filt,
                             hybrid_thresholds=HybridThresholds())
    sig = eng.analyze(history)
    if sig["tier"] == "NO_SIGNAL":
        pytest.skip("baseline produced no signal in this synthetic slice")
    if sig["tier"] in {"WEAK", "NORMAL"}:
        assert sig.get("ai_blocked") is True
        assert sig["tier"] == "NO_SIGNAL"


def test_engine_without_filter_unchanged(history):
    # Pin AI disabled so the assertion is not at the mercy of .env state.
    eng = MasterSignalEngine(ai_enabled=False)
    sig = eng.analyze(history)
    assert "bull_score" in sig
    assert not sig.get("ai_blocked", False)
