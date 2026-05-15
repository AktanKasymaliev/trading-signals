from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.backtest import run_backtest
from xau_pro_bot.models.trade_filter_model import TradeFilterModel


class _StubFilter:
    classes_ = [0, 1]
    def predict_proba(self, X): return [[0.9, 0.1]]


@pytest.fixture
def history():
    rng = np.random.default_rng(5)
    n = 600
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


def test_run_backtest_accepts_filter_model(tmp_path, history):
    fp = tmp_path / "f.joblib"
    joblib.dump({"model": _StubFilter(), "feature_cols": []}, fp)
    filt = TradeFilterModel(local_path=str(fp), threshold=0.55)
    base = run_backtest(history, timeout_bars=24, step=4)
    blocked = run_backtest(history, timeout_bars=24, step=4,
                           filter_model=filt)
    assert blocked.signals_generated <= base.signals_generated
