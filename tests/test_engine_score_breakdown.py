from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.signals.engine import MasterSignalEngine


def _synthetic_history(n=400, seed=0):
    rng = np.random.default_rng(seed)
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


def test_engine_return_includes_bull_and_bear_scores():
    eng = MasterSignalEngine()
    sig = eng.analyze(_synthetic_history())
    assert "bull_score" in sig
    assert "bear_score" in sig
    assert isinstance(sig["bull_score"], (int, float))
    assert isinstance(sig["bear_score"], (int, float))
    assert sig["score"] == int(max(sig["bull_score"], sig["bear_score"]))
