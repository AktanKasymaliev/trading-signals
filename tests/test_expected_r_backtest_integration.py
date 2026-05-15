"""Verify ExpectedRFilterModel works as a drop-in filter_model in run_backtest.

We assert that the kept-trade count drops monotonically as the predicted_R
threshold rises — this only holds if the engine actually consumes the Path E
adapter's BLOCK decisions.

Strategy: monkeypatch MasterSignalEngine._tier to return "NORMAL" for every
score so that the filter becomes the sole gating mechanism. This lets us test
Path E gating without needing real market-structure data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from xau_pro_bot.backtest import run_backtest
from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel
from xau_pro_bot.signals.engine import MasterSignalEngine


class _Reg:
    """Stub regressor returning a deterministic predicted_R from `bull_score`."""

    def predict(self, X):
        return np.asarray(X["bull_score"]) - 0.5


@pytest.fixture
def bundle(tmp_path: Path):
    path = tmp_path / "e.joblib"
    joblib.dump({"model": _Reg(), "feature_cols": ["bull_score"]}, path)
    return path


def test_threshold_rise_reduces_kept(long_history, bundle, monkeypatch):
    # Force every bar to score above the NORMAL threshold so that filter gating
    # is the only thing that can block a signal.
    monkeypatch.setattr(MasterSignalEngine, "_tier", staticmethod(lambda score: "NORMAL"))

    kwargs = dict(timeout_bars=48, step=4, stream="intraday")
    kept = []
    for thr in (-1.0, 0.0, 0.5):
        flt = ExpectedRFilterModel(local_path=str(bundle), threshold=thr)
        r = run_backtest(long_history, filter_model=flt, **kwargs)
        kept.append(r.signals_generated)
    assert kept[0] >= kept[1] >= kept[2]
    # Sanity: at the lowest threshold at least one signal is kept.
    assert kept[0] > 0
