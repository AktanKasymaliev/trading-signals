from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.indicators.swing import find_swing_setup
from xau_pro_bot.signals.swing_analyzer import SwingAnalyzer


def _wide_range_d1(swing_low=2000.0, swing_high=2200.0, n=210) -> pd.DataFrame:
    idx = pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc),
                        periods=n, freq="D")
    rng = np.random.default_rng(7)
    closes = np.concatenate([
        np.linspace(swing_low + 50, swing_high, n // 2),
        swing_high - rng.uniform(0, swing_high - swing_low - 30, n - n // 2),
    ])
    return pd.DataFrame({
        "Open": closes - 1, "High": closes + 3,
        "Low": closes - 3, "Close": closes, "Volume": 1000.0,
    }, index=idx)


def test_find_swing_detects_setup():
    df = _wide_range_d1(swing_low=2000.0, swing_high=2200.0)
    res = find_swing_setup(d1_df=df, h4_df=df)
    assert res is not None
    assert res["type"] in ("1000pip", "500pip")
    assert res["range_pips"] >= 500


def test_no_setup_when_range_too_small():
    n = 210
    idx = pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc),
                        periods=n, freq="D")
    closes = np.linspace(2000.0, 2010.0, n)
    df = pd.DataFrame({
        "Open": closes - 0.5, "High": closes + 1, "Low": closes - 1,
        "Close": closes, "Volume": 1000.0,
    }, index=idx)
    assert find_swing_setup(d1_df=df, h4_df=df) is None


def test_swing_analyzer_returns_signal_result():
    df = _wide_range_d1()
    data = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}
    sig = SwingAnalyzer().analyze(data)
    if sig is not None:
        assert sig["tier"] in ("STRONG", "NORMAL")
        assert sig["tp1"] is not None
        assert "horizon_label" in sig
