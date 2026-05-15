from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.backtest import run_backtest


def _hist(n: int = 600, seed: int = 2) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    m15 = pd.DataFrame(
        {
            "Open": base,
            "High": base + 2,
            "Low": base - 2,
            "Close": base + rng.normal(0, 0.3, n),
            "Volume": rng.integers(100, 1000, n).astype(float),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
    )
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def test_walk_from_skips_pre_window_cutoffs():
    h = _hist()
    h1 = h["H1"]
    full = run_backtest(h, timeout_bars=24, step=4)
    later = h1.index[int(len(h1) * 0.5)]
    half = run_backtest(h, timeout_bars=24, step=4, walk_from=later)
    assert half.signals_generated <= full.signals_generated


def test_walk_to_skips_post_window_cutoffs():
    h = _hist()
    h1 = h["H1"]
    full = run_backtest(h, timeout_bars=24, step=4)
    cutoff = h1.index[int(len(h1) * 0.5)]
    half = run_backtest(h, timeout_bars=24, step=4, walk_to=cutoff)
    assert half.signals_generated <= full.signals_generated
