"""DXY/US10Y feature wiring. Default OFF preserves bit-identical behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xau_pro_bot.models.path_d_harvest import (
    HarvestConfig, harvest_path_d_samples,
)


def _macro_csv(tmp_path: Path, name: str, periods: int = 8000) -> Path:
    idx = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    val = np.cumsum(np.random.default_rng(1).normal(0, 0.05, periods)) + 100.0
    p = tmp_path / name
    pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "close": val,
    }).to_csv(p, index=False)
    return p


def test_no_csv_means_no_new_columns(long_history):
    df = harvest_path_d_samples(
        long_history,
        HarvestConfig(step_h1=4, include_synthetic=True, synth_stride=8),
    )
    assert not df.empty
    for col in ("dxy_ret_1h", "dxy_ret_4h", "us10y_chg_1h", "us10y_chg_4h"):
        assert col not in df.columns


def test_dxy_csv_adds_dxy_features_only(long_history, tmp_path):
    dxy = _macro_csv(tmp_path, "dxy.csv")
    df = harvest_path_d_samples(
        long_history,
        HarvestConfig(step_h1=4, include_synthetic=True, synth_stride=8,
                      dxy_csv=str(dxy)),
    )
    assert not df.empty
    assert "dxy_ret_1h" in df.columns
    assert "dxy_ret_4h" in df.columns
    assert "us10y_chg_1h" not in df.columns
    assert df[["dxy_ret_1h", "dxy_ret_4h"]].isna().sum().sum() == 0


def test_us10y_csv_adds_us10y_features(long_history, tmp_path):
    us10y = _macro_csv(tmp_path, "us10y.csv")
    df = harvest_path_d_samples(
        long_history,
        HarvestConfig(step_h1=4, include_synthetic=True, synth_stride=8,
                      us10y_csv=str(us10y)),
    )
    assert "us10y_chg_1h" in df.columns
    assert "us10y_chg_4h" in df.columns
    assert df[["us10y_chg_1h", "us10y_chg_4h"]].isna().sum().sum() == 0
