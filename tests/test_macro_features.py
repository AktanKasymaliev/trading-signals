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


def test_dxy_csv_adds_slope_and_volatility(long_history, tmp_path):
    """Path F: extended macro feature set must include trend slope + volatility."""
    dxy = _macro_csv(tmp_path, "dxy.csv")
    df = harvest_path_d_samples(
        long_history,
        HarvestConfig(step_h1=4, include_synthetic=True, synth_stride=8,
                      dxy_csv=str(dxy)),
    )
    assert "dxy_trend_slope" in df.columns
    assert "dxy_vol" in df.columns
    assert df["dxy_trend_slope"].isna().sum() == 0
    assert df["dxy_vol"].isna().sum() == 0


def test_us10y_csv_adds_slope(long_history, tmp_path):
    us10y = _macro_csv(tmp_path, "us10y.csv")
    df = harvest_path_d_samples(
        long_history,
        HarvestConfig(step_h1=4, include_synthetic=True, synth_stride=8,
                      us10y_csv=str(us10y)),
    )
    assert "us10y_trend_slope" in df.columns
    assert df["us10y_trend_slope"].isna().sum() == 0


def test_eval_marks_no_macro_data_when_csvs_missing(capsys):
    """Path F: eval must surface NO_MACRO_DATA on stderr when CSVs absent."""
    from scripts.eval_path_d import _check_macro_csvs
    assert _check_macro_csvs(None, None) is False
    captured = capsys.readouterr()
    assert "NO_MACRO_DATA" in captured.err


def test_eval_marks_no_macro_data_when_one_csv_missing(tmp_path, capsys):
    from scripts.eval_path_d import _check_macro_csvs
    real = tmp_path / "dxy.csv"
    real.write_text("timestamp,close\n2024-01-01,100\n")
    assert _check_macro_csvs(str(real), None) is False
    captured = capsys.readouterr()
    assert "NO_MACRO_DATA" in captured.err
    assert "us10y" in captured.err


def test_eval_accepts_both_macro_csvs(tmp_path):
    from scripts.eval_path_d import _check_macro_csvs
    dxy = tmp_path / "dxy.csv"
    us10y = tmp_path / "us10y.csv"
    dxy.write_text("timestamp,close\n2024-01-01,100\n")
    us10y.write_text("timestamp,close\n2024-01-01,4.5\n")
    assert _check_macro_csvs(str(dxy), str(us10y)) is True
