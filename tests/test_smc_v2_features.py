from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.smc_v2_features import (
    REQUIRED_SMC_V2_FEATURES,
    build_smc_v2_features,
)


@pytest.fixture
def long_m15_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    base = 2000.0 + np.cumsum(np.random.normal(0, 1.0, n))
    return pd.DataFrame({
        "Open": base,
        "High": base + 2,
        "Low": base - 2,
        "Close": base + np.random.normal(0, 0.5, n),
        "Volume": np.random.randint(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))


def test_build_returns_dataframe_with_21_features(long_m15_df):
    df, complete = build_smc_v2_features({"M15": long_m15_df})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == REQUIRED_SMC_V2_FEATURES
    assert len(REQUIRED_SMC_V2_FEATURES) == 21
    assert complete is True


def test_required_features_order_matches_model():
    expected = [
        "Close", "High", "Low", "Open",
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_middle", "BB_lower",
        "FVG_Size", "FVG_Type", "OB_Type",
        "Close_lag1", "Close_lag2", "Close_lag3",
    ]
    assert REQUIRED_SMC_V2_FEATURES == expected


def test_lag_features_match_previous_closes(long_m15_df):
    df, _ = build_smc_v2_features({"M15": long_m15_df})
    closes = long_m15_df["Close"].iloc[-4:].tolist()
    assert df["Close_lag1"].iloc[0] == pytest.approx(closes[-2])
    assert df["Close_lag2"].iloc[0] == pytest.approx(closes[-3])
    assert df["Close_lag3"].iloc[0] == pytest.approx(closes[-4])


def test_fvg_type_and_ob_type_use_label_encoding(long_m15_df):
    # bearish=0, bullish=1, none=2 — values must be in {0,1,2}
    df, _ = build_smc_v2_features({"M15": long_m15_df})
    assert df["FVG_Type"].iloc[0] in (0, 1, 2)
    assert df["OB_Type"].iloc[0] in (0, 1, 2)


def test_incomplete_when_too_few_bars():
    short = pd.DataFrame({
        "Open": [1.0] * 20, "High": [1.0] * 20, "Low": [1.0] * 20,
        "Close": [1.0] * 20, "Volume": [1.0] * 20,
    }, index=pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC"))
    df, complete = build_smc_v2_features({"M15": short})
    assert complete is False
    assert df.isna().sum().sum() == 0


def test_no_nan_in_output(long_m15_df):
    df, _ = build_smc_v2_features({"M15": long_m15_df})
    assert np.isfinite(df.values).all()
