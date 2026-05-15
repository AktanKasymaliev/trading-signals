from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.models.features import REQUIRED_AI_FEATURES, build_ai_features


def test_build_ai_features_returns_one_row(all_tfs):
    features, _ = build_ai_features(all_tfs)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1
    assert list(features.columns) == REQUIRED_AI_FEATURES


def test_build_ai_features_does_not_mutate_input(all_tfs):
    before = {tf: df.copy(deep=True) for tf, df in all_tfs.items()}

    build_ai_features(all_tfs)  # noqa: F841 — return value intentionally ignored

    for tf, df in all_tfs.items():
        pd.testing.assert_frame_equal(df, before[tf])


def test_build_ai_features_handles_short_dfs(short_df):
    tfs = {tf: short_df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}

    features, _ = build_ai_features(tfs)

    assert len(features) == 1
    assert set(REQUIRED_AI_FEATURES).issubset(features.columns)
    assert np.isfinite(features["hour_utc"].iloc[0])
    assert np.isfinite(features["day_of_week"].iloc[0])


def test_build_ai_features_handles_missing_optional_indicator_columns(uptrend_df):
    tfs = {
        tf: uptrend_df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for tf in ("W1", "D1", "H4", "H1", "M15")
    }

    features, _ = build_ai_features(tfs)

    assert "rsi_h1" in features.columns
    assert "atr_h1" in features.columns
    assert features["ema8_above_ema21_h1"].iloc[0] in (-1, 0, 1)
    assert features["pd_zone_h4_encoded"].iloc[0] in (-1, 0, 1)
    assert features["wyckoff_bias_h4_encoded"].iloc[0] in (-1, 0, 1)
