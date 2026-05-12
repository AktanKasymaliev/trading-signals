from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.models.features import build_ai_features


def _empty_tfs() -> dict[str, pd.DataFrame]:
    return {tf: pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            for tf in ("W1", "D1", "H4", "H1", "M15")}


def test_build_ai_features_returns_tuple_with_complete_flag(all_tfs):
    features, complete = build_ai_features(all_tfs)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(complete, bool)
    assert complete is True


def test_build_ai_features_marks_incomplete_when_h1_missing():
    features, complete = build_ai_features(_empty_tfs())
    assert complete is False
    # all features still finite (imputed)
    assert features.isna().sum().sum() == 0


def test_build_ai_features_imputes_returns_with_zero():
    df = pd.DataFrame({
        "Open": [1.0], "High": [1.0], "Low": [1.0],
        "Close": [1.0], "Volume": [1.0],
    }, index=pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC"))
    tfs = {tf: df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}

    features, _ = build_ai_features(tfs)

    assert features["return_m15_1"].iloc[0] == 0.0
    assert np.isfinite(features.values).all()
