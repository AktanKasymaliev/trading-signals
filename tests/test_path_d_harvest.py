from __future__ import annotations

import pytest

from xau_pro_bot.models.path_d_harvest import (
    HarvestConfig,
    harvest_path_d_samples,
)



def test_harvest_baseline_only_emits_rows_with_outcome_metadata(long_history):
    cfg = HarvestConfig(step_h1=4, timeout_m15=192, include_synthetic=False)
    df = harvest_path_d_samples(long_history, cfg)
    if df.empty:
        pytest.skip("synthetic history did not produce baseline signals")
    required = {"entry", "sl", "tp_used", "direction", "tier",
                "bull_score", "bear_score", "score_gap",
                "outcome_class", "final_R", "mfe_R", "mae_R",
                "bars_to_outcome", "baseline_sample", "is_synthetic"}
    assert required.issubset(df.columns)
    assert df["baseline_sample"].all()
    assert (~df["is_synthetic"].astype(bool)).all()


def test_harvest_with_synthetic_adds_synthetic_rows(long_history):
    cfg = HarvestConfig(step_h1=4, timeout_m15=192,
                        include_synthetic=True, synth_stride=8,
                        synth_atr_sl=1.5, synth_rr=2.0)
    df = harvest_path_d_samples(long_history, cfg)
    if df.empty:
        pytest.skip("no rows harvested")
    assert df["is_synthetic"].astype(bool).any()
    synth = df[df["is_synthetic"].astype(bool)]
    assert (~synth["baseline_sample"]).all()
    assert synth["outcome_class"].notna().all()


def test_same_candle_conflicts_are_counted(long_history):
    cfg = HarvestConfig(step_h1=4, timeout_m15=192, include_synthetic=False)
    df = harvest_path_d_samples(long_history, cfg)
    if df.empty:
        pytest.skip("empty")
    assert "outcome_class" in df.columns
    counts = df["outcome_class"].value_counts().to_dict()
    assert set(counts.keys()).issubset({"TP", "SL", "UNRESOLVED",
                                         "SAME_CANDLE_SL_FIRST"})
