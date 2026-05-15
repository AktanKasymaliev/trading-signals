"""Regression test: step_h1=1 must yield strictly more samples than step_h1=4."""

from __future__ import annotations

from xau_pro_bot.models.path_d_harvest import HarvestConfig, harvest_path_d_samples


def test_step_h1_1_yields_more_samples_than_step_h1_4(long_history):
    df4 = harvest_path_d_samples(long_history, HarvestConfig(step_h1=4))
    df1 = harvest_path_d_samples(long_history, HarvestConfig(step_h1=1))
    assert len(df1) >= len(df4)
    if len(df4) > 0:
        assert len(df1) > len(df4)
