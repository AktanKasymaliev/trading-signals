from __future__ import annotations

import argparse

from xau_pro_bot.backtest import BacktestResult
from scripts.eval_path_d import (
    pick_best_threshold,
    run_all_modes,
    tier_filter_result,
)


def _mk(sig=100, wins=40, losses=50, blocked=0, rr_values=None):
    r = BacktestResult()
    r.signals_generated = sig
    r.wins = wins
    r.losses = losses
    r.blocked_signals = blocked
    r.rr_values = list(rr_values or [1.5] * sig)
    return r


def test_pick_best_threshold_prefers_higher_pf_then_more_trades():
    sweep = {
        0.50: {"pf": 0.9, "kept": 200, "expectancy": -0.05, "wr": 0.30, "blocked": 0},
        0.60: {"pf": 1.1, "kept": 140, "expectancy": +0.04, "wr": 0.38, "blocked": 0},
        0.65: {"pf": 1.1, "kept": 120, "expectancy": +0.04, "wr": 0.39, "blocked": 0},
        0.70: {"pf": 1.2, "kept":  40, "expectancy": +0.05, "wr": 0.40, "blocked": 0},
    }
    # min_kept_floor=60 excludes 0.70; tie between 0.60 and 0.65 on PF, more-kept wins → 0.60
    assert pick_best_threshold(sweep, min_kept=60) == 0.60


def test_run_all_modes_accepts_path_d_filter_calibrated_param():
    """Verify run_all_modes accepts path_d_filter_calibrated kwarg (None = no-op)."""
    import inspect
    sig = inspect.signature(run_all_modes)
    assert "path_d_filter_calibrated" in sig.parameters
    assert sig.parameters["path_d_filter_calibrated"].default is None


def test_k_mode_key_is_recognized_in_non_mode_keys():
    """K_path_d_filter_calibrated should NOT appear in _NON_MODE_KEYS."""
    from scripts.eval_path_d import _NON_MODE_KEYS
    assert "K_path_d_filter_calibrated" not in _NON_MODE_KEYS
    assert "threshold_sweeps" in _NON_MODE_KEYS
    assert "chosen_thresholds" in _NON_MODE_KEYS


def test_tier_filter_result_drops_below_tier():
    r = _mk(sig=10, wins=4, losses=6)
    r.per_tier = {
        "WEAK":   {"n": 5, "w": 1, "l": 4},
        "NORMAL": {"n": 3, "w": 2, "l": 1},
        "STRONG": {"n": 2, "w": 1, "l": 1},
    }
    out = tier_filter_result(r, keep={"NORMAL", "STRONG"})
    assert out.signals_generated == 5
    assert out.wins == 3
    assert out.losses == 2
