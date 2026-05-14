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


def test_pick_best_threshold_returns_none_when_no_threshold_meets_min_kept():
    """When every threshold has kept < min_kept, picker must return None
    (NO-GO) instead of falling back to the highest-PF row."""
    sweep = {
        0.10: {"pf": 9.0, "kept":  5, "expectancy": +1.20, "wr": 0.80, "blocked": 0},
        0.15: {"pf": 8.0, "kept":  3, "expectancy": +1.10, "wr": 0.75, "blocked": 0},
        0.20: {"pf": 7.0, "kept":  1, "expectancy": +1.00, "wr": 0.70, "blocked": 0},
    }
    assert pick_best_threshold(sweep, min_kept=100) is None


def test_pick_best_threshold_empty_sweep_returns_none():
    assert pick_best_threshold({}, min_kept=1) is None


def test_tier_filter_result_carries_pnl_r_and_equity_curve():
    """H/I/J rows must report real PF/Expectancy/MaxDD, not zeros.

    The old behaviour copied per-tier rr_values into rr_values but left
    pnl_r and equity_curve empty, so downstream PF/Expectancy reported 0.
    After the fix, both fields are synthesized from the kept tier rr_values.
    """
    r = BacktestResult()
    r.signals_generated = 6
    r.wins = 4
    r.losses = 2
    r.rr_values = [+1.5, +1.5, -1.0, +1.5, -1.0, +1.5]
    r.per_tier = {
        "WEAK":   {"n": 2, "w": 0, "l": 2, "rr": [-1.0, -1.0]},
        "NORMAL": {"n": 2, "w": 2, "l": 0, "rr": [+1.5, +1.5]},
        "STRONG": {"n": 2, "w": 2, "l": 0, "rr": [+1.5, +1.5]},
    }

    out = tier_filter_result(r, keep={"NORMAL", "STRONG"})

    assert out.rr_values == [+1.5, +1.5, +1.5, +1.5]
    assert list(out.pnl_r) == [+1.5, +1.5, +1.5, +1.5]
    assert list(out.equity_curve) == [+1.5, +3.0, +4.5, +6.0]
    assert out.profit_factor > 0.0
    assert out.expectancy > 0.0
