from __future__ import annotations

from xau_pro_bot.backtest import BacktestResult
from scripts.eval_path_d import (
    pick_best_threshold,
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
