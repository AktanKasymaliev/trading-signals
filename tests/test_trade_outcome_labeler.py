from __future__ import annotations

import pandas as pd
import pytest

from xau_pro_bot.models.trade_outcome import (
    Outcome,
    OutcomeClass,
    resolve_outcome_m15,
)


def _bars(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])


def test_tp_hit_buy_returns_positive_R():
    future = _bars([
        [100.0, 101.0, 99.5, 100.5],
        [100.5, 103.0, 100.4, 102.5],
    ])
    out = resolve_outcome_m15(entry=100.0, sl=99.0, tp=102.0,
                               direction="BUY", m15_future=future,
                               timeout_bars=192)
    assert out.hit_tp and not out.hit_sl
    assert out.final_R == pytest.approx(2.0)
    assert out.outcome_class == OutcomeClass.TP
    assert out.bars_to_outcome == 2
    assert out.tp_used == 102.0


def test_sl_hit_buy_returns_minus_one_R():
    future = _bars([[100.0, 100.2, 98.5, 99.0]])
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 192)
    assert out.hit_sl and not out.hit_tp
    assert out.final_R == -1.0
    assert out.outcome_class == OutcomeClass.SL


def test_same_candle_tp_and_sl_resolves_to_SL_first():
    future = _bars([[100.0, 103.0, 98.5, 99.5]])
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 192)
    assert out.outcome_class == OutcomeClass.SAME_CANDLE_SL_FIRST
    assert out.same_candle_conflict is True
    assert out.final_R == -1.0


def test_unresolved_at_timeout():
    future = _bars([[100.0, 100.5, 99.5, 100.1]] * 5)
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 5)
    assert out.outcome_class == OutcomeClass.UNRESOLVED
    assert out.final_R == 0.0
    assert out.bars_to_outcome == 5


def test_sell_outcome_tp_hit():
    future = _bars([[100.0, 100.5, 97.5, 98.0]])
    out = resolve_outcome_m15(100.0, 101.0, 98.0, "SELL", future, 192)
    assert out.hit_tp
    assert out.final_R == pytest.approx(2.0)


def test_mfe_mae_tracked_in_R_units():
    future = _bars([
        [100.0, 100.5, 99.7, 100.2],
        [100.2, 101.5, 99.2, 99.5],
        [99.5, 102.5, 99.0, 102.2],
    ])
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 192)
    assert out.mfe_R >= 2.0
    assert out.mae_R <= -0.8 + 1e-9


def test_zero_risk_raises():
    with pytest.raises(ValueError):
        resolve_outcome_m15(100.0, 100.0, 102.0, "BUY",
                            _bars([[100.0, 101.0, 99.0, 100.5]]), 192)
