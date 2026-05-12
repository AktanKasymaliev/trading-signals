from __future__ import annotations

from xau_pro_bot.models.calibration import ai_prediction_to_adjustment


def _pred(direction: str, confidence: float) -> dict:
    return {
        "direction": direction,
        "confidence": confidence,
        "prob_buy": None,
        "prob_sell": None,
        "prob_no_trade": None,
    }


def test_ai_agrees_buy_strong_confidence_adds_strong_bonus():
    adj = ai_prediction_to_adjustment(_pred("BUY", 0.76), "BUY")

    assert adj["score_delta_buy"] == 12
    assert adj["score_delta_sell"] == 0
    assert adj["block_signal"] is False
    assert adj["ai_direction"] == "BUY"
    assert adj["ai_confidence"] == 0.76
    assert "agrees" in adj["reason"]


def test_ai_agrees_sell_normal_confidence_adds_normal_bonus():
    adj = ai_prediction_to_adjustment(_pred("SELL", 0.66), "SELL")

    assert adj["score_delta_sell"] == 8
    assert adj["score_delta_buy"] == 0
    assert adj["block_signal"] is False


def test_ai_conflicts_with_buy_penalizes_buy():
    adj = ai_prediction_to_adjustment(_pred("SELL", 0.70), "BUY")

    assert adj["score_delta_buy"] == -10
    assert adj["score_delta_sell"] == 0
    assert adj["block_signal"] is False
    assert "conflicts" in adj["reason"]


def test_ai_no_trade_blocks_signal():
    adj = ai_prediction_to_adjustment(_pred("NO_TRADE", 0.65), "BUY")

    assert adj["block_signal"] is True
    assert adj["score_delta_buy"] == 0
    assert adj["score_delta_sell"] == 0
    assert "blocked" in adj["reason"]


def test_ai_low_confidence_does_nothing():
    adj = ai_prediction_to_adjustment(_pred("BUY", 0.40), "BUY")

    assert adj["score_delta_buy"] == 0
    assert adj["score_delta_sell"] == 0
    assert adj["block_signal"] is False
    assert "below" in adj["reason"]
