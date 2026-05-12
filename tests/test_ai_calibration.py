from __future__ import annotations

import pytest

from xau_pro_bot.models.calibration import ai_prediction_to_adjustment


@pytest.fixture(autouse=True)
def _clear_ai_env(monkeypatch):
    for name in (
        "AI_MIN_CONFIDENCE",
        "AI_STRONG_CONFIDENCE",
        "AI_NO_TRADE_THRESHOLD",
        "AI_SCORE_BONUS",
        "AI_STRONG_SCORE_BONUS",
        "AI_CONFLICT_PENALTY",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_ai_score_bonus_env_override_changes_normal_agreement_bonus(monkeypatch):
    monkeypatch.setenv("AI_SCORE_BONUS", "9")

    adj = ai_prediction_to_adjustment(_pred("BUY", 0.66), "BUY")

    assert adj["score_delta_buy"] == 9


def test_ai_strong_score_bonus_env_override_changes_strong_agreement_bonus(monkeypatch):
    monkeypatch.setenv("AI_STRONG_SCORE_BONUS", "14")

    adj = ai_prediction_to_adjustment(_pred("SELL", 0.76), "SELL")

    assert adj["score_delta_sell"] == 14


def test_ai_conflict_penalty_env_override_changes_conflict_penalty(monkeypatch):
    monkeypatch.setenv("AI_CONFLICT_PENALTY", "11")

    adj = ai_prediction_to_adjustment(_pred("SELL", 0.70), "BUY")

    assert adj["score_delta_buy"] == -11


def test_ai_no_trade_threshold_env_override_changes_blocking_threshold(monkeypatch):
    monkeypatch.setenv("AI_NO_TRADE_THRESHOLD", "0.70")

    below_override = ai_prediction_to_adjustment(_pred("NO_TRADE", 0.65), "BUY")
    at_override = ai_prediction_to_adjustment(_pred("NO_TRADE", 0.70), "BUY")

    assert below_override["block_signal"] is False
    assert at_override["block_signal"] is True
