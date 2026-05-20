"""Tests for AI NEUTRAL action when confidence is below minimum threshold."""

from __future__ import annotations

import pytest

from xau_pro_bot.models.ai_explanation import derive_action
from xau_pro_bot.models.calibration import ai_prediction_to_adjustment


@pytest.fixture(autouse=True)
def _clear_ai_env(monkeypatch):
    for name in (
        "AI_MIN_CONFIDENCE", "AI_NO_TRADE_THRESHOLD",
        "AI_SCORE_BONUS", "AI_STRONG_SCORE_BONUS",
        "AI_STRONG_CONFIDENCE", "AI_CONFLICT_PENALTY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_calibration_flags_low_confidence():
    adj = ai_prediction_to_adjustment(
        {"direction": "BUY", "confidence": 0.40}, "BUY",
    )
    assert adj["ai_low_confidence"] is True
    assert "below minimum" in adj["reason"]
    assert "deterministic signal unchanged" in adj["reason"]
    assert adj["score_delta_buy"] == 0
    assert adj["block_signal"] is False


def test_calibration_does_not_flag_normal_confidence():
    adj = ai_prediction_to_adjustment(
        {"direction": "BUY", "confidence": 0.70}, "BUY",
    )
    assert adj["ai_low_confidence"] is False


def test_derive_action_returns_neutral_when_low_confidence():
    action = derive_action(
        ai_enabled=True, ai_blocked=False,
        ai_direction="BUY", deterministic_direction="BUY",
        ai_low_confidence=True,
    )
    assert action == "NEUTRAL"


def test_derive_action_still_keep_when_normal_confidence():
    action = derive_action(
        ai_enabled=True, ai_blocked=False,
        ai_direction="BUY", deterministic_direction="BUY",
        ai_low_confidence=False,
    )
    assert action == "KEEP"


def test_derive_action_block_takes_precedence_over_neutral():
    action = derive_action(
        ai_enabled=True, ai_blocked=True,
        ai_direction="NO_TRADE", deterministic_direction="BUY",
        ai_low_confidence=True,
    )
    assert action == "BLOCK"


def test_ai_gate_propagates_low_confidence(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "false")
    import pandas as pd
    from xau_pro_bot.signals import ai_gate as ai_gate_mod
    from xau_pro_bot.signals.ai_gate import AIExplanationGate

    class _LowConfModel:
        feature_set = "legacy"

        def predict(self, _features):
            return {"direction": "BUY", "confidence": 0.30}

    # Bypass feature-completeness gate so we exercise the calibration branch.
    monkeypatch.setattr(
        ai_gate_mod, "build_ai_features",
        lambda data: (pd.DataFrame([{}]), True),
    )

    gate = AIExplanationGate(ai_enabled=True, ai_model=_LowConfModel())
    fields = gate.evaluate({"W1": None}, "BUY")
    assert fields["ai_low_confidence"] is True
    explanation = gate.build_explanation(
        fields, "BUY", "STRONG", {"penalties": []},
    )
    assert explanation["ai_action"] == "NEUTRAL"
