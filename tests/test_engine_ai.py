from __future__ import annotations

from typing import Any

import pandas as pd

from xau_pro_bot.signals.engine import MasterSignalEngine


class MockAIModel:
    def __init__(self, prediction: dict[str, Any]) -> None:
        self.prediction = prediction
        self.calls = 0
        self.seen_features: pd.DataFrame | None = None

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        self.calls += 1
        self.seen_features = features
        return self.prediction


def _ai_fields(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ai_enabled": result["ai_enabled"],
        "ai_direction": result["ai_direction"],
        "ai_confidence": result["ai_confidence"],
        "ai_reason": result["ai_reason"],
        "ai_blocked": result["ai_blocked"],
        "ai_score_delta_buy": result["ai_score_delta_buy"],
        "ai_score_delta_sell": result["ai_score_delta_sell"],
    }


def _all_tfs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {tf: df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}


def test_ai_disabled_keeps_baseline_scores_and_adds_disabled_fields(all_tfs):
    baseline = MasterSignalEngine(ai_enabled=False).analyze(all_tfs)
    result = MasterSignalEngine(ai_enabled=False).analyze(all_tfs)

    assert result["direction"] == baseline["direction"]
    assert result["tier"] == baseline["tier"]
    assert result["score"] == baseline["score"]
    assert _ai_fields(result) == {
        "ai_enabled": False,
        "ai_direction": None,
        "ai_confidence": None,
        "ai_reason": None,
        "ai_blocked": False,
        "ai_score_delta_buy": 0,
        "ai_score_delta_sell": 0,
    }


def test_default_constructor_ai_disabled_does_not_instantiate_model(
    monkeypatch,
    all_tfs,
):
    monkeypatch.delenv("AI_ENABLED", raising=False)

    def fail_if_instantiated(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("HFTradingModel should not be instantiated when AI is disabled")

    monkeypatch.setattr(
        "xau_pro_bot.signals.ai_gate.HFTradingModel",
        fail_if_instantiated,
    )

    result = MasterSignalEngine().analyze(all_tfs)

    assert _ai_fields(result) == {
        "ai_enabled": False,
        "ai_direction": None,
        "ai_confidence": None,
        "ai_reason": None,
        "ai_blocked": False,
        "ai_score_delta_buy": 0,
        "ai_score_delta_sell": 0,
    }


def test_env_disabled_false_does_not_instantiate_model(monkeypatch, all_tfs):
    monkeypatch.setenv("AI_ENABLED", "false")
    monkeypatch.setenv("AI_MODEL_ID", "some/model")

    def fail_if_instantiated(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("HFTradingModel should not be instantiated when AI is disabled")

    monkeypatch.setattr(
        "xau_pro_bot.signals.ai_gate.HFTradingModel",
        fail_if_instantiated,
    )

    result = MasterSignalEngine().analyze(all_tfs)

    assert result["ai_enabled"] is False
    assert _ai_fields(result) == {
        "ai_enabled": False,
        "ai_direction": None,
        "ai_confidence": None,
        "ai_reason": None,
        "ai_blocked": False,
        "ai_score_delta_buy": 0,
        "ai_score_delta_sell": 0,
    }


def test_ai_buy_high_confidence_increases_buy_score(downtrend_df):
    tfs = _all_tfs(downtrend_df)
    baseline = MasterSignalEngine(ai_enabled=False).analyze(tfs)
    model = MockAIModel({"direction": "BUY", "confidence": 0.76})

    result = MasterSignalEngine(ai_enabled=True, ai_model=model).analyze(tfs)

    assert baseline["direction"] == "BUY"
    assert model.calls == 1
    assert model.seen_features is not None
    assert result["ai_enabled"] is True
    assert result["ai_direction"] == "BUY"
    assert result["ai_confidence"] == 0.76
    assert result["ai_reason"] == "AI agrees with deterministic signal"
    assert result["ai_blocked"] is False
    assert result["ai_score_delta_buy"] == 12
    assert result["ai_score_delta_sell"] == 0
    assert result["score"] >= baseline["score"] + 12


def test_ai_no_trade_high_confidence_blocks_signal(all_tfs):
    model = MockAIModel({"direction": "NO_TRADE", "confidence": 0.65})

    result = MasterSignalEngine(ai_enabled=True, ai_model=model).analyze(all_tfs)

    assert model.calls == 1
    assert result["tier"] == "NO_SIGNAL"
    assert result["ai_enabled"] is True
    assert result["ai_direction"] == "NO_TRADE"
    assert result["ai_confidence"] == 0.65
    assert result["ai_blocked"] is True
    assert "blocks" in result["ai_reason"] or "blocked" in result["ai_reason"]


def test_ai_conflict_penalizes_deterministic_direction(all_tfs):
    baseline = MasterSignalEngine(ai_enabled=False).analyze(all_tfs)
    ai_direction = "SELL" if baseline["direction"] == "BUY" else "BUY"
    model = MockAIModel({"direction": ai_direction, "confidence": 0.70})

    result = MasterSignalEngine(ai_enabled=True, ai_model=model).analyze(all_tfs)

    assert model.calls == 1
    assert result["score"] <= baseline["score"]
    assert result["ai_direction"] == ai_direction
    assert result["ai_confidence"] == 0.70
    assert result["ai_blocked"] is False
    assert "conflicts" in result["ai_reason"]
