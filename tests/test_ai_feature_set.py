from __future__ import annotations

import pandas as pd
import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


class CapturingAI:
    def __init__(self) -> None:
        self.last_features_columns: list[str] | None = None

    def predict(self, features):
        self.last_features_columns = list(features.columns)
        return {
            "direction": "NO_TRADE",
            "confidence": 0.0,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }


def test_engine_uses_internal_feature_set_by_default(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    monkeypatch.delenv("AI_FEATURE_SET", raising=False)
    ai = CapturingAI()

    engine = MasterSignalEngine(ai_enabled=True, ai_model=ai)
    engine.analyze(all_tfs)

    assert ai.last_features_columns is not None
    assert "close_m15" in ai.last_features_columns
    assert "rsi_h1" in ai.last_features_columns


def test_engine_uses_smc_v2_feature_set_when_configured(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    monkeypatch.setenv("AI_FEATURE_SET", "smc_v2")
    ai = CapturingAI()

    engine = MasterSignalEngine(ai_enabled=True, ai_model=ai)
    engine.analyze(all_tfs)

    assert ai.last_features_columns is not None
    assert ai.last_features_columns[0] == "Close"
    assert "FVG_Size" in ai.last_features_columns
    assert len(ai.last_features_columns) == 21
