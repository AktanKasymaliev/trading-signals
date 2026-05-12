from __future__ import annotations

import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


class CountingMockAI:
    """Records the number of predict() calls and instance identity."""

    instances: list["CountingMockAI"] = []

    def __init__(self) -> None:
        self.calls = 0
        CountingMockAI.instances.append(self)

    def predict(self, features):
        self.calls += 1
        return {
            "direction": "BUY",
            "confidence": 0.50,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }


def test_engine_keeps_single_ai_instance_across_analyze_calls(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    CountingMockAI.instances.clear()
    ai = CountingMockAI()
    engine = MasterSignalEngine(ai_enabled=True, ai_model=ai)

    for _ in range(3):
        engine.analyze(all_tfs)

    assert engine.ai_model is ai
    assert ai.calls == 3
    assert len(CountingMockAI.instances) == 1


def test_router_reuses_engine_across_runs():
    from xau_pro_bot.signals.router import StreamRouter

    router = StreamRouter()
    intraday = router.analyzers["intraday"]

    assert intraday is router.analyzers["intraday"]
