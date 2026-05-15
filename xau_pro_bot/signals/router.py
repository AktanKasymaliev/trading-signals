"""Stream router: invokes all stream analyzers and returns non-null SignalResults."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from xau_pro_bot.signals.ai_gate import AIExplanationGate
from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.swing_analyzer import SwingAnalyzer
from xau_pro_bot.signals.scalp_analyzer import ScalpAnalyzer

log = logging.getLogger(__name__)


class _IntradayWrap:
    """Wraps MasterSignalEngine to add strategy_label and horizon_label."""

    def __init__(self, gate: AIExplanationGate | None = None) -> None:
        self._engine = MasterSignalEngine(gate=gate)

    def analyze(self, data: dict[str, pd.DataFrame]) -> dict | None:
        sig = self._engine.analyze(data)
        if sig is None or sig["tier"] == "NO_SIGNAL":
            return None
        labels: list[str] = []
        if sig["reasons"].get("smc"):
            labels.append("SMC")
        if sig["reasons"].get("ict"):
            labels.append("ICT")
        if sig["reasons"].get("classic"):
            labels.append("Classic")
        sig["strategy_label"] = "+".join(labels) or "Intraday"
        sig["horizon_label"] = "1-24 часа"
        return sig


class StreamRouter:
    def __init__(self) -> None:
        # One shared gate so the AI model loads once and is reused
        # across intraday/swing/scalp streams.
        gate = AIExplanationGate()
        self.analyzers: dict[str, Any] = {
            "intraday": _IntradayWrap(gate=gate),
            "swing":    SwingAnalyzer(gate=gate),
            "scalp":    ScalpAnalyzer(gate=gate),
        }

    def analyze(self, data: dict[str, pd.DataFrame]) -> list[dict]:
        out: list[dict] = []
        for stream_name, analyzer in self.analyzers.items():
            try:
                sig = analyzer.analyze(data)
            except Exception:
                log.exception("Stream %s failed", stream_name)
                continue
            if sig is None:
                continue
            sig["stream"] = stream_name
            out.append(sig)
        return out
