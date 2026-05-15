"""Swing stream analyzer wrapping find_swing_setup into SignalResult."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from xau_pro_bot.indicators.swing import find_swing_setup
from xau_pro_bot.indicators.ict import get_killzone


_HORIZON = {"1000pip": "1-4 недели", "500pip": "2-7 дней"}
_TIER = {"1000pip": "STRONG", "500pip": "NORMAL"}
_SCORE = {"1000pip": 80, "500pip": 65}


class SwingAnalyzer:
    def __init__(self, gate: Any | None = None) -> None:
        # gate is an AIExplanationGate-shaped object; typed loosely so
        # tests can inject lightweight doubles without importing the class.
        self._gate = gate

    def analyze(self, data: dict[str, pd.DataFrame]) -> dict | None:
        setup = find_swing_setup(d1_df=data["D1"], h4_df=data["H4"])
        if setup is None:
            return None
        sig: dict[str, Any] = {
            "direction": setup["direction"],
            "tier": _TIER[setup["type"]],
            "score": _SCORE[setup["type"]],
            "entry": setup["entry"],
            "sl": setup["sl"],
            "tp1": setup["tp"],
            "tp2": None,
            "tp3": None,
            "rr": setup["rr"],
            "tp2_unavailable": True,
            "killzone": get_killzone(),
            "reasons": {
                "swing": [f"{setup['type']} setup, range {setup['range_pips']} pips"],
                "macro": [], "smc": [], "ict": [], "classic": [], "penalties": [],
            },
            "ts_utc": datetime.now(timezone.utc),
            "strategy_label": f"Swing {'1000' if setup['type'] == '1000pip' else '500'}",
            "horizon_label": _HORIZON[setup["type"]],
            "atr_h1": 1.0,
        }
        if self._gate is not None:
            sig = self._gate.enrich(sig, data)
            if sig.get("ai_blocked"):
                return None
        return sig
