"""Scalp stream analyzer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from xau_pro_bot.indicators.scalping import scalp_signal


class ScalpAnalyzer:
    def __init__(self, gate: Any | None = None) -> None:
        self._gate = gate

    def analyze(self, data: dict[str, pd.DataFrame]) -> dict | None:
        res = scalp_signal(m15_df=data["M15"], h1_df=data["H1"], h4_df=data["H4"])
        if res is None:
            return None
        tier = "WEAK" if res["counter_trend"] else "NORMAL"
        score = 45 if res["counter_trend"] else 55
        risk = abs(res["entry"] - res["sl"])
        reward = abs(res["tp1"] - res["entry"])
        rr = round(reward / risk, 2) if risk > 0 else 0.0
        sig: dict[str, Any] = {
            "direction": res["direction"],
            "tier": tier,
            "score": score,
            "entry": res["entry"],
            "sl": res["sl"],
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "tp3": None,
            "rr": rr,
            "tp2_unavailable": False,
            "killzone": res["killzone"],
            "reasons": {
                "scalp": res["conditions_met"]
                          + (["counter-trend"] if res["counter_trend"] else []),
                "macro": [], "smc": [], "ict": [], "classic": [], "penalties": [],
            },
            "ts_utc": datetime.now(timezone.utc),
            "strategy_label": "Scalp M15",
            "horizon_label": "15-60 минут",
            "atr_h1": res["atr_m15"],
        }
        if self._gate is not None:
            sig = self._gate.enrich(sig, data)
            if sig.get("ai_blocked"):
                return None
        return sig
