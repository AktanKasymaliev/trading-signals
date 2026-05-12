"""Master scoring engine: combines layers, picks direction, computes levels."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.indicators import classic, wyckoff
from xau_pro_bot.indicators.ict import (
    find_fvg, find_order_blocks, find_liquidity, get_killzone,
)
from xau_pro_bot.indicators.sr_zones import find_sr_zones
from xau_pro_bot.models.calibration import ai_prediction_to_adjustment
from xau_pro_bot.models.features import build_ai_features
from xau_pro_bot.models.hf_model import HFTradingModel
from xau_pro_bot.signals.ict_signals import score_ict
from xau_pro_bot.signals.smc_signals import score_smc
from xau_pro_bot.signals.classic_signals import score_classic


class MasterSignalEngine:
    """Aggregates all scoring layers and produces a structured signal."""

    def __init__(
        self,
        ai_enabled: bool | None = None,
        ai_model: Any | None = None,
    ) -> None:
        ai_cfg = config.load_ai_config()
        self.ai_enabled = bool(ai_cfg["enabled"] if ai_enabled is None else ai_enabled)
        self.ai_model = ai_model
        if self.ai_enabled and self.ai_model is None:
            self.ai_model = HFTradingModel(
                model_id=str(ai_cfg["model_id"]),
                model_type=str(ai_cfg["model_type"]),
                cache_dir=str(ai_cfg["cache_dir"]),
                revision=str(ai_cfg["revision"]),
            )

    @staticmethod
    def _tier(score: float) -> str:
        if score >= config.STRONG_SIGNAL:
            return "STRONG"
        if score >= config.NORMAL_SIGNAL:
            return "NORMAL"
        if score >= config.WEAK_SIGNAL:
            return "WEAK"
        return "NO_SIGNAL"

    def _macro_bias(self, w1_df, d1_df) -> tuple[float, float, list[str]]:
        bull = bear = 0.0
        reasons: list[str] = []
        d1_last = d1_df.iloc[-1]
        e50 = d1_last.get("EMA_50", np.nan)
        e200 = d1_last.get("EMA_200", np.nan)
        if not pd.isna(e50) and not pd.isna(e200):
            if e50 > e200:
                bull += 20
                reasons.append("D1 EMA50 > EMA200")
            else:
                bear += 20
                reasons.append("D1 EMA50 < EMA200")
        w1_last = w1_df.iloc[-1]
        w1_prev = w1_df.iloc[-2] if len(w1_df) >= 2 else w1_last
        if not pd.isna(w1_last.get("EMA_50", np.nan)) and not pd.isna(w1_prev.get("EMA_50", np.nan)):
            if w1_last["EMA_50"] > w1_prev["EMA_50"]:
                bull += 8
            else:
                bear += 8
        wy = wyckoff.detect_wyckoff(d1_df)
        if wy["bias"] == "bull":
            bull += 5
            reasons.append(f"Wyckoff {wy['phase']}")
        elif wy["bias"] == "bear":
            bear += 5
            reasons.append(f"Wyckoff {wy['phase']}")
        return bull, bear, reasons

    def _enrich(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {tf: classic.add_classic(df) for tf, df in data.items()}

    def _macro_penalty(self, direction: str, d1_df) -> tuple[float, str | None]:
        d1_last = d1_df.iloc[-1]
        ema50 = d1_last.get("EMA_50", np.nan)
        ema200 = d1_last.get("EMA_200", np.nan)
        if pd.isna(ema50) or pd.isna(ema200):
            return 0.0, None
        d1_bull = ema50 > ema200
        if direction == "BUY" and not d1_bull:
            return 20.0, "D1 trend against BUY"
        if direction == "SELL" and d1_bull:
            return 20.0, "D1 trend against SELL"
        return 0.0, None

    def _disabled_ai_fields(self) -> dict[str, Any]:
        return {
            "ai_enabled": False,
            "ai_direction": None,
            "ai_confidence": None,
            "ai_reason": None,
            "ai_blocked": False,
            "ai_score_delta_buy": 0,
            "ai_score_delta_sell": 0,
        }

    def _run_ai_adjustment(
        self,
        data: dict[str, pd.DataFrame],
        deterministic_direction: str,
    ) -> dict[str, Any]:
        if not self.ai_enabled:
            return self._disabled_ai_fields()

        if self.ai_model is None:
            prediction = {"direction": "NO_TRADE", "confidence": 0.0}
        else:
            features = build_ai_features(data)
            prediction = self.ai_model.predict(features)

        adjustment = ai_prediction_to_adjustment(prediction, deterministic_direction)
        return {
            "ai_enabled": True,
            "ai_direction": adjustment["ai_direction"],
            "ai_confidence": adjustment["ai_confidence"],
            "ai_reason": adjustment["reason"],
            "ai_blocked": adjustment["block_signal"],
            "ai_score_delta_buy": adjustment["score_delta_buy"],
            "ai_score_delta_sell": adjustment["score_delta_sell"],
        }

    def _compute_levels(self, direction: str, h1_df, m15_df, d1_df) -> dict[str, Any]:
        entry = float(m15_df["Close"].iloc[-1])
        atr_m15 = float(m15_df["ATR_14"].iloc[-1]) if "ATR_14" in m15_df else 1.0
        if pd.isna(atr_m15) or atr_m15 <= 0:
            atr_m15 = max(entry * 0.001, 0.5)

        obs = find_order_blocks(h1_df, lookback=config.OB_LOOKBACK)
        fvgs = find_fvg(h1_df, max_gaps=5)
        liq = find_liquidity(h1_df, lookback=30)

        if direction == "BUY":
            ob_low = min(
                (ob["low"] for ob in obs if ob["type"] == "bull" and ob["low"] < entry),
                default=None,
            )
            fvg_bottom = min(
                (f["bottom"] for f in fvgs if f["type"] == "bull" and f["bottom"] < entry),
                default=None,
            )
            sl_candidates = [c for c in (ob_low, fvg_bottom) if c is not None]
            sl = (max(sl_candidates) if sl_candidates else entry - 5 * atr_m15) - atr_m15 * 0.3

            tp1 = next(
                (f["midpoint"] for f in fvgs if f["type"] == "bull" and f["midpoint"] > entry),
                None,
            )
            if tp1 is None:
                tp1 = entry + 2 * (entry - sl)
            tp2 = min((x for x in liq["buy_side"] if x > entry), default=None)
            tp3 = float(d1_df["High"].tail(50).max())
        else:
            ob_high = max(
                (ob["high"] for ob in obs if ob["type"] == "bear" and ob["high"] > entry),
                default=None,
            )
            fvg_top = max(
                (f["top"] for f in fvgs if f["type"] == "bear" and f["top"] > entry),
                default=None,
            )
            sl_candidates = [c for c in (ob_high, fvg_top) if c is not None]
            sl = (min(sl_candidates) if sl_candidates else entry + 5 * atr_m15) + atr_m15 * 0.3

            tp1 = next(
                (f["midpoint"] for f in fvgs if f["type"] == "bear" and f["midpoint"] < entry),
                None,
            )
            if tp1 is None:
                tp1 = entry - 2 * (sl - entry)
            tp2 = max((x for x in liq["sell_side"] if x < entry), default=None)
            tp3 = float(d1_df["Low"].tail(50).min())

        risk = abs(entry - sl)
        if risk <= 0:
            risk = atr_m15
        tp2_unavailable = False
        if tp2 is None:
            tp2_unavailable = True
            rr = abs(tp1 - entry) / risk
        else:
            rr = abs(tp2 - entry) / risk
            if rr < config.MIN_RR:
                tp2_unavailable = True
                tp2 = None
                rr = abs(tp1 - entry) / risk

        return {
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(float(tp1), 2) if tp1 is not None else None,
            "tp2": round(float(tp2), 2) if tp2 is not None else None,
            "tp3": round(float(tp3), 2) if tp3 is not None else None,
            "rr": round(float(rr), 2),
            "tp2_unavailable": tp2_unavailable,
            "atr_h1": float(h1_df["ATR_14"].iloc[-1]) if "ATR_14" in h1_df else atr_m15,
        }

    def analyze(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        enriched = self._enrich(data)
        w1, d1, h4, h1, m15 = (enriched[k] for k in ("W1", "D1", "H4", "H1", "M15"))

        h1_atr_val = h1["ATR_14"].iloc[-1] if "ATR_14" in h1 else 1.0
        h1_atr = float(h1_atr_val) if not pd.isna(h1_atr_val) else 1.0

        current_price = float(m15["Close"].iloc[-1])
        sr = find_sr_zones(h4_df=h4, d1_df=d1, current_price=current_price)
        liq = find_liquidity(h1, lookback=30)

        macro_bull, macro_bear, macro_reasons = self._macro_bias(w1, d1)
        smc_bull, smc_bear, smc_reasons = score_smc(h4, sr_zones=sr, liquidity=liq)
        ict_bull, ict_bear, ict_reasons = score_ict(h1, m15, h1_atr)
        cls_bull, cls_bear, cls_reasons = score_classic(h1, m15)

        bull_score = macro_bull + smc_bull + ict_bull + cls_bull
        bear_score = macro_bear + smc_bear + ict_bear + cls_bear

        direction = "BUY" if bull_score >= bear_score else "SELL"
        macro_pen, pen_reason = self._macro_penalty(direction, d1)
        if direction == "BUY":
            bull_score -= macro_pen
        else:
            bear_score -= macro_pen

        reasons = {
            "macro": macro_reasons,
            "smc": smc_reasons,
            "ict": ict_reasons,
            "classic": cls_reasons,
            "penalties": [pen_reason] if pen_reason else [],
        }

        ai_fields = self._run_ai_adjustment(data, direction)
        bull_score += ai_fields["ai_score_delta_buy"]
        bear_score += ai_fields["ai_score_delta_sell"]
        if ai_fields["ai_reason"]:
            reasons["ai"] = [ai_fields["ai_reason"]]

        final_score = max(bull_score, bear_score)
        tier = "NO_SIGNAL" if ai_fields["ai_blocked"] else self._tier(final_score)

        if tier == "NO_SIGNAL":
            return {
                "direction": direction,
                "tier": tier,
                "score": int(final_score),
                "entry": float(m15["Close"].iloc[-1]),
                "sl": None, "tp1": None, "tp2": None, "tp3": None,
                "rr": None,
                "killzone": get_killzone(),
                "reasons": reasons,
                "tp2_unavailable": False,
                "ts_utc": datetime.now(timezone.utc),
                **ai_fields,
            }

        levels = self._compute_levels(direction, h1, m15, d1)
        return {
            "direction": direction,
            "tier": tier,
            "score": int(final_score),
            **levels,
            "killzone": get_killzone(),
            "reasons": reasons,
            "ts_utc": datetime.now(timezone.utc),
            **ai_fields,
        }
