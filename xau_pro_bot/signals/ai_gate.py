"""Reusable AI evaluation + explanation helper.

Extracted from ``MasterSignalEngine`` so swing and scalp analyzers can apply
the same Path C legacy gate to their signals, instead of skipping AI
enrichment entirely. Behaviour for the intraday path is unchanged: the
engine delegates to this helper.

Public surface:
    AIExplanationGate(ai_enabled=None, ai_model=None)
        .evaluate(data, deterministic_direction) -> dict[str, Any]
        .build_explanation(ai_fields, direction, tier, reasons) -> dict
        .enrich(sig, data) -> dict  # convenience for post-hoc enrichment
        .disabled_fields() -> dict
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.models.ai_explanation import (
    derive_action, derive_risk_label, model_name, short_reason,
)
from xau_pro_bot.models.calibration import ai_prediction_to_adjustment
from xau_pro_bot.models.features import build_ai_features
from xau_pro_bot.models.features_stationary import build_stationary_features
from xau_pro_bot.models.smc_v2_features import build_smc_v2_features
from xau_pro_bot.models.hf_model import HFTradingModel


def prime_feature_set(model: Any) -> str:
    """Eagerly load the model bundle (if needed) and return its tagged
    feature_set, defaulting to 'legacy'. Lets Path F artifacts trigger
    the matching feature builder at inference time."""
    if model is None:
        return "legacy"
    fs = getattr(model, "feature_set", None)
    if fs:
        return str(fs)
    for loader in ("_load", "_get_model", "_load_sklearn"):
        fn = getattr(model, loader, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break
    return str(getattr(model, "feature_set", "legacy") or "legacy")


class AIExplanationGate:
    """Evaluates the AI gate and builds explanation fields for a signal.

    A single instance can be shared across intraday/swing/scalp analyzers
    so the model is loaded once. Behaviour stays neutral when AI is
    disabled or the feature builder reports incomplete inputs.
    """

    def __init__(
        self,
        ai_enabled: bool | None = None,
        ai_model: Any | None = None,
    ) -> None:
        ai_cfg = config.load_ai_config()
        self.ai_enabled = bool(ai_cfg["enabled"] if ai_enabled is None else ai_enabled)
        self.ai_feature_set = str(ai_cfg.get("feature_set", "internal"))
        self.ai_model = ai_model
        if self.ai_enabled and self.ai_model is None:
            self.ai_model = HFTradingModel(
                model_id=str(ai_cfg["model_id"]),
                model_type=str(ai_cfg["model_type"]),
                cache_dir=str(ai_cfg["cache_dir"]),
                revision=str(ai_cfg["revision"]),
                filename=str(ai_cfg["model_filename"]),
                local_path=str(ai_cfg["local_path"]),
            )

    def disabled_fields(self) -> dict[str, Any]:
        return {
            "ai_enabled": False,
            "ai_direction": None,
            "ai_confidence": None,
            "ai_reason": None,
            "ai_blocked": False,
            "ai_score_delta_buy": 0,
            "ai_score_delta_sell": 0,
        }

    def evaluate(
        self,
        data: dict[str, pd.DataFrame],
        deterministic_direction: str,
    ) -> dict[str, Any]:
        if not self.ai_enabled:
            return self.disabled_fields()

        if self.ai_model is None:
            prediction: dict[str, Any] = {
                "direction": "NO_TRADE",
                "confidence": 0.0,
            }
        else:
            model_fs = prime_feature_set(self.ai_model)
            if model_fs == "stationary":
                features, complete = build_stationary_features(data)
            elif self.ai_feature_set == "smc_v2":
                features, complete = build_smc_v2_features(data)
            else:
                features, complete = build_ai_features(data)
            if not complete:
                return {
                    "ai_enabled": True,
                    "ai_direction": None,
                    "ai_confidence": None,
                    "ai_reason": "AI skipped: incomplete input features",
                    "ai_blocked": False,
                    "ai_score_delta_buy": 0,
                    "ai_score_delta_sell": 0,
                }
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

    def build_explanation(
        self,
        ai_fields: dict[str, Any],
        deterministic_direction: str,
        tier: str,
        reasons: dict[str, list[str]],
    ) -> dict[str, Any]:
        ai_enabled = bool(ai_fields.get("ai_enabled"))
        action = derive_action(
            ai_enabled=ai_enabled,
            ai_blocked=bool(ai_fields.get("ai_blocked")),
            ai_direction=ai_fields.get("ai_direction"),
            deterministic_direction=deterministic_direction,
        )
        risk_label = derive_risk_label(
            tier=tier,
            penalties=reasons.get("penalties") or [],
            ai_action=action,
        )
        return {
            "ai_model_name": model_name(self.ai_feature_set, ai_enabled),
            "ai_feature_set": self.ai_feature_set if ai_enabled else None,
            "ai_action": action,
            "ai_reason_short": short_reason(ai_fields.get("ai_reason")),
            "ai_risk_label": risk_label,
        }

    def enrich(
        self,
        sig: dict[str, Any],
        data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Return a new sig dict with ai_* fields and explanation merged in.

        Caller decides whether ``ai_blocked`` should propagate as NO_SIGNAL.
        This function does not mutate ``sig`` and does not change tier."""
        direction = sig["direction"]
        tier = sig.get("tier", "NO_SIGNAL")
        reasons = sig.get("reasons") or {}
        ai_fields = self.evaluate(data, direction)
        explanation = self.build_explanation(
            ai_fields, direction, tier, reasons,
        )
        return {**sig, **ai_fields, **explanation}
