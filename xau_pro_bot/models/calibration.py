"""AI prediction calibration for deterministic signal scores."""

from __future__ import annotations

from typing import Any

from xau_pro_bot import config


def _empty(ai_direction: str, ai_confidence: float, reason: str,
           ai_low_confidence: bool = False) -> dict[str, Any]:
    return {
        "score_delta_buy": 0,
        "score_delta_sell": 0,
        "block_signal": False,
        "reason": reason,
        "ai_direction": ai_direction,
        "ai_confidence": ai_confidence,
        "ai_low_confidence": ai_low_confidence,
    }


def ai_prediction_to_adjustment(
    prediction: dict,
    deterministic_direction: str | None,
) -> dict[str, Any]:
    """Convert an AI prediction into score deltas and a block decision."""
    ai_config = config.load_ai_config()
    direction = str(prediction.get("direction") or "NO_TRADE").upper()
    confidence = float(prediction.get("confidence") or 0.0)

    if direction == "NO_TRADE" and confidence >= ai_config["no_trade_threshold"]:
        out = _empty(direction, confidence, "AI NO_TRADE confidence blocked signal")
        out["block_signal"] = True
        return out

    if confidence < ai_config["min_confidence"]:
        return _empty(
            direction, confidence,
            "AI confidence below minimum; deterministic signal unchanged",
            ai_low_confidence=True,
        )

    if deterministic_direction is None:
        return _empty(direction, confidence, "AI ignored without deterministic direction")

    deterministic_direction = deterministic_direction.upper()
    if direction == deterministic_direction:
        bonus = (
            ai_config["strong_score_bonus"]
            if confidence >= ai_config["strong_confidence"]
            else ai_config["score_bonus"]
        )
        out = _empty(direction, confidence, "AI agrees with deterministic signal")
        if direction == "BUY":
            out["score_delta_buy"] = bonus
        elif direction == "SELL":
            out["score_delta_sell"] = bonus
        return out

    if direction in {"BUY", "SELL"}:
        out = _empty(direction, confidence, "AI conflicts with deterministic signal")
        if deterministic_direction == "BUY":
            out["score_delta_buy"] = -ai_config["conflict_penalty"]
        elif deterministic_direction == "SELL":
            out["score_delta_sell"] = -ai_config["conflict_penalty"]
        return out

    return _empty(direction, confidence, "AI prediction did not change score")
