"""Helpers that convert raw AI gate output into human-readable analysis fields.

Used by the engine to enrich signal dicts with explanation metadata so the
Telegram formatter and backtest diagnostics can show *why* a signal was kept,
blocked, or downgraded. Pure functions — no I/O, no config side-effects."""

from __future__ import annotations


# Mapping: AI feature_set tag → human-friendly model name shown to users.
# Keep this list short and explicit; unknown tags fall back to the tag itself.
_FEATURE_SET_TO_NAME: dict[str, str] = {
    "internal": "Path C legacy",
    "legacy": "Path C legacy",
    "smc_v2": "Path C SMC v2",
    "stationary": "Path F stationary",
}


def model_name(feature_set: str | None, ai_enabled: bool) -> str | None:
    if not ai_enabled:
        return None
    if not feature_set:
        return "Path C legacy"
    return _FEATURE_SET_TO_NAME.get(feature_set, feature_set)


def derive_action(
    *,
    ai_enabled: bool,
    ai_blocked: bool,
    ai_direction: str | None,
    deterministic_direction: str,
    ai_low_confidence: bool = False,
) -> str | None:
    """KEEP / BLOCK / DOWNGRADE / NEUTRAL / None.

    - None: AI disabled or skipped (no decision to display).
    - BLOCK: gate refused the trade.
    - DOWNGRADE: AI took an opinion that conflicts with deterministic direction
      but did not hard-block (penalty applied).
    - NEUTRAL: AI evaluated but confidence below minimum threshold — it did
      NOT actually confirm the deterministic signal. Distinct from KEEP so
      the user is not misled into thinking AI endorsed the trade.
    - KEEP: AI agrees with sufficient confidence while letting the signal
      through.
    """
    if not ai_enabled:
        return None
    if ai_direction is None:
        return None  # AI was enabled but skipped (e.g. incomplete features).
    if ai_blocked:
        return "BLOCK"
    if ai_low_confidence:
        return "NEUTRAL"
    if ai_direction in ("BUY", "SELL") and ai_direction != deterministic_direction:
        return "DOWNGRADE"
    return "KEEP"


def derive_risk_label(
    *,
    tier: str,
    penalties: list[str] | None,
    ai_action: str | None,
) -> str:
    """HIGH_RISK / MEDIUM_RISK / CLEAN_SETUP.

    Simple, deterministic mapping (user-approved):
    - HIGH_RISK: WEAK tier OR any penalty present OR AI says BLOCK/DOWNGRADE.
    - CLEAN_SETUP: STRONG tier AND no penalty AND AI either KEEP or absent.
    - MEDIUM_RISK: everything else.
    """
    has_penalty = bool(penalties)
    if tier == "WEAK" or has_penalty or ai_action in ("BLOCK", "DOWNGRADE"):
        return "HIGH_RISK"
    if tier == "STRONG" and not has_penalty and ai_action in (None, "KEEP"):
        return "CLEAN_SETUP"
    return "MEDIUM_RISK"


def short_reason(reason: str | None, limit: int = 80) -> str | None:
    if not reason:
        return None
    text = " ".join(reason.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
