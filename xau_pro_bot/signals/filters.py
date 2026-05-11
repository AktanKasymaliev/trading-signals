"""Quality filters: dedup, ATR-reprice (early-exit), rate limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from xau_pro_bot import config
from xau_pro_bot.state import State


class SkipReason(str, Enum):
    NO_SIGNAL = "no_signal"
    WEAK_OUTSIDE_KZ = "weak_outside_kz"
    DEDUP = "dedup"
    RATE_LIMIT_DAY = "rate_limit_day"
    WEAK_COOLDOWN = "weak_cooldown"
    NO_TP1 = "no_tp1"


def should_send(sig: dict, state: State,
                bypass_dedup: bool = False) -> tuple[bool, SkipReason | None]:
    """Returns (True, None) if signal should be sent, else (False, reason)."""
    if sig["tier"] == "NO_SIGNAL":
        return False, SkipReason.NO_SIGNAL

    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1

    if sig["tier"] == "WEAK" and not sig.get("killzone"):
        return False, SkipReason.WEAK_OUTSIDE_KZ

    if state.count_today() >= config.MAX_SIGNALS_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY

    if sig["tier"] == "WEAK":
        last_weak = state.last_weak_ts()
        if last_weak is not None:
            elapsed = datetime.now(timezone.utc) - last_weak
            if elapsed < timedelta(hours=config.WEAK_COOLDOWN_HOURS):
                return False, SkipReason.WEAK_COOLDOWN

    if bypass_dedup:
        return True, None

    last = state.last_signal(direction=sig["direction"])
    if last is None:
        return True, None

    atr_h1 = sig.get("atr_h1", 1.0)
    moved = abs(sig["entry"] - last["entry"])
    if moved >= config.REPRICE_ATR_MULT * atr_h1:
        return True, None  # ATR-reprice wins (early-exit)

    last_ts = datetime.fromisoformat(last["ts_utc"])
    if datetime.now(timezone.utc) - last_ts >= timedelta(hours=config.DEDUP_HOURS):
        return True, None

    return False, SkipReason.DEDUP
