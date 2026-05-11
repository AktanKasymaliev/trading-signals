"""Quality filters: dedup, ATR-reprice (early-exit), per-stream rate limits."""

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
    SWING_DIRECTION_COOLDOWN = "swing_direction_cooldown"
    SCALP_OUTSIDE_KZ = "scalp_outside_kz"
    SCALP_GAP = "scalp_gap"
    UNKNOWN_STREAM = "unknown_stream"


def _intraday_check(sig, state, bypass_dedup):
    if sig["tier"] == "NO_SIGNAL":
        return False, SkipReason.NO_SIGNAL
    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1
    if sig["tier"] == "WEAK" and not sig.get("killzone"):
        return False, SkipReason.WEAK_OUTSIDE_KZ
    if state.count_today(stream="intraday") >= config.MAX_INTRADAY_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY
    if sig["tier"] == "WEAK":
        last_weak = state.last_weak_ts(stream="intraday")
        if last_weak is not None:
            elapsed = datetime.now(timezone.utc) - last_weak
            if elapsed < timedelta(hours=config.WEAK_COOLDOWN_HOURS):
                return False, SkipReason.WEAK_COOLDOWN
    if bypass_dedup:
        return True, None
    last = state.last_signal(direction=sig["direction"], stream="intraday")
    if last is None:
        return True, None
    atr_h1 = sig.get("atr_h1", 1.0)
    if abs(sig["entry"] - last["entry"]) >= config.REPRICE_ATR_MULT * atr_h1:
        return True, None
    last_ts = datetime.fromisoformat(last["ts_utc"])
    if datetime.now(timezone.utc) - last_ts >= timedelta(hours=config.DEDUP_HOURS):
        return True, None
    return False, SkipReason.DEDUP


def _swing_check(sig, state, bypass_dedup):
    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1
    if state.count_today(stream="swing") >= config.MAX_SWING_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY
    if bypass_dedup:
        return True, None
    last = state.last_signal(direction=sig["direction"], stream="swing")
    if last is None:
        return True, None
    last_ts = datetime.fromisoformat(last["ts_utc"])
    cooldown = timedelta(hours=config.SWING_DIRECTION_COOLDOWN_HOURS)
    if datetime.now(timezone.utc) - last_ts < cooldown:
        return False, SkipReason.SWING_DIRECTION_COOLDOWN
    return True, None


def _scalp_check(sig, state, bypass_dedup):
    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1
    if not sig.get("killzone"):
        return False, SkipReason.SCALP_OUTSIDE_KZ
    if state.count_today(stream="scalp") >= config.MAX_SCALP_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY
    if bypass_dedup:
        return True, None
    last_ts = state.last_scalp_ts()
    if last_ts is not None:
        gap = datetime.now(timezone.utc) - last_ts
        if gap < timedelta(minutes=config.SCALP_MIN_GAP_MINUTES):
            return False, SkipReason.SCALP_GAP
    return True, None


def should_send(sig: dict, state: State,
                bypass_dedup: bool = False) -> tuple[bool, SkipReason | None]:
    stream = sig.get("stream", "intraday")
    if stream == "intraday":
        return _intraday_check(sig, state, bypass_dedup)
    if stream == "swing":
        return _swing_check(sig, state, bypass_dedup)
    if stream == "scalp":
        return _scalp_check(sig, state, bypass_dedup)
    return False, SkipReason.UNKNOWN_STREAM
