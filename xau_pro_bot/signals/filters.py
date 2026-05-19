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
    DUPLICATE_ACTIVE = "duplicate_active"
    DUPLICATE_SCAN = "duplicate_scan"
    SWING_TARGET_TOO_FAR = "swing_target_too_far"
    SWING_SL_TOO_WIDE = "swing_sl_too_wide"
    SWING_DISABLED = "swing_disabled"


def _entry_eq(a: float | None, b: float | None, tol: float = 0.05) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _active_duplicate(sig, state) -> bool:
    """Return True if an ACTIVE signal with same stream+direction+entry exists."""
    stream = sig.get("stream", "intraday")
    try:
        rows = state.get_active()
    except Exception:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.DEDUP_HOURS)
    for r in rows:
        if r.get("stream", "intraday") != stream:
            continue
        if r.get("direction") != sig.get("direction"):
            continue
        if not _entry_eq(r.get("entry"), sig.get("entry")):
            continue
        try:
            ts = datetime.fromisoformat(r["ts_utc"])
        except Exception:
            return True
        if ts >= cutoff:
            return True
    return False


def _scan_duplicate(sig, scan_fingerprints) -> bool:
    if not scan_fingerprints:
        return False
    fp = (
        sig.get("stream", "intraday"),
        sig.get("direction"),
        round(float(sig.get("entry") or 0.0), 2),
    )
    return fp in scan_fingerprints


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


def _swing_sanity(sig) -> tuple[bool, SkipReason | None]:
    if not config.SWING_SEND_ENABLED:
        return False, SkipReason.SWING_DISABLED
    atr = sig.get("atr_h1") or sig.get("atr_d1") or 0.0
    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    if not atr or atr <= 0 or entry is None or sl is None or tp1 is None:
        return True, None
    sl_units = abs(float(entry) - float(sl)) / float(atr)
    tp1_units = abs(float(tp1) - float(entry)) / float(atr)
    if sl_units > config.SWING_MAX_SL_ATR:
        return False, SkipReason.SWING_SL_TOO_WIDE
    if tp1_units > config.SWING_MAX_TP1_ATR:
        return False, SkipReason.SWING_TARGET_TOO_FAR
    return True, None


def should_send(sig: dict, state: State,
                bypass_dedup: bool = False,
                scan_fingerprints: set | None = None,
                ) -> tuple[bool, SkipReason | None]:
    stream = sig.get("stream", "intraday")
    if not bypass_dedup:
        if _scan_duplicate(sig, scan_fingerprints):
            return False, SkipReason.DUPLICATE_SCAN
        # Active-signal cross-stream guard. Intraday already has its own
        # DEDUP path that returns SkipReason.DEDUP, so we only apply this
        # belt-and-suspenders check to swing/scalp.
        if stream in ("swing", "scalp") and _active_duplicate(sig, state):
            return False, SkipReason.DUPLICATE_ACTIVE
    if stream == "intraday":
        return _intraday_check(sig, state, bypass_dedup)
    if stream == "swing":
        ok, reason = _swing_sanity(sig)
        if not ok:
            return ok, reason
        return _swing_check(sig, state, bypass_dedup)
    if stream == "scalp":
        return _scalp_check(sig, state, bypass_dedup)
    return False, SkipReason.UNKNOWN_STREAM
