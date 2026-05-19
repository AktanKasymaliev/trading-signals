from datetime import datetime, timezone

import pytest

from xau_pro_bot.state import State
from xau_pro_bot.signals.filters import should_send, SkipReason


@pytest.fixture
def state(tmp_path):
    return State(db_path=str(tmp_path / "f.db"))


def _sig(direction="BUY", tier="STRONG", entry=2000.0, score=70, tp1=2010, tp2=2020):
    return {
        "direction": direction, "tier": tier, "entry": entry, "score": score,
        "sl": entry - 5, "tp1": tp1, "tp2": tp2, "tp3": entry + 30,
        "rr": 2.0, "killzone": "London KZ", "atr_h1": 5.0,
        "tp2_unavailable": False,
    }


def test_no_signal_blocked(state):
    sig = _sig(tier="NO_SIGNAL", score=20)
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.NO_SIGNAL


def test_weak_outside_killzone_blocked(state):
    sig = _sig(tier="WEAK", score=42)
    sig["killzone"] = None
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.WEAK_OUTSIDE_KZ


def test_dedup_within_2h_blocks(state):
    sig = _sig()
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.DEDUP


def test_atr_reprice_overrides_dedup(state):
    sig = _sig(entry=2000.0)
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    sig_moved = _sig(entry=2000.0 + 1.5 * 5.0 + 1)
    ok, reason = should_send(sig_moved, state)
    assert ok, f"expected ATR-reprice to bypass dedup, got reason={reason}"


def test_rate_limit_day(state):
    base = _sig()
    for i in range(6):
        state.record_signal({**base, "ts_utc": datetime.now(timezone.utc).isoformat(),
                             "reasons_json": "{}", "entry": 2000 + i * 100})
    next_sig = _sig(entry=9999.0)
    ok, reason = should_send(next_sig, state)
    assert not ok and reason == SkipReason.RATE_LIMIT_DAY


def test_weak_cooldown(state):
    sig = _sig(tier="WEAK", score=42)
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    new_sig = _sig(tier="WEAK", score=42, entry=2100.0)
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.WEAK_COOLDOWN


# ── Per-stream tests (R3) ─────────────────────────────


def test_swing_per_day_cap(state):
    sig = _sig()
    sig["stream"] = "swing"
    for _ in range(2):
        state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                             "reasons_json": "{}"})
    new_sig = _sig(entry=2200.0)
    new_sig["stream"] = "swing"
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.RATE_LIMIT_DAY


def test_swing_same_direction_24h(state):
    sig = _sig(direction="BUY")
    sig["stream"] = "swing"
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    new_sig = _sig(direction="BUY", entry=2050.0)
    new_sig["stream"] = "swing"
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.SWING_DIRECTION_COOLDOWN


def test_scalp_min_gap_30min(state):
    sig = _sig()
    sig["stream"] = "scalp"
    sig["killzone"] = "London KZ"
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    new_sig = _sig(entry=2010.0)
    new_sig["stream"] = "scalp"
    new_sig["killzone"] = "London KZ"
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.SCALP_GAP


def test_scalp_must_be_in_killzone(state):
    sig = _sig()
    sig["stream"] = "scalp"
    sig["killzone"] = None
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.SCALP_OUTSIDE_KZ


# ── Duplicate-send suppression ────────────────────────


def test_duplicate_scan_intraday_blocked(state):
    sig = _sig()
    fps = {("intraday", "BUY", round(sig["entry"], 2))}
    ok, reason = should_send(sig, state, scan_fingerprints=fps)
    assert not ok and reason == SkipReason.DUPLICATE_SCAN


def test_duplicate_scan_router_emits_twice(state):
    """Simulate two router passes within a single scan cycle producing the same signal."""
    sig = _sig()
    fps: set = set()
    ok1, _ = should_send(sig, state, scan_fingerprints=fps)
    assert ok1
    fps.add(("intraday", sig["direction"], round(sig["entry"], 2)))
    ok2, reason2 = should_send(sig, state, scan_fingerprints=fps)
    assert not ok2 and reason2 == SkipReason.DUPLICATE_SCAN


def test_duplicate_active_swing_blocked(state):
    sig = _sig()
    sig["stream"] = "swing"
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.DUPLICATE_ACTIVE


def test_duplicate_scan_different_direction_allowed(state):
    sig = _sig(direction="BUY")
    fps = {("intraday", "SELL", round(sig["entry"], 2))}
    ok, _ = should_send(sig, state, scan_fingerprints=fps)
    assert ok


# ── Swing sanity guards ───────────────────────────────


def _swing_sig(entry=4253.85, sl=3913.00, tp1=5597.23, atr_h1=10.0):
    return {
        "direction": "BUY", "tier": "STRONG", "entry": entry, "score": 80,
        "sl": sl, "tp1": tp1, "tp2": None, "tp3": None, "rr": 3.9,
        "killzone": None, "atr_h1": atr_h1, "stream": "swing",
        "tp2_unavailable": True,
    }


def test_swing_target_too_far_rejected(state, monkeypatch):
    from xau_pro_bot import config as _cfg
    monkeypatch.setattr(_cfg, "SWING_MAX_TP1_ATR", 50.0)
    monkeypatch.setattr(_cfg, "SWING_MAX_SL_ATR", 200.0)
    sig = _swing_sig()  # tp1 distance ~134 ATR (1343 / 10)
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.SWING_TARGET_TOO_FAR


def test_swing_sl_too_wide_rejected(state, monkeypatch):
    from xau_pro_bot import config as _cfg
    monkeypatch.setattr(_cfg, "SWING_MAX_SL_ATR", 10.0)
    monkeypatch.setattr(_cfg, "SWING_MAX_TP1_ATR", 9999.0)
    sig = _swing_sig()  # sl distance ~34 ATR
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.SWING_SL_TOO_WIDE


def test_swing_send_disabled(state, monkeypatch):
    from xau_pro_bot import config as _cfg
    monkeypatch.setattr(_cfg, "SWING_SEND_ENABLED", False)
    sig = _swing_sig()
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.SWING_DISABLED


def test_swing_within_atr_bounds_allowed(state, monkeypatch):
    from xau_pro_bot import config as _cfg
    monkeypatch.setattr(_cfg, "SWING_SEND_ENABLED", True)
    monkeypatch.setattr(_cfg, "SWING_MAX_SL_ATR", 5.0)
    monkeypatch.setattr(_cfg, "SWING_MAX_TP1_ATR", 8.0)
    sig = _swing_sig(entry=2000.0, sl=1980.0, tp1=2040.0, atr_h1=10.0)
    ok, _ = should_send(sig, state)
    assert ok
