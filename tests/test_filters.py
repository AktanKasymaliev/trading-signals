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
