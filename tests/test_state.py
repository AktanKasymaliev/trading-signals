from datetime import datetime, timedelta, timezone

import pytest

from xau_pro_bot.state import State


@pytest.fixture
def state(tmp_path):
    return State(db_path=str(tmp_path / "test.db"))


def _sig(direction="BUY", tier="STRONG", score=70, entry=2000.0,
         ts: datetime | None = None) -> dict:
    return {
        "ts_utc": (ts or datetime.now(timezone.utc)).isoformat(),
        "direction": direction,
        "tier": tier,
        "score": score,
        "entry": entry,
        "sl": entry - 10,
        "tp1": entry + 15,
        "tp2": entry + 30,
        "tp3": entry + 45,
        "rr": 2.0,
        "killzone": "London KZ",
        "reasons_json": "{}",
    }


def test_record_and_last_signal(state):
    sig = _sig()
    sid = state.record_signal(sig)
    assert sid > 0
    last = state.last_signal()
    assert last is not None
    assert last["direction"] == "BUY"
    assert last["entry"] == pytest.approx(2000.0)


def test_last_signal_filter_by_direction(state):
    state.record_signal(_sig(direction="BUY", entry=2000.0))
    state.record_signal(_sig(direction="SELL", entry=2100.0))
    assert state.last_signal(direction="BUY")["entry"] == pytest.approx(2000.0)
    assert state.last_signal(direction="SELL")["entry"] == pytest.approx(2100.0)


def test_count_today(state):
    now = datetime.now(timezone.utc)
    for _ in range(3):
        state.record_signal(_sig(ts=now))
    state.record_signal(_sig(ts=now - timedelta(days=2)))
    assert state.count_today() == 3


def test_count_today_by_tier(state):
    now = datetime.now(timezone.utc)
    state.record_signal(_sig(tier="STRONG", ts=now))
    state.record_signal(_sig(tier="WEAK", ts=now))
    state.record_signal(_sig(tier="WEAK", ts=now))
    assert state.count_today(tier="WEAK") == 2
    assert state.count_today(tier="STRONG") == 1


def test_last_weak_ts(state):
    assert state.last_weak_ts() is None
    state.record_signal(_sig(tier="WEAK"))
    assert state.last_weak_ts() is not None


def test_prune_old(state):
    old = datetime.now(timezone.utc) - timedelta(days=100)
    state.record_signal(_sig(ts=old))
    state.record_signal(_sig())
    removed = state.prune_old(days=90)
    assert removed == 1
    assert state.count_today() == 1
