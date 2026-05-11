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


# ── R3 multi-stream tests ─────────────────────────────
import sqlite3

from xau_pro_bot.state import State as _State


def test_migration_adds_stream_column(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL, direction TEXT NOT NULL, tier TEXT NOT NULL,
            score INTEGER NOT NULL, entry REAL NOT NULL, sl REAL NOT NULL,
            tp1 REAL, tp2 REAL, tp3 REAL, rr REAL,
            killzone TEXT, reasons_json TEXT
        );
    """)
    conn.execute(
        "INSERT INTO signals (ts_utc, direction, tier, score, entry, sl) "
        "VALUES (?, 'BUY', 'STRONG', 70, 2000, 1995)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()

    st = _State(db_path=db_path)
    cols = [r["name"] for r in st._conn.execute(
        "PRAGMA table_info(signals)").fetchall()]
    assert "stream" in cols
    row = st._conn.execute("SELECT stream FROM signals LIMIT 1").fetchone()
    assert row["stream"] == "intraday"
    st.close()


def test_record_signal_with_stream(state):
    sig = _sig()
    sig["stream"] = "swing"
    sid = state.record_signal(sig)
    assert sid > 0
    last = state.last_signal(stream="swing")
    assert last is not None
    assert last["stream"] == "swing"
    assert state.last_signal(stream="intraday") is None


def test_count_today_by_stream(state):
    base = _sig()
    state.record_signal({**base, "stream": "intraday"})
    state.record_signal({**base, "stream": "intraday"})
    state.record_signal({**base, "stream": "swing"})
    assert state.count_today(stream="intraday") == 2
    assert state.count_today(stream="swing") == 1
    assert state.count_today() == 3


def test_last_weak_ts_per_stream(state):
    base = _sig(tier="WEAK")
    state.record_signal({**base, "stream": "intraday"})
    assert state.last_weak_ts(stream="intraday") is not None
    assert state.last_weak_ts(stream="swing") is None


def test_last_scalp_ts(state):
    assert state.last_scalp_ts() is None
    sig = _sig()
    sig["stream"] = "scalp"
    state.record_signal(sig)
    assert state.last_scalp_ts() is not None
