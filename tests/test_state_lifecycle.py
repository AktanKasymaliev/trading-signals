"""State extensions for signal lifecycle (active, history, stats)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from xau_pro_bot.state import State


@pytest.fixture
def state(tmp_path):
    return State(db_path=str(tmp_path / "test.db"))


def _sig(**overrides) -> dict:
    base = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "direction": "BUY",
        "tier": "STRONG",
        "score": 70,
        "entry": 2000.0,
        "sl": 1990.0,
        "tp1": 2010.0,
        "tp2": 2020.0,
        "tp3": 2030.0,
        "rr": 2.0,
        "killzone": "London KZ",
        "reasons_json": "{}",
        "stream": "intraday",
        "ai_action": "PASS",
        "ai_risk_label": "CLEAN_SETUP",
        "ai_model_name": "path_c_lgb",
    }
    base.update(overrides)
    return base


def test_record_signal_persists_ai_fields(state):
    sid = state.record_signal(_sig())
    row = state._conn.execute(
        "SELECT ai_action, ai_risk_label, ai_model_name, status "
        "FROM signals WHERE id = ?", (sid,)).fetchone()
    assert row["ai_action"] == "PASS"
    assert row["ai_risk_label"] == "CLEAN_SETUP"
    assert row["ai_model_name"] == "path_c_lgb"
    assert row["status"] == "ACTIVE"


def test_get_active_returns_only_active(state):
    sid1 = state.record_signal(_sig())
    sid2 = state.record_signal(_sig(direction="SELL", entry=2100.0,
                                     sl=2110.0, tp1=2090.0))
    state.update_lifecycle(sid2, status="SL_HIT", closed=True,
                            final_R=-1.0, max_favorable_R=0.5,
                            max_adverse_R=1.0)
    active = state.get_active()
    assert len(active) == 1
    assert active[0]["id"] == sid1
    assert active[0]["status"] == "ACTIVE"


def test_update_lifecycle_persists_fields(state):
    sid = state.record_signal(_sig())
    state.update_lifecycle(sid, status="TP1_HIT", closed=False,
                            max_favorable_R=1.2, max_adverse_R=0.3)
    row = state._conn.execute(
        "SELECT status, closed_at, max_favorable_R, max_adverse_R, final_R "
        "FROM signals WHERE id = ?", (sid,)).fetchone()
    assert row["status"] == "TP1_HIT"
    assert row["closed_at"] is None
    assert row["max_favorable_R"] == pytest.approx(1.2)
    assert row["max_adverse_R"] == pytest.approx(0.3)


def test_update_lifecycle_closes_trade(state):
    sid = state.record_signal(_sig())
    state.update_lifecycle(sid, status="SL_HIT", closed=True,
                            final_R=-1.0, max_favorable_R=0.4,
                            max_adverse_R=1.0)
    row = state._conn.execute(
        "SELECT status, closed_at, final_R FROM signals WHERE id = ?",
        (sid,)).fetchone()
    assert row["status"] == "SL_HIT"
    assert row["closed_at"] is not None
    assert row["final_R"] == pytest.approx(-1.0)


def test_recent_closed_returns_last_n(state):
    for i in range(12):
        sid = state.record_signal(_sig(entry=2000.0 + i))
        state.update_lifecycle(sid, status="TP1_HIT", closed=True,
                                final_R=1.0, max_favorable_R=1.0,
                                max_adverse_R=0.2)
    recent = state.recent_closed(limit=10)
    assert len(recent) == 10
    # ordered newest first
    assert recent[0]["id"] > recent[-1]["id"]


def test_lifecycle_stats_by_risk_label(state):
    for _ in range(3):
        sid = state.record_signal(_sig(ai_risk_label="CLEAN_SETUP"))
        state.update_lifecycle(sid, status="TP1_HIT", closed=True,
                                final_R=1.0, max_favorable_R=1.0,
                                max_adverse_R=0.2)
    for _ in range(2):
        sid = state.record_signal(_sig(ai_risk_label="HIGH_RISK"))
        state.update_lifecycle(sid, status="SL_HIT", closed=True,
                                final_R=-1.0, max_favorable_R=0.3,
                                max_adverse_R=1.0)
    stats = state.lifecycle_stats_by_risk()
    assert stats["CLEAN_SETUP"]["wins"] == 3
    assert stats["CLEAN_SETUP"]["losses"] == 0
    assert stats["HIGH_RISK"]["wins"] == 0
    assert stats["HIGH_RISK"]["losses"] == 2


def test_lifecycle_stats_window(state):
    now = datetime.now(timezone.utc)
    # 3 wins today
    for _ in range(3):
        sid = state.record_signal(_sig(ts_utc=now.isoformat()))
        state.update_lifecycle(sid, status="TP1_HIT", closed=True,
                                final_R=1.5, max_favorable_R=1.5,
                                max_adverse_R=0.2)
    # 1 loss 2 days ago
    old = now - timedelta(days=2)
    sid = state.record_signal(_sig(ts_utc=old.isoformat()))
    state.update_lifecycle(sid, status="SL_HIT", closed=True,
                            final_R=-1.0, max_favorable_R=0.3,
                            max_adverse_R=1.0)
    today = state.lifecycle_metrics(days=1)
    assert today["wins"] == 3
    assert today["losses"] == 0
    assert today["wr"] == pytest.approx(1.0)
    assert today["expectancy"] == pytest.approx(1.5)
    week = state.lifecycle_metrics(days=7)
    assert week["wins"] == 3
    assert week["losses"] == 1
    assert week["wr"] == pytest.approx(0.75)
    assert week["pf"] == pytest.approx(4.5, abs=0.01)


def test_lifecycle_migration_adds_columns(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL, direction TEXT NOT NULL, tier TEXT NOT NULL,
            score INTEGER NOT NULL, entry REAL NOT NULL, sl REAL NOT NULL,
            tp1 REAL, tp2 REAL, tp3 REAL, rr REAL,
            killzone TEXT, reasons_json TEXT, stream TEXT
        );
    """)
    conn.execute(
        "INSERT INTO signals (ts_utc, direction, tier, score, entry, sl, stream) "
        "VALUES (?, 'BUY', 'STRONG', 70, 2000, 1995, 'intraday')",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()

    st = State(db_path=db_path)
    cols = [r["name"] for r in st._conn.execute(
        "PRAGMA table_info(signals)").fetchall()]
    for c in ("status", "closed_at", "max_favorable_R", "max_adverse_R",
              "final_R", "ai_action", "ai_risk_label", "ai_model_name"):
        assert c in cols, f"missing column {c}"
    row = st._conn.execute(
        "SELECT status FROM signals LIMIT 1").fetchone()
    assert row["status"] == "ACTIVE"
    st.close()
