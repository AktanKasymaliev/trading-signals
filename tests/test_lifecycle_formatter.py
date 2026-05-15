"""Formatter tests for /active and /history Telegram replies."""

from __future__ import annotations

from datetime import datetime, timezone

from xau_pro_bot.formatter import (
    format_active_signals,
    format_history,
    format_lifecycle_transition,
)


def _row(**overrides) -> dict:
    base = {
        "id": 1,
        "ts_utc": "2026-05-15T10:00:00+00:00",
        "stream": "intraday",
        "direction": "BUY",
        "entry": 2000.0,
        "sl": 1990.0,
        "tp1": 2010.0,
        "tp2": 2020.0,
        "tp3": 2030.0,
        "status": "ACTIVE",
        "max_favorable_R": 0.4,
        "max_adverse_R": 0.2,
        "final_R": None,
        "closed_at": None,
        "ai_risk_label": "CLEAN_SETUP",
    }
    base.update(overrides)
    return base


def test_format_active_empty():
    out = format_active_signals([])
    assert "нет" in out.lower() or "no active" in out.lower()


def test_format_active_lists_signals():
    rows = [
        _row(id=1, direction="BUY", status="ACTIVE"),
        _row(id=2, direction="SELL", status="TP1_HIT", entry=2100.0,
             sl=2110.0, tp1=2090.0),
    ]
    out = format_active_signals(rows)
    assert "BUY" in out
    assert "SELL" in out
    assert "TP1_HIT" in out
    assert "ACTIVE" in out
    assert "#1" in out and "#2" in out


def test_format_history_empty():
    out = format_history([])
    assert "пуст" in out.lower() or "empty" in out.lower()


def test_format_history_lists_rows():
    rows = [
        _row(id=10, status="TP2_HIT", final_R=2.0,
             closed_at="2026-05-15T11:00:00+00:00"),
        _row(id=9, status="SL_HIT", final_R=-1.0,
             closed_at="2026-05-15T10:30:00+00:00"),
    ]
    out = format_history(rows)
    assert "TP2_HIT" in out
    assert "SL_HIT" in out
    assert "+2.0" in out or "2.00" in out
    assert "-1.0" in out or "-1.00" in out


def test_format_lifecycle_transition_tp1():
    msg = format_lifecycle_transition(
        signal_id=5,
        direction="BUY",
        old_status="ACTIVE",
        new_status="TP1_HIT",
        closed=False,
        final_R=None,
        entry=2000.0,
        price=2010.0,
    )
    assert "TP1_HIT" in msg
    assert "#5" in msg
    assert "BUY" in msg


def test_format_lifecycle_transition_sl_close():
    msg = format_lifecycle_transition(
        signal_id=7,
        direction="SELL",
        old_status="ACTIVE",
        new_status="SL_HIT",
        closed=True,
        final_R=-1.0,
        entry=2100.0,
        price=2110.0,
    )
    assert "SL_HIT" in msg
    assert "-1" in msg
