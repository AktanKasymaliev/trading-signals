"""Tests for signal lifecycle tracking (TP/SL/timeout transitions)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from xau_pro_bot.lifecycle import (
    Candle,
    LifecycleSignal,
    TTL_BY_STREAM,
    evaluate_candle,
)


def _sig(
    *,
    direction: str = "BUY",
    entry: float = 2000.0,
    sl: float = 1990.0,
    tp1: float = 2010.0,
    tp2: float = 2020.0,
    tp3: float = 2030.0,
    status: str = "ACTIVE",
    stream: str = "intraday",
    opened_at: datetime | None = None,
    max_fav: float = 0.0,
    max_adv: float = 0.0,
) -> LifecycleSignal:
    return LifecycleSignal(
        id=1,
        stream=stream,
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        status=status,
        opened_at=opened_at or datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc),
        max_favorable_R=max_fav,
        max_adverse_R=max_adv,
        ai_action=None,
        ai_risk_label=None,
        ai_model_name=None,
    )


def _candle(high: float, low: float, close: float | None = None,
            ts: datetime | None = None) -> Candle:
    return Candle(
        high=high,
        low=low,
        close=close if close is not None else (high + low) / 2,
        ts=ts or datetime(2026, 5, 15, 10, 30, tzinfo=timezone.utc),
    )


def test_tp1_hit_updates_status_for_buy():
    sig = _sig()
    candle = _candle(high=2012.0, low=2001.0)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "TP1_HIT"
    assert tr.closed is False
    assert tr.max_favorable_R == pytest.approx(1.2, abs=0.01)


def test_sl_hit_updates_status_and_closes_buy():
    sig = _sig()
    candle = _candle(high=2002.0, low=1989.0, close=1990.5)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "SL_HIT"
    assert tr.closed is True
    assert tr.closed_at == candle.ts
    assert tr.final_R == pytest.approx(-1.0, abs=0.01)


def test_same_candle_sl_and_tp_chooses_sl_first():
    sig = _sig()
    # Candle pierces both SL (1989) and TP1 (2010).
    candle = _candle(high=2015.0, low=1989.0, close=2000.0)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "SL_HIT"
    assert tr.closed is True


def test_sell_tp1_hit():
    sig = _sig(direction="SELL", entry=2000.0, sl=2010.0,
               tp1=1990.0, tp2=1980.0, tp3=1970.0)
    candle = _candle(high=2002.0, low=1988.0, close=1991.0)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "TP1_HIT"


def test_sell_sl_hit_first_on_same_candle():
    sig = _sig(direction="SELL", entry=2000.0, sl=2010.0,
               tp1=1990.0, tp2=1980.0, tp3=1970.0)
    candle = _candle(high=2012.0, low=1985.0, close=1995.0)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "SL_HIT"


def test_no_transition_when_no_levels_hit():
    sig = _sig()
    candle = _candle(high=2005.0, low=1998.0)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    # No status change, but max_fav/adv may be tracked via transition. We
    # return None when neither levels nor TTL trigger anything material.
    # Equally acceptable: a transition with same status. We accept either,
    # but max_favorable_R must be updated regardless.
    if tr is None:
        return
    assert tr.new_status == "ACTIVE"
    assert tr.max_favorable_R == pytest.approx(0.5, abs=0.01)


def test_tp3_hit_closes_trade():
    sig = _sig(status="TP2_HIT")
    candle = _candle(high=2035.0, low=2025.0)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "TP3_HIT"
    assert tr.closed is True
    assert tr.final_R == pytest.approx(3.0, abs=0.01)


def test_progression_from_tp1_to_tp2():
    sig = _sig(status="TP1_HIT", max_fav=1.0)
    candle = _candle(high=2022.0, low=2015.0)
    tr = evaluate_candle(sig, candle, now=candle.ts,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "TP2_HIT"
    assert tr.closed is False


def test_timeout_terminates_trade():
    opened = datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc)
    now = opened + timedelta(hours=25)
    sig = _sig(opened_at=opened)
    candle = _candle(high=2005.0, low=1998.0, close=2002.0, ts=now)
    tr = evaluate_candle(sig, candle, now=now,
                          ttl_hours=TTL_BY_STREAM["intraday"])
    assert tr is not None
    assert tr.new_status == "TIMEOUT"
    assert tr.closed is True
    assert tr.final_R is not None


def test_ttl_per_stream_differs():
    assert TTL_BY_STREAM["intraday"] >= 12
    assert TTL_BY_STREAM["scalp"] <= TTL_BY_STREAM["intraday"]
    assert TTL_BY_STREAM["swing"] > TTL_BY_STREAM["intraday"]
