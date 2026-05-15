"""Tests for daily/weekly paper-trading report (State.paper_report)."""

from __future__ import annotations

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
        "ai_action": "KEEP",
        "ai_risk_label": "CLEAN_SETUP",
        "ai_model_name": "path_c_lgb",
    }
    base.update(overrides)
    return base


def test_empty_report(state):
    rep = state.paper_report(days=1)
    assert rep["total"] == 0
    assert rep["active"] == 0
    assert rep["closed"] == 0
    assert rep["wins"] == 0
    assert rep["losses"] == 0
    assert rep["timeouts"] == 0
    assert rep["wr"] == 0.0
    assert rep["pf"] == 0.0
    assert rep["expectancy"] == 0.0
    assert rep["total_final_R"] == 0.0
    assert rep["max_adverse_R"] is None
    assert rep["by_stream"] == {}
    assert rep["by_risk"] == {}
    assert rep["by_action"] == {}
    assert rep["by_tier"] == {}


def test_report_with_active_only(state):
    state.record_signal(_sig())
    state.record_signal(_sig(direction="SELL", entry=2100.0, sl=2110.0,
                              tp1=2090.0))
    rep = state.paper_report(days=1)
    assert rep["total"] == 2
    assert rep["active"] == 2
    assert rep["closed"] == 0
    assert rep["wins"] == 0
    assert rep["losses"] == 0
    assert rep["timeouts"] == 0


def test_report_closed_tp_sl_timeout(state):
    # TP win
    sid = state.record_signal(_sig())
    state.update_lifecycle(sid, status="TP1_HIT", closed=True, final_R=1.5,
                            max_favorable_R=1.5, max_adverse_R=0.2)
    # SL loss
    sid = state.record_signal(_sig(direction="SELL"))
    state.update_lifecycle(sid, status="SL_HIT", closed=True, final_R=-1.0,
                            max_favorable_R=0.4, max_adverse_R=1.0)
    # TIMEOUT
    sid = state.record_signal(_sig())
    state.update_lifecycle(sid, status="TIMEOUT", closed=True, final_R=0.3,
                            max_favorable_R=0.6, max_adverse_R=0.4)

    rep = state.paper_report(days=1)
    assert rep["total"] == 3
    assert rep["closed"] == 3
    assert rep["active"] == 0
    assert rep["wins"] == 1
    assert rep["losses"] == 1
    assert rep["timeouts"] == 1
    assert rep["wr"] == pytest.approx(0.5)  # excludes timeout
    assert rep["pf"] == pytest.approx(1.5, abs=0.01)
    assert rep["total_final_R"] == pytest.approx(0.8, abs=0.01)
    assert rep["max_adverse_R"] == pytest.approx(1.0)


def test_breakdown_by_stream(state):
    for stream in ("intraday", "intraday", "swing", "scalp"):
        sid = state.record_signal(_sig(stream=stream))
        state.update_lifecycle(sid, status="TP1_HIT", closed=True,
                                final_R=1.0, max_favorable_R=1.0,
                                max_adverse_R=0.2)
    rep = state.paper_report(days=1)
    assert rep["by_stream"]["intraday"]["total"] == 2
    assert rep["by_stream"]["swing"]["total"] == 1
    assert rep["by_stream"]["scalp"]["total"] == 1
    assert rep["by_stream"]["intraday"]["wins"] == 2


def test_breakdown_by_risk_label(state):
    sid = state.record_signal(_sig(ai_risk_label="CLEAN_SETUP"))
    state.update_lifecycle(sid, status="TP1_HIT", closed=True, final_R=1.0)
    sid = state.record_signal(_sig(ai_risk_label="MEDIUM_RISK"))
    state.update_lifecycle(sid, status="SL_HIT", closed=True, final_R=-1.0)
    sid = state.record_signal(_sig(ai_risk_label="HIGH_RISK"))
    state.update_lifecycle(sid, status="SL_HIT", closed=True, final_R=-1.0)
    rep = state.paper_report(days=1)
    assert rep["by_risk"]["CLEAN_SETUP"]["wins"] == 1
    assert rep["by_risk"]["MEDIUM_RISK"]["losses"] == 1
    assert rep["by_risk"]["HIGH_RISK"]["losses"] == 1


def test_breakdown_by_action_and_tier(state):
    sid = state.record_signal(_sig(ai_action="KEEP", tier="STRONG"))
    state.update_lifecycle(sid, status="TP1_HIT", closed=True, final_R=1.5)
    sid = state.record_signal(_sig(ai_action="DOWNGRADE", tier="NORMAL"))
    state.update_lifecycle(sid, status="SL_HIT", closed=True, final_R=-1.0)
    sid = state.record_signal(_sig(ai_action="BLOCK", tier="WEAK"))
    state.update_lifecycle(sid, status="TIMEOUT", closed=True, final_R=-0.2)
    rep = state.paper_report(days=1)
    assert rep["by_action"]["KEEP"]["wins"] == 1
    assert rep["by_action"]["DOWNGRADE"]["losses"] == 1
    assert rep["by_action"]["BLOCK"]["timeouts"] == 1
    assert rep["by_tier"]["STRONG"]["wins"] == 1
    assert rep["by_tier"]["NORMAL"]["losses"] == 1
    assert rep["by_tier"]["WEAK"]["timeouts"] == 1


def test_weekly_window_excludes_older(state):
    now = datetime.now(timezone.utc)
    sid = state.record_signal(_sig(ts_utc=now.isoformat()))
    state.update_lifecycle(sid, status="TP1_HIT", closed=True, final_R=1.0)
    old = now - timedelta(days=10)
    sid = state.record_signal(_sig(ts_utc=old.isoformat()))
    state.update_lifecycle(sid, status="SL_HIT", closed=True, final_R=-1.0)
    rep_day = state.paper_report(days=1)
    rep_week = state.paper_report(days=7)
    assert rep_day["total"] == 1
    assert rep_week["total"] == 1
    assert rep_week["losses"] == 0


def test_formatter_empty():
    from xau_pro_bot.formatter import format_paper_report
    rep = {
        "period_days": 1, "total": 0, "active": 0, "closed": 0,
        "wins": 0, "losses": 0, "timeouts": 0, "wr": 0.0, "pf": 0.0,
        "expectancy": 0.0, "total_final_R": 0.0, "max_adverse_R": None,
        "by_stream": {}, "by_risk": {}, "by_action": {}, "by_tier": {},
    }
    out = format_paper_report(rep)
    assert "0" in out
    assert "пуст" in out.lower() or "недостаточно" in out.lower() \
        or "нет данных" in out.lower()


def test_formatter_small_sample_warning():
    from xau_pro_bot.formatter import format_paper_report
    rep = {
        "period_days": 1, "total": 2, "active": 1, "closed": 1,
        "wins": 1, "losses": 0, "timeouts": 0, "wr": 1.0, "pf": 0.0,
        "expectancy": 1.0, "total_final_R": 1.0, "max_adverse_R": 0.2,
        "by_stream": {"intraday": {"total": 1, "wins": 1, "losses": 0,
                                     "timeouts": 0, "sum_R": 1.0}},
        "by_risk": {}, "by_action": {}, "by_tier": {},
    }
    out = format_paper_report(rep)
    assert "недостаточно" in out.lower() or "small sample" in out.lower() \
        or "мало данных" in out.lower()


def test_formatter_full_report():
    from xau_pro_bot.formatter import format_paper_report
    rep = {
        "period_days": 7, "total": 30, "active": 5, "closed": 25,
        "wins": 15, "losses": 8, "timeouts": 2, "wr": 0.65, "pf": 1.8,
        "expectancy": 0.35, "total_final_R": 8.75, "max_adverse_R": 1.4,
        "by_stream": {
            "intraday": {"total": 15, "wins": 9, "losses": 5,
                          "timeouts": 1, "sum_R": 4.5},
            "swing":    {"total": 10, "wins": 5, "losses": 3,
                          "timeouts": 1, "sum_R": 3.0},
        },
        "by_risk": {
            "CLEAN_SETUP": {"total": 12, "wins": 9, "losses": 2,
                             "timeouts": 1, "sum_R": 6.0},
        },
        "by_action": {
            "KEEP": {"total": 20, "wins": 12, "losses": 6,
                      "timeouts": 1, "sum_R": 7.0},
        },
        "by_tier": {
            "STRONG": {"total": 18, "wins": 12, "losses": 4,
                        "timeouts": 1, "sum_R": 7.5},
        },
    }
    out = format_paper_report(rep)
    assert "7" in out
    assert "30" in out
    assert "intraday" in out
    assert "CLEAN_SETUP" in out
    assert "KEEP" in out
    assert "STRONG" in out
    assert "1.8" in out  # pf
