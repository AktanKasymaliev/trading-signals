from datetime import datetime, timezone

import pytest

from xau_pro_bot.formatter import (
    format_strong_signal, format_weak_signal,
    format_no_signal_killzone,
)


def _sig(tier="STRONG", tp2_unavailable=False):
    return {
        "direction": "SELL", "tier": tier, "score": 81,
        "entry": 3312.50, "sl": 3324.00,
        "tp1": 3298.00, "tp2": 3280.00 if not tp2_unavailable else None,
        "tp3": 3261.00, "rr": 2.8,
        "killzone": "London KZ",
        "tp2_unavailable": tp2_unavailable,
        "reasons": {
            "ict": ["OTE zone"], "smc": ["CHOCH H4"],
            "macro": ["Wyckoff Distribution"],
            "classic": ["RSI 72"], "penalties": [],
        },
        "ts_utc": datetime(2026, 5, 11, 9, 47, tzinfo=timezone.utc),
    }


def test_strong_signal_contains_required_fields():
    text = format_strong_signal(_sig())
    assert "Сильный сигнал" in text
    assert "SELL" in text
    assert "3,312.50" in text
    assert "Score: 81/100" in text
    assert "Уверенность" not in text


def test_strong_signal_with_tp2_unavailable():
    text = format_strong_signal(_sig(tp2_unavailable=True))
    assert "TP2: недоступен" in text


def test_weak_signal_short_format():
    text = format_weak_signal(_sig(tier="WEAK"))
    assert "TP1" in text
    assert "Анализ:" not in text


def test_no_signal_brief():
    text = format_no_signal_killzone(
        killzone="London KZ", price=3298.10, rsi=52.0,
    )
    assert "London KZ" in text
    assert "3,298.10" in text


def test_strong_signal_shows_strategy_and_horizon():
    sig = _sig()
    sig["strategy_label"] = "Swing 500"
    sig["horizon_label"] = "2-7 дней"
    text = format_strong_signal(sig)
    assert "Стратегия: Swing 500" in text
    assert "Горизонт: 2-7 дней" in text


def test_strong_signal_shows_compact_ai_line_when_enabled(monkeypatch):
    # Pin AI_EXPLAIN off so we exercise the legacy single-line path
    # regardless of .env state.
    from xau_pro_bot import config as _cfg
    monkeypatch.delenv("AI_EXPLAIN", raising=False)
    monkeypatch.setattr(_cfg, "AI_EXPLAIN", False, raising=False)

    sig = _sig()
    sig.update({
        "ai_enabled": True,
        "ai_direction": "BUY",
        "ai_confidence": 0.72,
        "ai_reason": "AI agrees with deterministic signal",
        "ai_blocked": False,
    })

    text = format_strong_signal(sig)

    assert "AI: BUY 0.72 confidence — AI agrees with deterministic signal" in text


def test_weak_signal_shows_compact_ai_line_when_enabled(monkeypatch):
    from xau_pro_bot import config as _cfg
    monkeypatch.delenv("AI_EXPLAIN", raising=False)
    monkeypatch.setattr(_cfg, "AI_EXPLAIN", False, raising=False)

    sig = _sig()
    sig.update({
        "ai_enabled": True,
        "ai_direction": "SELL",
        "ai_confidence": 0.66,
        "ai_reason": "AI conflicts with deterministic signal",
        "ai_blocked": False,
    })

    text = format_weak_signal(sig)

    assert "AI: SELL 0.66 confidence — AI conflicts with deterministic signal" in text


def test_strong_signal_omits_ai_line_when_disabled():
    sig = _sig()
    text = format_strong_signal(sig)
    assert "AI:" not in text
