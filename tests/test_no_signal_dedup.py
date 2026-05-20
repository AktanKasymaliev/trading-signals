"""Tests for no-signal Telegram dedup helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from xau_pro_bot.signals.diagnostics import (
    no_signal_fingerprint,
    prune_no_signal_cache,
    should_send_no_signal,
    summarize_bias,
)


def test_no_signal_dedup_blocks_within_window():
    cache: dict = {}
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    fp = no_signal_fingerprint(
        stream="intraday", killzone="London KZ", price=3300.0, bucket_size=2.0,
    )
    assert should_send_no_signal(cache, fp, now=now, window_minutes=30)
    assert not should_send_no_signal(
        cache, fp, now=now + timedelta(minutes=15), window_minutes=30,
    )


def test_no_signal_dedup_allows_after_window():
    cache: dict = {}
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    fp = no_signal_fingerprint(
        stream="intraday", killzone="London KZ", price=3300.0, bucket_size=2.0,
    )
    assert should_send_no_signal(cache, fp, now=now, window_minutes=30)
    later = now + timedelta(minutes=31)
    assert should_send_no_signal(cache, fp, now=later, window_minutes=30)


def test_no_signal_dedup_different_price_bucket_allowed():
    cache: dict = {}
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    fp1 = no_signal_fingerprint(
        stream="intraday", killzone="London KZ", price=3300.0, bucket_size=2.0,
    )
    fp2 = no_signal_fingerprint(
        stream="intraday", killzone="London KZ", price=3320.0, bucket_size=2.0,
    )
    assert fp1 != fp2
    assert should_send_no_signal(cache, fp1, now=now, window_minutes=30)
    assert should_send_no_signal(cache, fp2, now=now, window_minutes=30)


def test_no_signal_dedup_different_killzone_allowed():
    cache: dict = {}
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    fp1 = no_signal_fingerprint(
        stream="intraday", killzone="London KZ", price=3300.0, bucket_size=2.0,
    )
    fp2 = no_signal_fingerprint(
        stream="intraday", killzone="NY AM KZ", price=3300.0, bucket_size=2.0,
    )
    assert should_send_no_signal(cache, fp1, now=now, window_minutes=30)
    assert should_send_no_signal(cache, fp2, now=now, window_minutes=30)


def test_prune_removes_old_entries():
    cache: dict = {}
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    fp = no_signal_fingerprint(
        stream="intraday", killzone="London KZ", price=3300.0, bucket_size=2.0,
    )
    should_send_no_signal(cache, fp, now=now, window_minutes=30)
    assert fp in cache
    prune_no_signal_cache(
        cache, now=now + timedelta(minutes=120), window_minutes=30,
    )
    assert fp not in cache


def test_summarize_bias_counts_directions():
    diags = [
        {"deterministic_direction": "BUY", "tier": "STRONG",
         "ai_blocked": False, "bull_score": 70, "bear_score": 30},
        {"deterministic_direction": "SELL", "tier": "NO_SIGNAL",
         "ai_blocked": True, "ai_reasons": None,
         "block_reason": "AI conflicts: confidence too low",
         "bull_score": 30, "bear_score": 45},
        {"deterministic_direction": "SELL", "tier": "NO_SIGNAL",
         "ai_blocked": True, "ai_reasons": None,
         "block_reason": "AI conflicts: confidence too low",
         "bull_score": 25, "bear_score": 48},
    ]
    out = summarize_bias(diags)
    assert out["n"] == 3
    assert out["buy"] == 1
    assert out["sell"] == 2
    assert out["no_signal"] == 2
    assert out["ai_blocked_sell"] == 2
    assert out["top_sell_skip_reasons"][0][0] == "AI conflicts: confidence too low"
    assert out["top_sell_skip_reasons"][0][1] == 2


def test_summarize_bias_empty():
    assert summarize_bias([])["n"] == 0
