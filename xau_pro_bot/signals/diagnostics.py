"""Live-bot diagnostics helpers: no-signal dedup + scan ring buffer.

Kept as pure functions so they can be tested without spinning up the bot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def no_signal_fingerprint(
    *,
    stream: str | None,
    killzone: str | None,
    price: float | None,
    bucket_size: float,
) -> tuple:
    """Bucket no-signal updates by stream/session/price slot."""
    if price is None or bucket_size <= 0:
        bucket = 0
    else:
        bucket = int(round(float(price) / bucket_size))
    return (stream or "intraday", killzone or "—", bucket)


def should_send_no_signal(
    cache: dict[tuple, datetime],
    fingerprint: tuple,
    *,
    now: datetime,
    window_minutes: int,
) -> bool:
    """Return True if a no-signal update with this fingerprint may be sent.

    Mutates ``cache`` on True to record the send.
    """
    horizon = now - timedelta(minutes=max(0, window_minutes))
    last = cache.get(fingerprint)
    if last is not None and last >= horizon:
        return False
    cache[fingerprint] = now
    return True


def minutes_since_last_no_signal(
    cache: dict[tuple, datetime],
    fingerprint: tuple,
    *,
    now: datetime,
) -> int | None:
    """Return integer minutes since the last send of ``fingerprint``, or None."""
    last = cache.get(fingerprint)
    if last is None:
        return None
    delta = now - last
    return max(0, int(delta.total_seconds() // 60))


def prune_no_signal_cache(
    cache: dict[tuple, datetime],
    *,
    now: datetime,
    window_minutes: int,
) -> None:
    horizon = now - timedelta(minutes=max(0, window_minutes) * 2)
    stale = [k for k, ts in cache.items() if ts < horizon]
    for k in stale:
        cache.pop(k, None)


def diag_fields(sig: dict[str, Any]) -> dict[str, Any]:
    """Extract bias-debug fields from a signal dict (safe on partial dicts)."""
    reasons = sig.get("reasons") or {}
    block_reason = None
    if sig.get("ai_blocked"):
        block_reason = sig.get("ai_reason") or sig.get("ai_reason_short")
    return {
        "ts": (sig.get("ts_utc").isoformat()
               if isinstance(sig.get("ts_utc"), datetime) else sig.get("ts_utc")),
        "stream": sig.get("stream", "intraday"),
        "tier": sig.get("tier"),
        "bull_score": sig.get("bull_score"),
        "bear_score": sig.get("bear_score"),
        "net_bull": (sig.get("bull_score") or 0.0) - (sig.get("bear_score") or 0.0)
        if sig.get("bull_score") is not None or sig.get("bear_score") is not None
        else None,
        "net_bear": (sig.get("bear_score") or 0.0) - (sig.get("bull_score") or 0.0)
        if sig.get("bull_score") is not None or sig.get("bear_score") is not None
        else None,
        "deterministic_direction": sig.get("direction"),
        "ai_direction": sig.get("ai_direction"),
        "ai_confidence": sig.get("ai_confidence"),
        "ai_action": sig.get("ai_action"),
        "ai_blocked": sig.get("ai_blocked"),
        "block_reason": block_reason,
        "final_direction": sig.get("direction") if sig.get("tier") != "NO_SIGNAL" else None,
        "killzone": sig.get("killzone"),
        "ai_reasons": (reasons.get("ai") or [None])[0] if reasons.get("ai") else None,
    }


def summarize_bias(diags: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ring buffer for /debug_bias output."""
    if not diags:
        return {
            "n": 0, "buy": 0, "sell": 0, "no_signal": 0,
            "ai_blocked_buy": 0, "ai_blocked_sell": 0,
            "avg_bull": 0.0, "avg_bear": 0.0,
            "top_sell_skip_reasons": [],
        }
    buy = sum(1 for d in diags if d.get("deterministic_direction") == "BUY")
    sell = sum(1 for d in diags if d.get("deterministic_direction") == "SELL")
    no_sig = sum(1 for d in diags if d.get("tier") == "NO_SIGNAL")
    ai_blocked_buy = sum(
        1 for d in diags
        if d.get("ai_blocked") and d.get("deterministic_direction") == "BUY"
    )
    ai_blocked_sell = sum(
        1 for d in diags
        if d.get("ai_blocked") and d.get("deterministic_direction") == "SELL"
    )
    bulls = [d.get("bull_score") for d in diags if d.get("bull_score") is not None]
    bears = [d.get("bear_score") for d in diags if d.get("bear_score") is not None]
    avg_bull = sum(bulls) / len(bulls) if bulls else 0.0
    avg_bear = sum(bears) / len(bears) if bears else 0.0
    sell_reasons: dict[str, int] = {}
    for d in diags:
        if d.get("deterministic_direction") == "SELL" and d.get("tier") == "NO_SIGNAL":
            key = d.get("block_reason") or d.get("ai_reasons") or "deterministic_below_threshold"
            sell_reasons[key] = sell_reasons.get(key, 0) + 1
    top = sorted(sell_reasons.items(), key=lambda kv: -kv[1])[:5]
    return {
        "n": len(diags),
        "buy": buy,
        "sell": sell,
        "no_signal": no_sig,
        "ai_blocked_buy": ai_blocked_buy,
        "ai_blocked_sell": ai_blocked_sell,
        "avg_bull": round(avg_bull, 2),
        "avg_bear": round(avg_bear, 2),
        "top_sell_skip_reasons": top,
    }


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
