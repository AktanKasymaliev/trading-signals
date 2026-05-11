"""Telegram Markdown signal formatter (Russian, matches spec template)."""

from __future__ import annotations

from datetime import datetime


KZ_FLAGS = {
    "London KZ": "🇬🇧",
    "NY AM KZ": "🇺🇸",
    "NY PM KZ": "🇺🇸",
    "Asian KZ": "🇯🇵",
}


def _fmt_price(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:,.2f}"


def _fmt_pts(diff: float) -> str:
    return f"{diff:+.1f} pts"


def _tp2_line(sig: dict) -> str:
    if sig.get("tp2_unavailable") or sig.get("tp2") is None:
        return " •  TP2: недоступен (RR < 1.8)"
    diff = sig["tp2"] - sig["entry"]
    return f" •  TP2: `{_fmt_price(sig['tp2'])}` ({abs(diff):.1f} pts) — ликвидность"


def _direction_header(sig: dict) -> str:
    if sig["direction"] == "BUY":
        return "🟢 Сильный сигнал — BUY"
    return "🔴 Сильный сигнал — SELL"


def _analysis_block(sig: dict) -> str:
    lines = ["📐 Анализ:"]
    for source in ("ict", "smc", "macro", "classic"):
        for r in sig["reasons"].get(source, []):
            lines.append(f"• {r} ✅")
    for r in sig["reasons"].get("penalties", []):
        lines.append(f"• {r} ⚠️")
    return "\n".join(lines)


def format_strong_signal(sig: dict) -> str:
    flag = KZ_FLAGS.get(sig.get("killzone") or "", "")
    sl_diff = sig["sl"] - sig["entry"]
    tp1_diff = (sig["tp1"] - sig["entry"]) if sig["tp1"] is not None else 0
    tp3_diff = (sig["tp3"] - sig["entry"]) if sig["tp3"] is not None else 0
    ts: datetime = sig["ts_utc"]

    parts = [
        _direction_header(sig),
        "━━━━━━━━━━━━━━━━━━━",
        f"🔹 Вход: `{_fmt_price(sig['entry'])}`",
        f"🔺 Stop Loss: `{_fmt_price(sig['sl'])}` ({_fmt_pts(sl_diff)})",
        "🎯 Цели:",
        f" •  TP1: `{_fmt_price(sig['tp1'])}` ({abs(tp1_diff):.1f} pts) — FVG",
        _tp2_line(sig),
        f" •  TP3: `{_fmt_price(sig['tp3'])}` ({abs(tp3_diff):.1f} pts) — D1",
        "━━━━━━━━━━━━━━━━━━━",
        f"📊 R:R → 1:{sig['rr']:.1f}",
        f"🧠 Score: {sig['score']}/100",
        f"⏱ Сессия: {sig.get('killzone') or '—'} {flag} | M15→H1",
    ]
    if sig.get("strategy_label"):
        parts.append(f"📐 Стратегия: {sig['strategy_label']}")
    if sig.get("horizon_label"):
        parts.append(f"⏳ Горизонт: {sig['horizon_label']}")
    parts.extend([
        "━━━━━━━━━━━━━━━━━━━",
        _analysis_block(sig),
        "━━━━━━━━━━━━━━━━━━━",
        f"🕐 {ts.strftime('%d.%m.%Y %H:%M')} UTC",
    ])
    return "\n".join(parts)


def format_weak_signal(sig: dict) -> str:
    flag = KZ_FLAGS.get(sig.get("killzone") or "", "")
    sl_diff = sig["sl"] - sig["entry"]
    parts = [
        f"⚠️ Слабый сигнал — {sig['direction']}",
        f"🔹 Вход: `{_fmt_price(sig['entry'])}`",
        f"🔺 SL: `{_fmt_price(sig['sl'])}` ({_fmt_pts(sl_diff)})",
        f"🎯 TP1: `{_fmt_price(sig['tp1'])}`",
        _tp2_line(sig),
        f"🧠 Score: {sig['score']}/100",
        f"⏱ {sig.get('killzone') or '—'} {flag}",
    ]
    if sig.get("strategy_label"):
        parts.append(f"📐 Стратегия: {sig['strategy_label']}")
    if sig.get("horizon_label"):
        parts.append(f"⏳ Горизонт: {sig['horizon_label']}")
    return "\n".join(parts)


def format_no_signal_killzone(killzone: str, price: float,
                              rsi: float | None) -> str:
    rsi_text = f"{rsi:.0f}" if rsi is not None else "—"
    return (
        f"⏳ {killzone} | Нет сигнала\n"
        f"💰 XAU: `{_fmt_price(price)}` | RSI H1: {rsi_text}"
    )


def format_status(snapshot: dict) -> str:
    """Generic /status response builder."""
    lines = [
        "📊 Market Status",
        f"💰 XAU/USD: `{_fmt_price(snapshot.get('price'))}`",
        f"🕐 Killzone: {snapshot.get('killzone') or 'none'}",
        f"📈 D1 trend: {snapshot.get('d1_trend', '—')}",
        f"📊 H4 structure: {snapshot.get('h4_structure', '—')}",
        f"🌀 Wyckoff: {snapshot.get('wyckoff', '—')}",
    ]
    return "\n".join(lines)
