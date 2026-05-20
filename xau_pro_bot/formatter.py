"""Telegram Markdown signal formatter (Russian, matches spec template)."""

from __future__ import annotations

from datetime import datetime

from xau_pro_bot import config


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


_RISK_LABEL_RU = {
    "HIGH_RISK": "HIGH",
    "MEDIUM_RISK": "MEDIUM",
    "CLEAN_SETUP": "CLEAN",
}

_AI_ACTION_LABEL = {
    "KEEP": "KEEP",
    "BLOCK": "BLOCK",
    "DOWNGRADE": "DOWNGRADE",
    "NEUTRAL": "NEUTRAL",
}


def _ai_explain_enabled() -> bool:
    # Read live so tests/monkeypatch can flip the flag at runtime.
    import os
    return os.getenv("AI_EXPLAIN", "false").strip().lower() in {
        "1", "true", "yes", "on",
    } or getattr(config, "AI_EXPLAIN", False)


def _ai_block(sig: dict) -> list[str]:
    """Multi-line analysis-assistant AI block (AI_EXPLAIN=true)."""
    if not sig.get("ai_enabled"):
        return []
    raw_action = sig.get("ai_action") or "—"
    action = _AI_ACTION_LABEL.get(raw_action, raw_action)
    model = sig.get("ai_model_name") or "AI"
    risk = _RISK_LABEL_RU.get(sig.get("ai_risk_label") or "", "—")
    reason = sig.get("ai_reason_short") or sig.get("ai_reason") or "—"
    return [
        f"🧠 AI filter: {action}",
        f"Модель: {model}",
        f"Риск: {risk}",
        f"Причина: {reason}",
    ]


def _ai_line(sig: dict) -> str | None:
    if not sig.get("ai_enabled"):
        return None
    direction = sig.get("ai_direction") or "NO_TRADE"
    confidence = sig.get("ai_confidence")
    reason = sig.get("ai_reason") or "AI checked"
    conf_text = f"{float(confidence):.2f}" if confidence is not None else "0.00"
    return f"AI: {direction} {conf_text} confidence — {reason}"


def _ai_section(sig: dict) -> list[str]:
    if not sig.get("ai_enabled"):
        return []
    if _ai_explain_enabled():
        return _ai_block(sig)
    line = _ai_line(sig)
    return [line] if line else []


def _direction_header(sig: dict) -> str:
    if sig["direction"] == "BUY":
        return "🟢 Сильный сигнал — BUY"
    return "🔴 Сильный сигнал — SELL"


def _analysis_block(sig: dict) -> str:
    lines = ["📐 Анализ:"]
    reasons = sig.get("reasons") or {}
    for source in ("swing", "ict", "smc", "macro", "classic"):
        for r in reasons.get(source, []):
            lines.append(f"• {r} ✅")
    for r in reasons.get("penalties", []):
        lines.append(f"• {r} ⚠️")
    return "\n".join(lines) if len(lines) > 1 else ""


def _tp3_line(sig: dict) -> str | None:
    tp3 = sig.get("tp3")
    if tp3 is None:
        return None
    diff = tp3 - sig["entry"]
    return f" •  TP3: `{_fmt_price(tp3)}` ({abs(diff):.1f} pts) — D1"


def format_strong_signal(sig: dict) -> str:
    flag = KZ_FLAGS.get(sig.get("killzone") or "", "")
    sl_diff = sig["sl"] - sig["entry"]
    tp1_diff = (sig["tp1"] - sig["entry"]) if sig["tp1"] is not None else 0
    ts: datetime = sig["ts_utc"]

    parts = [
        _direction_header(sig),
        "━━━━━━━━━━━━━━━━━━━",
        f"🔹 Вход: `{_fmt_price(sig['entry'])}`",
        f"🔺 Stop Loss: `{_fmt_price(sig['sl'])}` ({_fmt_pts(sl_diff)})",
        "🎯 Цели:",
        f" •  TP1: `{_fmt_price(sig['tp1'])}` ({abs(tp1_diff):.1f} pts) — FVG",
        _tp2_line(sig),
    ]
    tp3_line = _tp3_line(sig)
    if tp3_line:
        parts.append(tp3_line)
    parts.append("━━━━━━━━━━━━━━━━━━━")
    parts.extend([
        f"📊 R:R → 1:{sig['rr']:.1f}",
        f"🧠 Score: {sig['score']}/100",
        f"⏱ Сессия: {sig.get('killzone') or '—'} {flag} | M15→H1",
    ])
    if sig.get("strategy_label"):
        parts.append(f"📐 Стратегия: {sig['strategy_label']}")
    if sig.get("horizon_label"):
        parts.append(f"⏳ Горизонт: {sig['horizon_label']}")
    parts.extend(_ai_section(sig))
    analysis = _analysis_block(sig)
    if analysis:
        parts.extend(["━━━━━━━━━━━━━━━━━━━", analysis])
    parts.extend([
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
    parts.extend(_ai_section(sig))
    return "\n".join(parts)


def format_no_signal_killzone(killzone: str, price: float,
                              rsi: float | None) -> str:
    rsi_text = f"{rsi:.0f}" if rsi is not None else "—"
    return (
        f"⏳ {killzone} | Нет сигнала\n"
        f"💰 XAU: `{_fmt_price(price)}` | RSI H1: {rsi_text}"
    )


_STATUS_EMOJI = {
    "ACTIVE": "🟡",
    "TP1_HIT": "🎯",
    "TP2_HIT": "🎯",
    "TP3_HIT": "🏁",
    "SL_HIT": "🛑",
    "TIMEOUT": "⌛",
    "CANCELLED": "✖️",
}


def _fmt_R(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}R"


def format_active_signals(rows: list[dict]) -> str:
    if not rows:
        return "📭 Активных сигналов нет."
    parts = ["📒 Активные сигналы:"]
    for r in rows:
        emoji = _STATUS_EMOJI.get(r.get("status", "ACTIVE"), "•")
        parts.append(
            f"{emoji} #{r['id']} [{r.get('stream', 'intraday')}] "
            f"{r['direction']} @ {_fmt_price(r['entry'])} → "
            f"SL {_fmt_price(r['sl'])} / TP1 {_fmt_price(r.get('tp1'))} "
            f"| {r.get('status', 'ACTIVE')} "
            f"(MFE {r.get('max_favorable_R', 0.0):.2f}R / "
            f"MAE {r.get('max_adverse_R', 0.0):.2f}R)"
        )
    return "\n".join(parts)


def format_history(rows: list[dict]) -> str:
    if not rows:
        return "📭 История пуста."
    parts = ["🗂 Последние закрытые сигналы:"]
    for r in rows:
        emoji = _STATUS_EMOJI.get(r.get("status", ""), "•")
        parts.append(
            f"{emoji} #{r['id']} [{r.get('stream', 'intraday')}] "
            f"{r['direction']} @ {_fmt_price(r['entry'])} → "
            f"{r.get('status', '—')} {_fmt_R(r.get('final_R'))}"
        )
    return "\n".join(parts)


def format_lifecycle_transition(
    *,
    signal_id: int,
    direction: str,
    old_status: str,
    new_status: str,
    closed: bool,
    final_R: float | None,
    entry: float,
    price: float,
) -> str:
    emoji = _STATUS_EMOJI.get(new_status, "🔔")
    tail = f" → закрыт {_fmt_R(final_R)}" if closed else ""
    return (
        f"{emoji} #{signal_id} {direction} {old_status} → {new_status}{tail}\n"
        f"Вход {_fmt_price(entry)} | Цена {_fmt_price(price)}"
    )


def format_stats(metrics_today: dict, metrics_week: dict,
                 active_count: int, by_risk: dict[str, dict]) -> str:
    def _block(title: str, m: dict) -> str:
        pf = m.get("pf", 0.0)
        pf_text = "∞" if pf == float("inf") else f"{pf:.2f}"
        return (
            f"{title}: {m['wins']}W / {m['losses']}L | "
            f"WR {m['wr']*100:.0f}% | PF {pf_text} | "
            f"Exp {m['expectancy']:+.2f}R"
        )

    parts = [
        "📊 Lifecycle stats",
        _block("Сегодня", metrics_today),
        _block("7 дней", metrics_week),
        f"Активных: {active_count}",
    ]
    if by_risk:
        parts.append("По AI-риску:")
        for lbl, data in sorted(by_risk.items()):
            parts.append(
                f" • {lbl}: {data['wins']}W / {data['losses']}L "
                f"(avg {data['avg_R']:+.2f}R)"
            )
    return "\n".join(parts)


_MIN_CLOSED_FOR_METRICS = 10


def _pf_text(pf: float) -> str:
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"


def _format_bucket_lines(title: str,
                          buckets: dict[str, dict]) -> list[str]:
    if not buckets:
        return []
    lines = [title]
    for key, b in sorted(buckets.items(),
                          key=lambda kv: -kv[1].get("total", 0)):
        decided = b.get("wins", 0) + b.get("losses", 0)
        wr = (b["wins"] / decided) if decided else 0.0
        lines.append(
            f" • {key}: {b['total']} (W{b['wins']}/L{b['losses']}/"
            f"T{b['timeouts']}) WR {wr*100:.0f}% "
            f"ΣR {b.get('sum_R', 0.0):+.2f}"
        )
    return lines


def format_paper_report(rep: dict) -> str:
    period = rep.get("period_days", 1)
    title = "📅 Daily report" if period == 1 else f"📅 {period}-day report"
    parts = [
        title,
        f"Всего: {rep['total']} (актив {rep['active']}, закрыто {rep['closed']})",
    ]
    if rep["total"] == 0:
        parts.append("Нет данных за период.")
        return "\n".join(parts)

    parts.append(
        f"W {rep['wins']} / L {rep['losses']} / T {rep['timeouts']}"
    )
    if rep["closed"] >= _MIN_CLOSED_FOR_METRICS:
        parts.append(
            f"WR {rep['wr']*100:.0f}% | PF {_pf_text(rep['pf'])} | "
            f"Exp {rep['expectancy']:+.2f}R | ΣR {rep['total_final_R']:+.2f}"
        )
    else:
        parts.append(
            f"ΣR {rep['total_final_R']:+.2f} "
            f"(мало данных: {rep['closed']} закрытых, метрики "
            f"WR/PF/Exp недостаточно надёжны)"
        )
    if rep.get("max_adverse_R") is not None:
        parts.append(f"Max adverse: {rep['max_adverse_R']:.2f}R")

    parts.extend(_format_bucket_lines("По стримам:", rep.get("by_stream", {})))
    parts.extend(_format_bucket_lines("По AI-риску:", rep.get("by_risk", {})))
    parts.extend(_format_bucket_lines("По AI-action:", rep.get("by_action", {})))
    parts.extend(_format_bucket_lines("По tier:", rep.get("by_tier", {})))
    return "\n".join(parts)


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
