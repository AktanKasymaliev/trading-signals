"""Telegram bot entrypoint with AsyncIOScheduler."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import subprocess
import sys
from datetime import datetime
from typing import Any

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes,
)

from xau_pro_bot import config, data, formatter
from xau_pro_bot.indicators import classic
from xau_pro_bot.indicators.ict import get_killzone
from xau_pro_bot.lifecycle import (
    Candle, TTL_BY_STREAM, evaluate_candle, lifecycle_signal_from_row,
)
from xau_pro_bot.signals.router import StreamRouter
from xau_pro_bot.signals.filters import should_send
from xau_pro_bot.state import State


def _setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    err = logging.handlers.RotatingFileHandler(
        "errors.log", maxBytes=2_000_000, backupCount=2)
    err.setLevel(logging.ERROR)
    err.setFormatter(logging.Formatter(fmt))
    root.addHandler(err)


_signal_log = logging.getLogger("signals")
_signal_handler = logging.handlers.RotatingFileHandler(
    "signals.log", maxBytes=2_000_000, backupCount=3)
_signal_handler.setFormatter(logging.Formatter("%(message)s"))
_signal_log.addHandler(_signal_handler)
_signal_log.setLevel(logging.INFO)


ENV: dict[str, str] = {}
STATE: State | None = None
ROUTER = StreamRouter()
# Module-level fingerprint cache of recently sent signals.
# Shared across overlapping scan jobs to suppress duplicate sends emitted
# in the gap between Telegram send and DB persist.
_RECENT_SENT: dict[tuple, datetime] = {}


def _git_commit_sha() -> str:
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _log_startup_banner() -> None:
    ai_cfg = config.load_ai_config()
    logging.info(
        "Boot: commit=%s ai_enabled=%s ai_explain=%s feature_set=%s "
        "model_id=%s model_filename=%s hybrid_mode=%s",
        _git_commit_sha(),
        ai_cfg["enabled"], ai_cfg["explain"], ai_cfg["feature_set"],
        ai_cfg["model_id"] or "—",
        ai_cfg["model_filename"] or "—",
        ai_cfg["hybrid_mode"],
    )


def _log_signal(sig: dict[str, Any], status: str) -> None:
    line = " | ".join(str(x) for x in (
        sig["ts_utc"].isoformat() if isinstance(sig["ts_utc"], datetime) else sig["ts_utc"],
        sig["direction"], sig["tier"], sig["score"],
        sig.get("entry"), sig.get("sl"), sig.get("tp1"), sig.get("tp2"),
        sig.get("rr"), sig.get("killzone"), status,
    ))
    _signal_log.info(line)


def _persist(sig: dict[str, Any]) -> int:
    assert STATE is not None
    return STATE.record_signal({
        "ts_utc": sig["ts_utc"].isoformat(),
        "direction": sig["direction"],
        "tier": sig["tier"],
        "score": sig["score"],
        "entry": sig["entry"],
        "sl": sig.get("sl") or 0.0,
        "tp1": sig.get("tp1"),
        "tp2": sig.get("tp2"),
        "tp3": sig.get("tp3"),
        "rr": sig.get("rr"),
        "killzone": sig.get("killzone"),
        "reasons_json": json.dumps(sig["reasons"], ensure_ascii=False),
        "stream": sig.get("stream", "intraday"),
        "ai_action": sig.get("ai_action"),
        "ai_risk_label": sig.get("ai_risk_label"),
        "ai_model_name": sig.get("ai_model_name"),
    })


def _format(sig: dict[str, Any]) -> str:
    if sig["tier"] in ("STRONG", "NORMAL"):
        return formatter.format_strong_signal(sig)
    if sig["tier"] == "WEAK":
        return formatter.format_weak_signal(sig)
    raise ValueError(f"Cannot format tier {sig['tier']}")


def _scan_fingerprint(sig: dict[str, Any]) -> tuple:
    return (
        sig.get("stream", "intraday"),
        sig.get("direction"),
        round(float(sig.get("entry") or 0.0), 2),
    )


def _prune_recent_sent() -> None:
    cutoff = datetime.utcnow().replace(tzinfo=None)
    from datetime import timedelta as _td
    horizon = cutoff - _td(hours=config.DEDUP_HOURS)
    stale = [k for k, ts in _RECENT_SENT.items() if ts < horizon]
    for k in stale:
        _RECENT_SENT.pop(k, None)


async def _scan_and_send(app: Application, *, bypass_dedup: bool = False) -> None:
    assert STATE is not None
    _prune_recent_sent()
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
    except Exception:
        logging.exception("Data fetch failed")
        return

    try:
        results = ROUTER.analyze(tfs)
    except Exception:
        logging.exception("Router failed")
        return

    if not results:
        kz = get_killzone()
        if kz:
            rsi = None
            try:
                enriched = classic.add_classic(tfs["H1"])
                val = enriched["RSI_14"].iloc[-1]
                rsi = float(val) if not pd.isna(val) else None
            except Exception:
                logging.exception("No-signal RSI calculation failed")
            price = float(tfs["M15"]["Close"].iloc[-1])
            msg = formatter.format_no_signal_killzone(
                killzone=kz, price=price, rsi=rsi)
            try:
                await app.bot.send_message(
                    chat_id=ENV["TELEGRAM_CHAT_ID"], text=msg,
                    parse_mode=ParseMode.MARKDOWN)
            except Exception:
                logging.exception("Telegram no-signal send failed")
        return

    for sig in results:
        ok, reason = should_send(
            sig, STATE, bypass_dedup=bypass_dedup,
            scan_fingerprints=set(_RECENT_SENT.keys()),
        )
        if not ok:
            _log_signal(sig, f"skipped:{reason.value if reason else 'unknown'}")
            continue

        text = _format(sig)
        try:
            await app.bot.send_message(
                chat_id=ENV["TELEGRAM_CHAT_ID"], text=text,
                parse_mode=ParseMode.MARKDOWN)
            _persist(sig)
            _RECENT_SENT[_scan_fingerprint(sig)] = datetime.utcnow().replace(tzinfo=None)
            _log_signal(sig, "sent")
        except Exception:
            logging.exception("Telegram send failed")
            _log_signal(sig, "send_failed")


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "XAU Pro Bot готов.\n"
        "Команды: /signal /status /levels /help /settings "
        "/stats /active /history /daily_report /weekly_report")


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Анализирую…")
    await _scan_and_send(ctx.application, bypass_dedup=True)


async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
        from xau_pro_bot.indicators.smc import detect_structure
        from xau_pro_bot.indicators.wyckoff import detect_wyckoff

        d1 = classic.add_classic(tfs["D1"])
        h4 = tfs["H4"]
        ema50 = d1["EMA_50"].iloc[-1]
        ema200 = d1["EMA_200"].iloc[-1]
        d1_trend = "bull" if (not pd.isna(ema50) and not pd.isna(ema200)
                              and ema50 > ema200) else "bear"
        struct = detect_structure(h4, swing_len=5)
        wy = detect_wyckoff(tfs["D1"])
        snapshot = {
            "price": float(tfs["M15"]["Close"].iloc[-1]),
            "killzone": get_killzone(),
            "d1_trend": d1_trend,
            "h4_structure": struct["last_event"] or "—",
            "wyckoff": f"{wy['phase']} ({wy['bias']})",
        }
        await update.message.reply_text(
            formatter.format_status(snapshot), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        logging.exception("/status failed")
        await update.message.reply_text("Ошибка получения данных.")


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Тиры:\n"
        "STRONG (≥65) — всегда\n"
        "NORMAL (50–64) — по фильтрам\n"
        "WEAK (40–49) — только в killzone\n\n"
        "Лимиты: 6 сигналов/сутки, WEAK 1 раз в 4ч.\n"
        "Score — внутренняя метрика confluence, не вероятность."
    )


async def cmd_settings(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"STRONG≥{config.STRONG_SIGNAL}, NORMAL≥{config.NORMAL_SIGNAL}, "
        f"WEAK≥{config.WEAK_SIGNAL}\n"
        f"Dedup {config.DEDUP_HOURS}h, RR≥{config.MIN_RR}\n"
        f"Scan: KZ {config.KILLZONE_SCAN_INTERVAL}s / out "
        f"{config.BACKGROUND_SCAN_INTERVAL}s")


async def cmd_levels(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
        from xau_pro_bot.indicators.ict import (
            find_fvg, find_order_blocks, find_liquidity,
        )
        h1 = tfs["H1"]
        fvgs = find_fvg(h1, 3)
        obs = find_order_blocks(h1, 50)[:3]
        liq = find_liquidity(h1, lookback=30)
        lines = ["📍 Ключевые уровни (H1)"]
        for f in fvgs:
            lines.append(f"FVG {f['type']}: {f['bottom']:.2f}–{f['top']:.2f}")
        for ob in obs:
            lines.append(f"OB {ob['type']}: {ob['low']:.2f}–{ob['high']:.2f}")
        if liq["buy_side"]:
            lines.append(f"Buy-side liq: {liq['buy_side'][:3]}")
        if liq["sell_side"]:
            lines.append(f"Sell-side liq: {liq['sell_side'][:3]}")
        await update.message.reply_text("\n".join(lines))
    except Exception:
        logging.exception("/levels failed")
        await update.message.reply_text("Ошибка.")


async def cmd_stats(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    assert STATE is not None
    today = STATE.lifecycle_metrics(days=1)
    week = STATE.lifecycle_metrics(days=7)
    active = len(STATE.get_active())
    by_risk = STATE.lifecycle_stats_by_risk()
    await update.message.reply_text(
        formatter.format_stats(today, week, active, by_risk))


async def cmd_daily_report(update: Update,
                             _: ContextTypes.DEFAULT_TYPE) -> None:
    assert STATE is not None
    rep = STATE.paper_report(days=1)
    await update.message.reply_text(formatter.format_paper_report(rep))


async def cmd_weekly_report(update: Update,
                              _: ContextTypes.DEFAULT_TYPE) -> None:
    assert STATE is not None
    rep = STATE.paper_report(days=7)
    await update.message.reply_text(formatter.format_paper_report(rep))


async def _scheduled_daily_report(app: Application) -> None:
    assert STATE is not None
    try:
        rep = STATE.paper_report(days=1)
        text = formatter.format_paper_report(rep)
        await app.bot.send_message(
            chat_id=ENV["TELEGRAM_CHAT_ID"], text=text,
            parse_mode=ParseMode.MARKDOWN)
    except Exception:
        logging.exception("Scheduled daily report failed")


def _daily_report_enabled() -> bool:
    return os.getenv("DAILY_REPORT_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


async def cmd_active(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    assert STATE is not None
    rows = STATE.get_active()
    await update.message.reply_text(formatter.format_active_signals(rows))


async def cmd_history(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    assert STATE is not None
    rows = STATE.recent_closed(limit=10)
    await update.message.reply_text(formatter.format_history(rows))


def _candle_from_m15(m15_df: pd.DataFrame) -> Candle:
    last = m15_df.iloc[-1]
    ts = m15_df.index[-1]
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    return Candle(
        high=float(last["High"]),
        low=float(last["Low"]),
        close=float(last["Close"]),
        ts=ts,
    )


async def _monitor_active(app: Application) -> None:
    assert STATE is not None
    rows = STATE.get_active()
    if not rows:
        return
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
    except Exception:
        logging.exception("Monitor data fetch failed")
        return
    candle = _candle_from_m15(tfs["M15"])
    now = datetime.now(tz=candle.ts.tzinfo) if candle.ts.tzinfo else datetime.utcnow()
    for row in rows:
        try:
            sig = lifecycle_signal_from_row(row)
            ttl = TTL_BY_STREAM.get(sig.stream, 24.0)
            tr = evaluate_candle(sig, candle, now=now, ttl_hours=ttl)
        except Exception:
            logging.exception("Lifecycle eval failed for signal %s", row.get("id"))
            continue
        if tr is None:
            continue
        try:
            STATE.update_lifecycle(
                tr.signal_id,
                status=tr.new_status,
                closed=tr.closed,
                final_R=tr.final_R,
                max_favorable_R=tr.max_favorable_R,
                max_adverse_R=tr.max_adverse_R,
                closed_at=tr.closed_at,
            )
        except Exception:
            logging.exception("Lifecycle update failed for signal %s", tr.signal_id)
            continue
        if tr.new_status != tr.old_status:
            text = formatter.format_lifecycle_transition(
                signal_id=tr.signal_id,
                direction=sig.direction,
                old_status=tr.old_status,
                new_status=tr.new_status,
                closed=tr.closed,
                final_R=tr.final_R,
                entry=sig.entry,
                price=tr.price,
            )
            try:
                await app.bot.send_message(
                    chat_id=ENV["TELEGRAM_CHAT_ID"], text=text,
                    parse_mode=ParseMode.MARKDOWN)
            except Exception:
                logging.exception("Lifecycle telegram send failed")


async def _scheduled_scan(app: Application) -> None:
    await _scan_and_send(app, bypass_dedup=False)


def _build_scheduler(app: Application) -> AsyncIOScheduler:
    sched = AsyncIOScheduler(timezone="UTC")
    sched.add_job(_scheduled_scan, "interval",
                  seconds=config.KILLZONE_SCAN_INTERVAL,
                  args=[app], id="kz_scan",
                  misfire_grace_time=60, coalesce=True)
    sched.add_job(_scheduled_scan, "interval",
                  seconds=config.BACKGROUND_SCAN_INTERVAL,
                  args=[app], id="bg_scan",
                  misfire_grace_time=60, coalesce=True)
    sched.add_job(_monitor_active, "interval",
                  seconds=config.KILLZONE_SCAN_INTERVAL,
                  args=[app], id="lifecycle_monitor",
                  misfire_grace_time=60, coalesce=True)
    sched.add_job(lambda: STATE.prune_old(90) if STATE else None,
                  "cron", hour=0, minute=15, id="prune")
    if _daily_report_enabled():
        sched.add_job(_scheduled_daily_report, "cron",
                      hour=23, minute=55, args=[app], id="daily_report",
                      misfire_grace_time=300, coalesce=True)
        logging.info("Daily report scheduler enabled (23:55 UTC).")
    return sched


def main() -> None:
    _setup_logging()
    global ENV, STATE
    ENV = config.load_env(required=[
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TWELVE_DATA_API_KEY"])
    STATE = State(db_path=os.getenv("STATE_DB_PATH", "./state.db"))

    app = ApplicationBuilder().token(ENV["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("levels", cmd_levels))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("active", cmd_active))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("daily_report", cmd_daily_report))
    app.add_handler(CommandHandler("weekly_report", cmd_weekly_report))

    sched = _build_scheduler(app)

    async def on_startup(_: Application) -> None:
        sched.start()
        logging.info("Scheduler started.")

    async def on_shutdown(_: Application) -> None:
        sched.shutdown(wait=False)
        if STATE is not None:
            STATE.close()

    app.post_init = on_startup
    app.post_shutdown = on_shutdown

    logging.info("Starting XAU Pro Bot…")
    _log_startup_banner()
    app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)


if __name__ == "__main__":
    main()
