"""Telegram bot entrypoint with AsyncIOScheduler."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
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
from xau_pro_bot.signals.engine import MasterSignalEngine
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
ENGINE = MasterSignalEngine()


def _log_signal(sig: dict[str, Any], status: str) -> None:
    line = " | ".join(str(x) for x in (
        sig["ts_utc"].isoformat() if isinstance(sig["ts_utc"], datetime) else sig["ts_utc"],
        sig["direction"], sig["tier"], sig["score"],
        sig.get("entry"), sig.get("sl"), sig.get("tp1"), sig.get("tp2"),
        sig.get("rr"), sig.get("killzone"), status,
    ))
    _signal_log.info(line)


def _persist(sig: dict[str, Any]) -> None:
    assert STATE is not None
    STATE.record_signal({
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
    })


def _format(sig: dict[str, Any]) -> str:
    if sig["tier"] in ("STRONG", "NORMAL"):
        return formatter.format_strong_signal(sig)
    if sig["tier"] == "WEAK":
        return formatter.format_weak_signal(sig)
    raise ValueError(f"Cannot format tier {sig['tier']}")


async def _scan_and_send(app: Application, *, bypass_dedup: bool = False) -> None:
    assert STATE is not None
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
    except Exception:
        logging.exception("Data fetch failed")
        return

    try:
        sig = ENGINE.analyze(tfs)
    except Exception:
        logging.exception("Engine analyze failed")
        return

    ok, reason = should_send(sig, STATE, bypass_dedup=bypass_dedup)

    if not ok:
        _log_signal(sig, f"skipped:{reason.value if reason else 'unknown'}")
        if sig["tier"] == "NO_SIGNAL" and sig.get("killzone"):
            rsi = None
            try:
                enriched = classic.add_classic(tfs["H1"])
                val = enriched["RSI_14"].iloc[-1]
                rsi = float(val) if not pd.isna(val) else None
            except Exception:
                pass
            msg = formatter.format_no_signal_killzone(
                killzone=sig["killzone"], price=sig["entry"], rsi=rsi)
            try:
                await app.bot.send_message(
                    chat_id=ENV["TELEGRAM_CHAT_ID"], text=msg,
                    parse_mode=ParseMode.MARKDOWN)
            except Exception:
                logging.exception("Telegram no-signal send failed")
        return

    text = _format(sig)
    try:
        await app.bot.send_message(
            chat_id=ENV["TELEGRAM_CHAT_ID"], text=text,
            parse_mode=ParseMode.MARKDOWN)
        _persist(sig)
        _log_signal(sig, "sent")
    except Exception:
        logging.exception("Telegram send failed")
        _log_signal(sig, "send_failed")


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "XAU Pro Bot готов.\n"
        "Команды: /signal /status /levels /help /settings /stats")


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
    today = STATE.count_today()
    strong = STATE.count_today(tier="STRONG")
    weak = STATE.count_today(tier="WEAK")
    await update.message.reply_text(
        f"Сегодня: {today} сигналов (STRONG={strong}, WEAK={weak})")


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
    sched.add_job(lambda: STATE.prune_old(90) if STATE else None,
                  "cron", hour=0, minute=15, id="prune")
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
    app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)


if __name__ == "__main__":
    main()
