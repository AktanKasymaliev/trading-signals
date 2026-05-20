"""Bot-level routing tests for the no-signal branch.

Covers:
- scheduler honors SEND_NO_SIGNAL_UPDATES=false and stays silent;
- scheduler still sends NO_SIGNAL within window when flag is true;
- manual /signal returns compact "no changes" reply on duplicate;
- /signal_debug bypasses no-signal dedup and always sends a full update.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot import bot as bot_module
from xau_pro_bot import config


class _FakeBot:
    def __init__(self) -> None:
        self.sends: list[dict] = []

    async def send_message(self, **kwargs):
        self.sends.append(kwargs)


def _make_tfs() -> dict[str, pd.DataFrame]:
    """Minimal H1/M15 frames for RSI/price calc paths."""
    idx_h1 = pd.date_range("2026-05-20", periods=240, freq="h", tz="UTC")
    closes = 3300.0 + np.linspace(0, 10, len(idx_h1))
    h1 = pd.DataFrame({
        "Open": closes, "High": closes + 0.5,
        "Low": closes - 0.5, "Close": closes,
    }, index=idx_h1)
    idx_m15 = pd.date_range("2026-05-20", periods=10, freq="15min", tz="UTC")
    m15 = pd.DataFrame({
        "Open": [3310.0] * 10, "High": [3311.0] * 10,
        "Low": [3309.0] * 10, "Close": [3310.5] * 10,
    }, index=idx_m15)
    return {"H1": h1, "M15": m15}


@pytest.fixture
def fake_app():
    return SimpleNamespace(bot=_FakeBot())


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    bot_module._RECENT_NO_SIGNAL.clear()
    monkeypatch.setitem(bot_module.ENV, "TELEGRAM_CHAT_ID", "test-chat")
    yield
    bot_module._RECENT_NO_SIGNAL.clear()


def test_scheduled_no_signal_suppressed_when_flag_off(monkeypatch, fake_app):
    monkeypatch.setattr(config, "SEND_NO_SIGNAL_UPDATES", False)
    monkeypatch.setattr(bot_module, "get_killzone", lambda: "NY AM KZ")
    tfs = _make_tfs()
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="scheduled", force_no_signal=False,
    ))
    assert fake_app.bot.sends == []


def test_scheduled_no_signal_sent_when_flag_on(monkeypatch, fake_app):
    monkeypatch.setattr(config, "SEND_NO_SIGNAL_UPDATES", True)
    monkeypatch.setattr(bot_module, "get_killzone", lambda: "NY AM KZ")
    tfs = _make_tfs()
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="scheduled", force_no_signal=False,
    ))
    assert len(fake_app.bot.sends) == 1
    assert "Нет сигнала" in fake_app.bot.sends[0]["text"]


def test_scheduled_dedup_suppresses_duplicate(monkeypatch, fake_app):
    monkeypatch.setattr(config, "SEND_NO_SIGNAL_UPDATES", True)
    monkeypatch.setattr(bot_module, "get_killzone", lambda: "NY AM KZ")
    tfs = _make_tfs()
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="scheduled", force_no_signal=False,
    ))
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="scheduled", force_no_signal=False,
    ))
    # Second call must be deduped — only one send total.
    assert len(fake_app.bot.sends) == 1


def test_manual_duplicate_returns_compact_reply(monkeypatch, fake_app):
    monkeypatch.setattr(config, "SEND_NO_SIGNAL_UPDATES", True)
    monkeypatch.setattr(bot_module, "get_killzone", lambda: "NY AM KZ")
    tfs = _make_tfs()
    # Prime cache via a manual call (full update).
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="manual", force_no_signal=False,
    ))
    # Second manual call must produce a compact "no changes" reply.
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="manual", force_no_signal=False,
    ))
    assert len(fake_app.bot.sends) == 2
    second = fake_app.bot.sends[1]["text"]
    assert "Анализ без изменений" in second
    assert "мин назад" in second


def test_manual_replies_even_when_flag_off(monkeypatch, fake_app):
    """SEND_NO_SIGNAL_UPDATES=false must NOT silence manual /signal."""
    monkeypatch.setattr(config, "SEND_NO_SIGNAL_UPDATES", False)
    monkeypatch.setattr(bot_module, "get_killzone", lambda: "NY AM KZ")
    tfs = _make_tfs()
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="manual", force_no_signal=False,
    ))
    assert len(fake_app.bot.sends) == 1
    assert "Нет сигнала" in fake_app.bot.sends[0]["text"]


def test_signal_debug_bypasses_dedup(monkeypatch, fake_app):
    monkeypatch.setattr(config, "SEND_NO_SIGNAL_UPDATES", True)
    monkeypatch.setattr(bot_module, "get_killzone", lambda: "NY AM KZ")
    tfs = _make_tfs()
    # Pretend a fresh send was just made.
    now = datetime.now(timezone.utc)
    bot_module._RECENT_NO_SIGNAL[("intraday", "NY AM KZ", int(round(3310.5 / 2.0)))] = now
    asyncio.run(bot_module._handle_no_signal(
        fake_app, tfs, source="manual", force_no_signal=True,
    ))
    assert len(fake_app.bot.sends) == 1
    assert "Нет сигнала" in fake_app.bot.sends[0]["text"]
