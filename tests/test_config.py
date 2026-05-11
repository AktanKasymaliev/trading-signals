import pytest

from xau_pro_bot import config


def test_thresholds_match_spec():
    assert config.STRONG_SIGNAL == 65
    assert config.NORMAL_SIGNAL == 50
    assert config.WEAK_SIGNAL == 40
    assert config.MIN_RR == 1.8


def test_rate_limits():
    assert config.MAX_SIGNALS_PER_DAY == 6
    assert config.WEAK_COOLDOWN_HOURS == 4


def test_scan_intervals():
    assert config.KILLZONE_SCAN_INTERVAL == 300
    assert config.BACKGROUND_SCAN_INTERVAL == 900


def test_load_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        config.load_env(required=["TELEGRAM_BOT_TOKEN"])


def test_load_env_returns_dict(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "key")
    env = config.load_env(required=["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TWELVE_DATA_API_KEY"])
    assert env["TELEGRAM_BOT_TOKEN"] == "abc"
    assert env["TELEGRAM_CHAT_ID"] == "123"
