import importlib

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


def test_ai_defaults_disabled(monkeypatch):
    monkeypatch.delenv("AI_ENABLED", raising=False)
    monkeypatch.delenv("AI_MODEL_ID", raising=False)
    monkeypatch.delenv("AI_MODEL_TYPE", raising=False)
    monkeypatch.delenv("AI_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("AI_STRONG_CONFIDENCE", raising=False)
    monkeypatch.delenv("AI_NO_TRADE_THRESHOLD", raising=False)
    monkeypatch.delenv("AI_SCORE_BONUS", raising=False)
    monkeypatch.delenv("AI_STRONG_SCORE_BONUS", raising=False)
    monkeypatch.delenv("AI_CONFLICT_PENALTY", raising=False)
    monkeypatch.delenv("AI_CACHE_DIR", raising=False)

    cfg = config.load_ai_config()

    assert cfg["enabled"] is False
    assert cfg["model_id"] == ""
    assert cfg["model_type"] == "sklearn"
    assert cfg["min_confidence"] == 0.65
    assert cfg["strong_confidence"] == 0.75
    assert cfg["no_trade_threshold"] == 0.60
    assert cfg["score_bonus"] == 8
    assert cfg["strong_score_bonus"] == 12
    assert cfg["conflict_penalty"] == 10
    assert cfg["cache_dir"] == "./models_cache"


def test_ai_env_overrides(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("AI_MODEL_ID", "owner/xau-model")
    monkeypatch.setenv("AI_MODEL_TYPE", "transformers")
    monkeypatch.setenv("AI_MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("AI_STRONG_CONFIDENCE", "0.82")
    monkeypatch.setenv("AI_NO_TRADE_THRESHOLD", "0.64")
    monkeypatch.setenv("AI_SCORE_BONUS", "9")
    monkeypatch.setenv("AI_STRONG_SCORE_BONUS", "14")
    monkeypatch.setenv("AI_CONFLICT_PENALTY", "11")
    monkeypatch.setenv("AI_CACHE_DIR", "/tmp/hf-cache")

    cfg = config.load_ai_config()

    assert cfg["enabled"] is True
    assert cfg["model_id"] == "owner/xau-model"
    assert cfg["model_type"] == "transformers"
    assert cfg["min_confidence"] == 0.7
    assert cfg["strong_confidence"] == 0.82
    assert cfg["no_trade_threshold"] == 0.64
    assert cfg["score_bonus"] == 9
    assert cfg["strong_score_bonus"] == 14
    assert cfg["conflict_penalty"] == 11
    assert cfg["cache_dir"] == "/tmp/hf-cache"


def test_bool_parser_accepts_common_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("AI_ENABLED", value)
        assert config.load_ai_config()["enabled"] is True

    for value in ("0", "false", "FALSE", "no", "off", ""):
        monkeypatch.setenv("AI_ENABLED", value)
        assert config.load_ai_config()["enabled"] is False


def test_import_ignores_malformed_disabled_ai_numeric_env(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "false")
    monkeypatch.setenv("AI_MIN_CONFIDENCE", "abc")

    reloaded = importlib.reload(config)

    assert reloaded.AI_ENABLED is False
    assert reloaded.AI_MIN_CONFIDENCE == 0.65


def test_load_ai_config_reports_malformed_numeric_env(monkeypatch):
    monkeypatch.setenv("AI_MIN_CONFIDENCE", "abc")

    with pytest.raises(RuntimeError, match="AI_MIN_CONFIDENCE"):
        config.load_ai_config()


def test_ai_config_includes_revision_default_empty(monkeypatch):
    monkeypatch.delenv("AI_MODEL_REVISION", raising=False)
    cfg = config.load_ai_config()
    assert cfg["revision"] == ""


def test_ai_config_revision_from_env(monkeypatch):
    sha = "a" * 40
    monkeypatch.setenv("AI_MODEL_REVISION", sha)
    cfg = config.load_ai_config()
    assert cfg["revision"] == sha


def test_ai_config_includes_model_filename_default_empty(monkeypatch):
    monkeypatch.delenv("AI_MODEL_FILENAME", raising=False)
    cfg = config.load_ai_config()
    assert cfg["model_filename"] == ""


def test_ai_config_model_filename_from_env(monkeypatch):
    monkeypatch.setenv("AI_MODEL_FILENAME", "trading_model_15m.pkl")
    cfg = config.load_ai_config()
    assert cfg["model_filename"] == "trading_model_15m.pkl"


def test_ai_config_feature_set_default_internal(monkeypatch):
    monkeypatch.delenv("AI_FEATURE_SET", raising=False)
    cfg = config.load_ai_config()
    assert cfg["feature_set"] == "internal"


def test_ai_config_feature_set_smc_v2(monkeypatch):
    monkeypatch.setenv("AI_FEATURE_SET", "smc_v2")
    cfg = config.load_ai_config()
    assert cfg["feature_set"] == "smc_v2"
