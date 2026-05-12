from __future__ import annotations

import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


def test_engine_passes_revision_from_config_to_adapter(monkeypatch):
    sha = "b" * 40
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("AI_MODEL_ID", "owner/model")
    monkeypatch.setenv("AI_MODEL_TYPE", "sklearn")
    monkeypatch.setenv("AI_MODEL_REVISION", sha)
    monkeypatch.setenv("AI_CACHE_DIR", "/tmp/cache")

    engine = MasterSignalEngine()

    assert engine.ai_model is not None
    assert engine.ai_model.model_id == "owner/model"
    assert engine.ai_model.model_type == "sklearn"
    assert engine.ai_model.revision == sha
    assert engine.ai_model.cache_dir == "/tmp/cache"


def test_engine_revision_empty_when_not_configured(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("AI_MODEL_ID", "owner/model")
    monkeypatch.delenv("AI_MODEL_REVISION", raising=False)

    engine = MasterSignalEngine()

    assert engine.ai_model is not None
    assert engine.ai_model.revision == ""
