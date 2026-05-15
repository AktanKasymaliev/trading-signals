from __future__ import annotations
import joblib
import pandas as pd
import pytest

from xau_pro_bot.models.hf_model import HFTradingModel


class StubModel:
    classes_ = [-1, 0, 1]
    def predict_proba(self, X):
        return [[0.1, 0.2, 0.7]]


def test_local_path_skips_hf_download(tmp_path, monkeypatch):
    p = tmp_path / "model.joblib"
    joblib.dump(StubModel(), p)
    called = []
    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download",
                        lambda **kw: called.append("nope") or "/x")
    m = HFTradingModel(model_id="", model_type="sklearn", local_path=str(p))
    pred = m.predict(pd.DataFrame([{"x": 1.0}]))
    assert called == []
    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.7


def test_local_path_overrides_revision_requirement(tmp_path):
    p = tmp_path / "model.joblib"
    joblib.dump(StubModel(), p)
    m = HFTradingModel(model_id="", model_type="sklearn",
                       local_path=str(p), revision="")
    pred = m.predict(pd.DataFrame([{"x": 1.0}]))
    assert pred["direction"] == "BUY"


def test_config_includes_local_path(monkeypatch):
    from xau_pro_bot import config
    monkeypatch.setenv("AI_MODEL_LOCAL_PATH", "/tmp/m.joblib")
    cfg = config.load_ai_config()
    assert cfg["local_path"] == "/tmp/m.joblib"
