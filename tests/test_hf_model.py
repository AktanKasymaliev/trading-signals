from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from xau_pro_bot.models.hf_model import HFTradingModel


class ProbModel:
    classes_ = np.array(["BUY", "SELL", "NO_TRADE"])

    def predict_proba(self, features):
        return np.array([[0.72, 0.18, 0.10]])


class NumericProbModel:
    classes_ = np.array([1, -1, 0])

    def predict_proba(self, features):
        return np.array([[0.20, 0.70, 0.10]])


class PredictOnlyModel:
    def predict(self, features):
        return np.array(["SELL"])


def test_sklearn_predict_proba_output(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: ProbModel(),
    )
    model = HFTradingModel("owner/model", "sklearn", cache_dir="/tmp/cache")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.72
    assert pred["prob_buy"] == 0.72
    assert pred["prob_sell"] == 0.18
    assert pred["prob_no_trade"] == 0.10


def test_sklearn_numeric_classes(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: NumericProbModel(),
    )
    model = HFTradingModel("owner/model", "sklearn")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "SELL"
    assert pred["confidence"] == 0.70
    assert pred["prob_buy"] == 0.20
    assert pred["prob_sell"] == 0.70
    assert pred["prob_no_trade"] == 0.10


def test_predict_only_model_uses_default_confidence(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: PredictOnlyModel(),
    )
    model = HFTradingModel("owner/model", "sklearn")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "SELL"
    assert pred["confidence"] == 0.50
    assert pred["prob_buy"] is None
    assert pred["prob_sell"] is None
    assert pred["prob_no_trade"] is None


def test_safe_fallback_on_download_exception(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", boom)
    model = HFTradingModel("owner/model", "sklearn")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert pred["prob_buy"] is None
    assert pred["prob_sell"] is None
    assert pred["prob_no_trade"] is None
    assert "network unavailable" in pred["error"]


def test_no_model_loaded_at_import_or_init_time(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/model.joblib"

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: PredictOnlyModel(),
    )

    model = HFTradingModel("owner/model", "sklearn")

    assert calls == []
    model.predict(pd.DataFrame([{"x": 1.0}]))
    assert calls != []


def test_transformers_missing_dependency_message(monkeypatch):
    def missing_dependency(name):
        if name in {"transformers", "torch"}:
            raise ImportError(name)
        return importlib.import_module(name)

    model = HFTradingModel("owner/model", "transformers")
    monkeypatch.setattr(importlib, "import_module", missing_dependency)

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert "optional transformers dependency" in pred["error"]
