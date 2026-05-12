from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd


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


def _hf_model_module():
    return importlib.import_module("xau_pro_bot.models.hf_model")


def test_sklearn_predict_proba_output(monkeypatch):
    hf_model = _hf_model_module()
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: ProbModel(),
    )
    model = hf_model.HFTradingModel(
        "owner/model",
        "sklearn",
        cache_dir="/tmp/cache",
        revision="abc123",
    )

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.72
    assert pred["prob_buy"] == 0.72
    assert pred["prob_sell"] == 0.18
    assert pred["prob_no_trade"] == 0.10


def test_sklearn_numeric_classes(monkeypatch):
    hf_model = _hf_model_module()
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: NumericProbModel(),
    )
    model = hf_model.HFTradingModel("owner/model", "sklearn", revision="abc123")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "SELL"
    assert pred["confidence"] == 0.70
    assert pred["prob_buy"] == 0.20
    assert pred["prob_sell"] == 0.70
    assert pred["prob_no_trade"] == 0.10


def test_predict_only_model_uses_default_confidence(monkeypatch):
    hf_model = _hf_model_module()
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: PredictOnlyModel(),
    )
    model = hf_model.HFTradingModel("owner/model", "sklearn", revision="abc123")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "SELL"
    assert pred["confidence"] == 0.50
    assert pred["prob_buy"] is None
    assert pred["prob_sell"] is None
    assert pred["prob_no_trade"] is None


def test_safe_fallback_on_download_exception(monkeypatch):
    hf_model = _hf_model_module()

    def boom(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", boom)
    model = hf_model.HFTradingModel("owner/model", "sklearn", revision="abc123")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert pred["prob_buy"] is None
    assert pred["prob_sell"] is None
    assert pred["prob_no_trade"] is None
    assert "network unavailable" in pred["error"]


def test_import_does_not_download_or_load_model(monkeypatch):
    download_calls = []
    load_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return "/tmp/model.joblib"

    def fake_load(path):
        load_calls.append(path)
        return PredictOnlyModel()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("joblib.load", fake_load)
    sys.modules.pop("xau_pro_bot.models.hf_model", None)

    importlib.import_module("xau_pro_bot.models.hf_model")

    assert download_calls == []
    assert load_calls == []


def test_no_model_loaded_before_predict(monkeypatch):
    hf_model = _hf_model_module()
    download_calls = []
    load_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return "/tmp/model.joblib"

    def fake_load(path):
        load_calls.append(path)
        return PredictOnlyModel()

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr("xau_pro_bot.models.hf_model.joblib.load", fake_load)

    model = hf_model.HFTradingModel("owner/model", "sklearn", revision="abc123")

    assert download_calls == []
    assert load_calls == []
    model.predict(pd.DataFrame([{"x": 1.0}]))
    assert download_calls != []
    assert load_calls != []


def test_sklearn_artifact_filenames_are_tried_in_order(monkeypatch):
    hf_model = _hf_model_module()
    filenames = []

    def fake_download(**kwargs):
        filename = kwargs["filename"]
        filenames.append(filename)
        if filename != "trading_model.joblib":
            raise RuntimeError(f"missing {filename}")
        return "/tmp/trading_model.joblib"

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: PredictOnlyModel(),
    )
    model = hf_model.HFTradingModel("owner/model", "sklearn", revision="abc123")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "SELL"
    assert filenames == [
        "model.joblib",
        "model.pkl",
        "classifier.joblib",
        "trading_model.joblib",
    ]


def test_sklearn_without_revision_does_not_download_or_load(monkeypatch):
    hf_model = _hf_model_module()
    download_calls = []
    load_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return "/tmp/model.joblib"

    def fake_load(path):
        load_calls.append(path)
        return ProbModel()

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr("xau_pro_bot.models.hf_model.joblib.load", fake_load)
    model = hf_model.HFTradingModel("owner/model", "sklearn")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert pred["prob_buy"] is None
    assert pred["prob_sell"] is None
    assert pred["prob_no_trade"] is None
    assert "pinned revision" in pred["error"]
    assert download_calls == []
    assert load_calls == []


def test_sklearn_with_revision_passes_revision_to_download(monkeypatch):
    hf_model = _hf_model_module()
    download_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return "/tmp/model.joblib"

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: ProbModel(),
    )
    model = hf_model.HFTradingModel("owner/model", "sklearn", revision="abc123")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.72
    assert download_calls[0]["revision"] == "abc123"


def test_custom_mode_returns_neutral_with_error():
    hf_model = _hf_model_module()
    model = hf_model.HFTradingModel("owner/model", "custom")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert pred["prob_buy"] is None
    assert pred["prob_sell"] is None
    assert pred["prob_no_trade"] is None
    assert "custom" in pred["error"] or "injected adapter" in pred["error"]


def test_transformers_missing_dependency_message(monkeypatch):
    hf_model = _hf_model_module()

    def missing_dependency(name):
        if name in {"transformers", "torch"}:
            raise ImportError(name)
        return importlib.import_module(name)

    model = hf_model.HFTradingModel("owner/model", "transformers")
    monkeypatch.setattr(importlib, "import_module", missing_dependency)

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert "optional transformers dependency" in pred["error"]
