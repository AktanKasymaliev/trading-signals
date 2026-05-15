from __future__ import annotations

import pandas as pd

from xau_pro_bot.models.hf_model import HFTradingModel


def test_load_sklearn_uses_custom_filename(monkeypatch):
    downloaded: list[str] = []

    def fake_download(*, repo_id, filename, cache_dir, revision):
        downloaded.append(filename)
        return "/tmp/dummy.pkl"

    class DummyModel:
        classes_ = [0, 1]

        def predict_proba(self, features):
            return [[0.4, 0.6]]

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr("xau_pro_bot.models.hf_model.joblib.load", lambda path: DummyModel())

    sha = "a" * 40
    model = HFTradingModel(
        model_id="owner/m",
        model_type="sklearn",
        revision=sha,
        filename="trading_model_15m.pkl",
    )
    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert downloaded == ["trading_model_15m.pkl"]
    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.6


def test_load_sklearn_falls_back_to_defaults_when_filename_empty(monkeypatch):
    downloaded: list[str] = []

    def fake_download(*, repo_id, filename, cache_dir, revision):
        downloaded.append(filename)
        if filename == "model.joblib":
            return "/tmp/dummy.joblib"
        raise FileNotFoundError(filename)

    class DummyModel:
        classes_ = [1]

        def predict(self, features):
            return [1]

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr("xau_pro_bot.models.hf_model.joblib.load", lambda path: DummyModel())

    sha = "b" * 40
    model = HFTradingModel(
        model_id="owner/m",
        model_type="sklearn",
        revision=sha,
        filename="",
    )
    model.predict(pd.DataFrame([{"x": 1.0}]))

    assert downloaded[0] == "model.joblib"
