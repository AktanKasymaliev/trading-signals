from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.models.hf_model import HFTradingModel


class LongShortModel:
    classes_ = np.array(["LONG", "SHORT", "FLAT"])

    def predict_proba(self, features):
        return np.array([[0.72, 0.18, 0.10]])


class UnknownLabelModel:
    classes_ = np.array(["CLASS_A", "CLASS_B"])

    def predict_proba(self, features):
        return np.array([[0.30, 0.70]])


def _force_loaded(model, instance):
    """Bypass _load() so tests don't need monkeypatched HF download."""
    model._model = instance


def test_long_short_flat_labels_map_to_buy_sell_no_trade():
    adapter = HFTradingModel("owner/model", "sklearn")
    _force_loaded(adapter, LongShortModel())

    pred = adapter.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.72
    assert pred["prob_buy"] == 0.72
    assert pred["prob_sell"] == 0.18
    assert pred["prob_no_trade"] == 0.10


def test_unknown_labels_fall_back_to_no_trade_with_error():
    adapter = HFTradingModel("owner/model", "sklearn")
    _force_loaded(adapter, UnknownLabelModel())

    pred = adapter.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert "error" in pred
    assert "unrecognized" in pred["error"].lower()
