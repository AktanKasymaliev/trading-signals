"""Path D filter model adapter (GOOD/BAD → KEEP/BLOCK)."""

from __future__ import annotations

import enum
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FilterDecision(str, enum.Enum):
    KEEP = "KEEP"
    BLOCK = "BLOCK"


class TradeFilterModel:
    """Loads a Path D filter joblib bundle `{model, feature_cols}` and
    returns `{good_prob, bad_prob, decision, threshold_used}`.

    On load/predict failure returns a neutral KEEP with `error` populated.
    """

    def __init__(self, local_path: str, threshold: float = 0.55) -> None:
        self.local_path = local_path
        self.threshold = float(threshold)
        self._bundle: dict | None = None

    def _load(self) -> dict:
        if self._bundle is None:
            self._bundle = joblib.load(self.local_path)
        return self._bundle

    def _align(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in cols:
            out[c] = X[c] if c in X.columns else 0.0
        return out

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        try:
            bundle = self._load()
            model = bundle["model"]
            cols = bundle["feature_cols"]
            X = self._align(features, cols)
            probs = np.asarray(model.predict_proba(X))[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            good_idx = classes.index(1) if 1 in classes else 1
            bad_idx = classes.index(0) if 0 in classes else 0
            good = float(probs[good_idx])
            bad = float(probs[bad_idx])
            decision = (FilterDecision.KEEP if good >= self.threshold
                        else FilterDecision.BLOCK)
            return {
                "good_prob": good, "bad_prob": bad,
                "decision": decision, "threshold_used": self.threshold,
                "error": None,
            }
        except Exception as exc:
            log.exception("TradeFilterModel.predict failed")
            return {
                "good_prob": None, "bad_prob": None,
                "decision": FilterDecision.KEEP,
                "threshold_used": self.threshold, "error": str(exc),
            }
