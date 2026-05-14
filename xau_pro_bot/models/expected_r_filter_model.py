"""Path E expected-R adapter (predicted_R >= threshold -> KEEP)."""

from __future__ import annotations

import enum
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ExpectedRDecision(str, enum.Enum):
    KEEP = "KEEP"
    BLOCK = "BLOCK"


class ExpectedRFilterModel:
    """Loads a Path E joblib bundle `{model, feature_cols}` and returns
    `{predicted_r, decision, threshold_used, error}`.

    On any load/predict failure returns a neutral KEEP with `error` populated.
    """

    def __init__(self, local_path: str, threshold: float = 0.05) -> None:
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
            pred = float(np.asarray(model.predict(X))[0])
            decision = (ExpectedRDecision.KEEP if pred >= self.threshold
                        else ExpectedRDecision.BLOCK)
            return {
                "predicted_r": pred,
                "decision": decision,
                "threshold_used": self.threshold,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            log.exception("ExpectedRFilterModel.predict failed")
            return {
                "predicted_r": None,
                "decision": ExpectedRDecision.KEEP,
                "threshold_used": self.threshold,
                "error": str(exc),
            }
