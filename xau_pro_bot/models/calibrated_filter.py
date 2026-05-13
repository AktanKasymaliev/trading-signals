"""Probability calibration for the Path D filter.

Wraps a base LightGBM classifier in CalibratedClassifierCV(method='isotonic', cv=3).
Exposes classes_ and predict_proba so existing TradeFilterModel.predict
continues to work without changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class CalibratedFilterWrapper:
    method: str = "isotonic"
    cv: int = 3
    base_params: dict | None = None
    estimator_: CalibratedClassifierCV | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CalibratedFilterWrapper":
        import lightgbm as lgb
        params = self.base_params or dict(
            objective="binary", learning_rate=0.03, max_depth=5, num_leaves=31,
            min_data_in_leaf=120, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, class_weight="balanced", n_estimators=400,
            n_jobs=-1, verbose=-1, random_state=42,
        )
        base = lgb.LGBMClassifier(**params)
        self.estimator_ = CalibratedClassifierCV(base, method=self.method, cv=self.cv)
        self.estimator_.fit(X, y)
        return self

    @property
    def classes_(self) -> np.ndarray:
        assert self.estimator_ is not None
        return self.estimator_.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.estimator_ is not None
        return self.estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.estimator_ is not None
        return self.estimator_.predict_proba(X)


def probability_distribution_stats(probs: np.ndarray) -> dict[str, float]:
    a = np.asarray(probs, dtype=float)
    if a.size == 0:
        return {k: float("nan") for k in
                ("min", "p10", "p25", "median", "p75", "p90", "max")}
    return {
        "min":    float(np.min(a)),
        "p10":    float(np.quantile(a, 0.10)),
        "p25":    float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "p75":    float(np.quantile(a, 0.75)),
        "p90":    float(np.quantile(a, 0.90)),
        "max":    float(np.max(a)),
    }
