"""Path E experimental: regress final_R on baseline-context features.

This is a design stub. It is intentionally NOT wired into eval. Adopt only if
iteration-2 verdict (Task 16) is NO-GO with recommendation #4 (build Path E).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import lightgbm as lgb  # noqa: F401


def train_expected_r_regressor(df: pd.DataFrame):
    """Train a LightGBM regressor predicting per-trade final_R.

    Args:
        df: harvest output with `baseline_sample` flag and `final_R` target.

    Returns:
        (model, metrics) where metrics is a dict with sample counts and prediction
        quantiles on the held-out test slice.
    """
    import lightgbm as lgb
    from xau_pro_bot.models.train_path_d import _feature_cols, split_time_70_15_15

    data = df[df["baseline_sample"]].dropna(subset=["final_R"])
    tr, va, te = split_time_70_15_15(data)
    fcols = _feature_cols(data)
    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_data_in_leaf=120,
        n_estimators=600,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    )
    model.fit(
        tr[fcols],
        tr["final_R"],
        eval_set=[(va[fcols], va["final_R"])],
        callbacks=[lgb.early_stopping(40)],
    )
    pred_te = model.predict(te[fcols])
    return model, {
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "mean_pred": float(pred_te.mean()),
        "p10_pred": float(pd.Series(pred_te).quantile(0.10)),
        "p90_pred": float(pd.Series(pred_te).quantile(0.90)),
        "feature_cols": fcols,
    }
