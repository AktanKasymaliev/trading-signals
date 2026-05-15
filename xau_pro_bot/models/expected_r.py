"""Path E: regress final_R on baseline-context features.

Produces a LightGBM regressor whose `predict` output is interpreted as the
expected per-trade R-multiple. Gating in eval thresholds on predicted_R.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _default_params() -> dict[str, Any]:
    return dict(
        objective="regression",
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_data_in_leaf=120,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=600,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
        importance_type="gain",
    )


def train_expected_r_regressor(
    df: pd.DataFrame,
    *,
    base_params: dict[str, Any] | None = None,
):
    """Train a LightGBM regressor on per-trade final_R.

    Args:
        df: harvest output with `baseline_sample` flag and `final_R` target.
        base_params: optional overrides (used by tests with smaller datasets).

    Returns:
        (model, metrics). `metrics["feature_importance"]` is a dict
        feature_name -> gain importance, sorted descending.

    Raises:
        ValueError: if the baseline slice has no usable rows or split produces
            an empty slice.
    """
    import lightgbm as lgb
    from xau_pro_bot.models.train_path_d import _feature_cols, split_time_70_15_15

    data = df[df["baseline_sample"]].dropna(subset=["final_R"])
    if data.empty:
        raise ValueError("expected_r: no baseline rows with final_R available")

    tr, va, te = split_time_70_15_15(data)
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise ValueError("expected_r: time split produced an empty slice")

    fcols = _feature_cols(data)
    params = _default_params()
    if base_params:
        params.update(base_params)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        tr[fcols],
        tr["final_R"],
        eval_set=[(va[fcols], va["final_R"])],
        callbacks=[lgb.early_stopping(40)],
    )
    pred_te = model.predict(te[fcols])

    importance = dict(zip(fcols, model.feature_importances_.tolist()))
    importance = dict(
        sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    return model, {
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "mean_pred": float(pred_te.mean()),
        "p10_pred": float(pd.Series(pred_te).quantile(0.10)),
        "p50_pred": float(pd.Series(pred_te).quantile(0.50)),
        "p90_pred": float(pd.Series(pred_te).quantile(0.90)),
        "feature_cols": fcols,
        "feature_importance": importance,
    }
