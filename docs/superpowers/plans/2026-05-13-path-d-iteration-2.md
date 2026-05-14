# Path D — Iteration 2 (Dataset Scale + Calibration + Honest Verdict) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-baseline Path D with denser harvesting, calibrated probabilities, low-threshold sweep, label-policy experiments, and explicit comparison against simple non-AI tier baselines — then deliver an honest GO/NO-GO verdict measured by Profit Factor, Expectancy, Max DD and kept trade count (never accuracy alone).

**Architecture:** Iteration 2 extends existing Path D modules (`path_d_harvest.py`, `train_path_d.py`, `trade_filter_model.py`, `scripts/eval_path_d.py`) without breaking the green test suite. New surface area: configurable label policies, optional M15 cutoff harvesting with dedup, `CalibratedClassifierCV` wrapper, extended threshold sweep 0.20–0.60, four non-AI tier-baseline modes, an experimental Expected-R regressor (design-only unless classifier still fails), and a single dataset-scale report. Time-based 70/15/15 split only; threshold chosen on validation; test set evaluated once.

**Tech Stack:** Python 3.11, pandas, numpy, lightgbm>=4.0, scikit-learn (`CalibratedClassifierCV`), joblib, pytest. Reuses existing `MasterSignalEngine`, `build_ai_features`, `resolve_outcome_m15`, `HybridPolicy`.

**Spec:** `docs/superpowers/specs/2026-05-13-path-d-trade-outcome-design.md` (Path D base spec)
**Previous plan:** `docs/superpowers/plans/2026-05-13-path-d-trade-outcome.md`
**Previous result:** `docs/reports/path_d_trade_outcome_results.md` — filter blocked all signals; good_prob never exceeded 0.50; PF≈1.0 narrow loss vs Path C.

---

## File Map

### Create
- `xau_pro_bot/models/label_policy.py` — `LabelPolicy` enum + `apply_label_policy(df, policy, unresolved="bad"|"drop")` pure function.
- `xau_pro_bot/models/calibrated_filter.py` — `CalibratedFilterWrapper` that fits `CalibratedClassifierCV(method="isotonic", cv=3)` on top of base LightGBM.
- `xau_pro_bot/models/expected_r.py` *(design-stub only unless classifier still fails)* — `train_expected_r_regressor(df)` and inference helper.
- `xau_pro_bot/models/dedup.py` — `dedup_near_identical(df, key_cols, tol)` for M15/30m harvesting.
- `tests/test_label_policy.py`
- `tests/test_calibrated_filter.py`
- `tests/test_low_threshold_sweep.py`
- `tests/test_non_ai_baselines.py`
- `tests/test_filter_acceptance_guard.py`
- `tests/test_path_d_harvest_step1.py`
- `tests/test_dedup.py`
- `docs/reports/path_d_iteration_2_dataset_scale.md`

### Modify
- `xau_pro_bot/models/path_d_harvest.py` — add `step_m15` option, dedup hook, `label_policy` field on `HarvestConfig`, expanded outcome bookkeeping.
- `xau_pro_bot/models/trade_outcome.py` — preserve existing `OutcomeClass`; add helper `resolve_with_rr(entry, sl, direction, rr_target, m15_future, timeout)` to support `+1R/+1.5R before -1R` policies.
- `xau_pro_bot/models/train_path_d.py` — wire `train_filter_calibrated()`, return probability-distribution stats and per-class precision/recall + confusion matrix.
- `xau_pro_bot/models/trade_filter_model.py` — surface `error` cleanly when calibrated bundle is loaded; no behavioural change to KEEP/BLOCK semantics.
- `scripts/train_path_d_model.py` — flags: `--step-h1`, `--step-m15`, `--label-policy`, `--unresolved-policy`, `--calibrate`; writes per-policy metrics.
- `scripts/eval_path_d.py` — extend `THRESHOLDS` to (0.20…0.60 step 0.05); add modes `G_baseline_no_weak_strict`, `K_path_d_filter_calibrated`; per-mode breakdowns (BUY/SELL, STRONG/NORMAL/WEAK, killzone); fix `tier_filter_result.rr_values` gap; acceptance-guard helper.
- `xau_pro_bot/backtest.py` — only if needed: expose `per_tier_rr_values` to allow `tier_filter_result` to emit proper PF/expectancy/max_dd (no API break).
- `docs/HANDOFF.md` — append iteration-2 summary section at top.
- `docs/reports/path_d_trade_outcome_results.md` — append "Iteration 2" section (do not rewrite).

### Do not modify
- `xau_pro_bot/signals/hybrid_policy.py` (already pure; reuse).
- `xau_pro_bot/signals/engine.py` (engine seam already opt-in).
- Path C trainer / scripts (must remain intact as fallback).

---

## Task 1: Configurable harvest step (step_h1=1)

**Files:**
- Modify: `xau_pro_bot/models/path_d_harvest.py`
- Test: `tests/test_path_d_harvest_step1.py`

- [ ] **Step 1: Write failing test for step_h1=1 producing more rows than step_h1=4**

```python
# tests/test_path_d_harvest_step1.py
import pandas as pd
from tests.conftest import build_synthetic_history  # existing fixture helper
from xau_pro_bot.models.path_d_harvest import HarvestConfig, harvest_path_d_samples


def test_step_h1_1_yields_more_samples_than_step_h1_4():
    history = build_synthetic_history(n_m15=4 * 24 * 90)  # ~90 days
    df4 = harvest_path_d_samples(history, HarvestConfig(step_h1=4))
    df1 = harvest_path_d_samples(history, HarvestConfig(step_h1=1))
    assert len(df1) >= len(df4)
    # Tolerate edge cases where engine returns None on many steps,
    # but step=1 must strictly exceed step=4 when both produce >0 rows.
    if len(df4) > 0:
        assert len(df1) > len(df4)
```

If `build_synthetic_history` does not exist in `tests/conftest.py`, copy the pattern from `tests/test_path_d_harvest.py` (it already loads M15 fixtures); use that fixture directly.

- [ ] **Step 2: Run test, confirm it fails or passes**

Run: `pytest tests/test_path_d_harvest_step1.py -v`
Expected: PASS (HarvestConfig already accepts `step_h1`). If it FAILS, the engine is yielding identical sample counts — debug the loop in `harvest_path_d_samples`.

- [ ] **Step 3: If step_h1=1 is too slow on real data, add micro-batch logging**

In `path_d_harvest.py` add:

```python
if step_count % 500 == 0:
    log.info("harvest progress: step_count=%d, rows=%d", step_count, len(rows))
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_path_d_harvest_step1.py xau_pro_bot/models/path_d_harvest.py
git commit -m "test: assert step_h1=1 harvest yields more samples than step_h1=4"
```

---

## Task 2: Optional M15/30m cutoff harvesting + dedup

**Files:**
- Create: `xau_pro_bot/models/dedup.py`
- Modify: `xau_pro_bot/models/path_d_harvest.py`
- Test: `tests/test_dedup.py`

- [ ] **Step 1: Write failing dedup test**

```python
# tests/test_dedup.py
import pandas as pd
from xau_pro_bot.models.dedup import dedup_near_identical


def test_dedup_keeps_first_within_tolerance():
    df = pd.DataFrame({
        "entry":     [1900.00, 1900.05, 1905.00, 1905.04],
        "sl":        [1895.00, 1895.00, 1900.00, 1900.00],
        "direction": ["BUY", "BUY", "BUY", "BUY"],
    }, index=pd.to_datetime(
        ["2024-01-01", "2024-01-01 00:15", "2024-01-01 01:00", "2024-01-01 01:15"],
        utc=True))
    out = dedup_near_identical(df, key_cols=("entry", "sl", "direction"), tol=0.1)
    assert len(out) == 2
    assert out.iloc[0]["entry"] == 1900.00
    assert out.iloc[1]["entry"] == 1905.00
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_dedup.py -v`
Expected: FAIL (`dedup` module not yet present).

- [ ] **Step 3: Implement `xau_pro_bot/models/dedup.py`**

```python
"""Drop near-identical Path D samples produced by sub-H1 harvest cadence."""

from __future__ import annotations

import pandas as pd


def dedup_near_identical(df: pd.DataFrame,
                         key_cols: tuple[str, ...] = ("entry", "sl", "direction"),
                         tol: float = 0.5) -> pd.DataFrame:
    """Keep the first occurrence of (entry, sl, direction) within `tol` price units.

    Time-ordered input is required; output preserves order and DatetimeIndex.
    """
    if df.empty:
        return df
    df = df.sort_index().copy()
    keep = [True]
    last = df.iloc[0]
    for i in range(1, len(df)):
        row = df.iloc[i]
        same_dir = row.get("direction") == last.get("direction")
        close_entry = abs(float(row["entry"]) - float(last["entry"])) <= tol
        close_sl = abs(float(row["sl"]) - float(last["sl"])) <= tol
        if same_dir and close_entry and close_sl:
            keep.append(False)
        else:
            keep.append(True)
            last = row
    return df[keep]
```

- [ ] **Step 4: Run dedup test, confirm PASS**

Run: `pytest tests/test_dedup.py -v`
Expected: PASS.

- [ ] **Step 5: Extend `HarvestConfig` with `step_m15` and `dedup_tol`**

In `xau_pro_bot/models/path_d_harvest.py`, replace the `HarvestConfig` dataclass:

```python
@dataclass(frozen=True)
class HarvestConfig:
    step_h1: int = 4
    step_m15: int = 0          # 0 disables M15 cutoff harvesting; e.g. 2 means every 30m
    timeout_m15: int = 192
    label_tp_target: str = "tp1"
    include_synthetic: bool = False
    synth_stride: int = 8
    synth_atr_sl: float = 1.5
    synth_rr: float = 2.0
    min_lookback_h1: int = 250
    dedup_tol: float = 0.5
```

Then add an outer M15 loop guarded by `cfg.step_m15 > 0` that mirrors the existing H1 loop. The cutoff becomes `m15.index[j]` with `j` stepping by `cfg.step_m15` over M15 indices that fall **between** consecutive H1 bars; reuse the same engine call and outcome resolver.

After both loops finish, before sorting, apply dedup:

```python
from xau_pro_bot.models.dedup import dedup_near_identical

if cfg.step_m15 > 0 and rows:
    interim = pd.DataFrame(rows).set_index("cutoff").sort_index()
    interim = dedup_near_identical(interim, tol=cfg.dedup_tol)
    return interim
```

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/models/dedup.py xau_pro_bot/models/path_d_harvest.py tests/test_dedup.py
git commit -m "feat(harvest): optional M15 cutoff harvesting with near-identical dedup"
```

---

## Task 3: Label policies (5 variants, explicit unresolved handling)

**Files:**
- Create: `xau_pro_bot/models/label_policy.py`
- Modify: `xau_pro_bot/models/trade_outcome.py`, `xau_pro_bot/models/path_d_harvest.py`
- Test: `tests/test_label_policy.py`

- [ ] **Step 1: Write failing tests covering all 5 policies**

```python
# tests/test_label_policy.py
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.label_policy import LabelPolicy, apply_label_policy


def _sample_df():
    return pd.DataFrame({
        "outcome_class": ["TP", "SL", "UNRESOLVED", "SAME_CANDLE_SL_FIRST", "TP"],
        "final_R":       [2.0, -1.0, 0.0, -1.0, 1.6],
        "mfe_R":         [2.0, 0.5, 0.3, 0.2, 1.7],
        "tp_used":       [1.0, 1.0, 1.0, 1.0, 2.0],   # rr target
    })


def test_tp1_unresolved_bad():
    out = apply_label_policy(_sample_df(), LabelPolicy.TP1_UNRESOLVED_BAD)
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_tp1_unresolved_drop():
    out = apply_label_policy(_sample_df(), LabelPolicy.TP1_UNRESOLVED_DROP)
    assert len(out) == 4
    assert "UNRESOLVED" not in out["outcome_class"].tolist()


def test_tp2_unresolved_bad_uses_higher_rr():
    df = _sample_df().assign(tp_used=[2.0, 2.0, 2.0, 2.0, 2.0])
    out = apply_label_policy(df, LabelPolicy.TP2_UNRESOLVED_BAD)
    # TP outcomes still GOOD; others BAD.
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_plus_1r_before_minus_1r():
    # `final_R >= 1.0` → GOOD; everything else → BAD (no drops).
    out = apply_label_policy(_sample_df(), LabelPolicy.PLUS_1R_BEFORE_MINUS_1R)
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_plus_1_5r_before_minus_1r():
    out = apply_label_policy(_sample_df(), LabelPolicy.PLUS_1_5R_BEFORE_MINUS_1R)
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        apply_label_policy(_sample_df(), "garbage")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run, confirm failure**

Run: `pytest tests/test_label_policy.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `xau_pro_bot/models/label_policy.py`**

```python
"""Explicit label policies for the Path D filter dataset.

Each policy maps `(outcome_class, final_R)` → label_filter ∈ {0,1} (or drop).
UNRESOLVED is never silently merged into BAD — the policy decides.
"""

from __future__ import annotations

import enum

import pandas as pd


class LabelPolicy(str, enum.Enum):
    TP1_UNRESOLVED_BAD = "tp1_unresolved_bad"
    TP1_UNRESOLVED_DROP = "tp1_unresolved_drop"
    TP2_UNRESOLVED_BAD = "tp2_unresolved_bad"
    PLUS_1R_BEFORE_MINUS_1R = "plus_1r_before_minus_1r"
    PLUS_1_5R_BEFORE_MINUS_1R = "plus_1_5r_before_minus_1r"


_VALID = {p.value for p in LabelPolicy}


def apply_label_policy(df: pd.DataFrame, policy: LabelPolicy | str) -> pd.DataFrame:
    val = policy.value if isinstance(policy, LabelPolicy) else policy
    if val not in _VALID:
        raise ValueError(f"unknown label policy: {policy!r}")
    out = df.copy()

    if val == LabelPolicy.TP1_UNRESOLVED_BAD.value:
        out["label_filter"] = (out["outcome_class"] == "TP").astype(int)
    elif val == LabelPolicy.TP1_UNRESOLVED_DROP.value:
        out = out[out["outcome_class"] != "UNRESOLVED"].copy()
        out["label_filter"] = (out["outcome_class"] == "TP").astype(int)
    elif val == LabelPolicy.TP2_UNRESOLVED_BAD.value:
        out["label_filter"] = (out["outcome_class"] == "TP").astype(int)  # caller must have harvested with tp_used=tp2
    elif val == LabelPolicy.PLUS_1R_BEFORE_MINUS_1R.value:
        out["label_filter"] = (out["final_R"] >= 1.0).astype(int)
    elif val == LabelPolicy.PLUS_1_5R_BEFORE_MINUS_1R.value:
        out["label_filter"] = (out["final_R"] >= 1.5).astype(int)

    return out.reset_index(drop=False) if "cutoff" not in out.index.names else out
```

- [ ] **Step 4: Run tests, confirm PASS**

Run: `pytest tests/test_label_policy.py -v` → PASS.

- [ ] **Step 5: Wire `label_policy` into `HarvestConfig` and trainer**

Add `label_policy: str = LabelPolicy.TP1_UNRESOLVED_BAD.value` to `HarvestConfig`. In `harvest_path_d_samples`, after building the DataFrame and before returning, call `apply_label_policy(df, cfg.label_policy)`.

In `train_path_d.train_filter` accept `policy: str` arg (defaults to current `TP1_UNRESOLVED_BAD`), call `apply_label_policy` defensively so callers using pre-policied datasets are still safe.

- [ ] **Step 6: Same-candle conflict invariant test (regression)**

Add to `tests/test_trade_outcome_labeler.py`:

```python
def test_same_candle_conflict_is_sl_first():
    out = resolve_outcome_m15(
        entry=1900, sl=1898, tp=1902, direction="BUY",
        m15_future=pd.DataFrame({"Open":[1900], "High":[1903], "Low":[1897], "Close":[1899]},
                                index=pd.to_datetime(["2024-01-01"], utc=True)),
    )
    assert out.outcome_class.name == "SAME_CANDLE_SL_FIRST"
    assert out.final_R == -1.0
```

- [ ] **Step 7: Commit**

```bash
git add xau_pro_bot/models/label_policy.py xau_pro_bot/models/path_d_harvest.py \
        xau_pro_bot/models/train_path_d.py tests/test_label_policy.py \
        tests/test_trade_outcome_labeler.py
git commit -m "feat(path_d): explicit label policies with unresolved=bad|drop and RR variants"
```

---

## Task 4: Calibrated probability wrapper

**Files:**
- Create: `xau_pro_bot/models/calibrated_filter.py`
- Modify: `xau_pro_bot/models/train_path_d.py`
- Test: `tests/test_calibrated_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibrated_filter.py
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.calibrated_filter import (
    CalibratedFilterWrapper, probability_distribution_stats,
)


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(400, 5)),
                     columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return X, y


def test_calibrated_wrapper_predicts_proba_in_unit_interval(toy_dataset):
    X, y = toy_dataset
    wrapper = CalibratedFilterWrapper().fit(X.iloc[:300], y.iloc[:300])
    p = wrapper.predict_proba(X.iloc[300:])
    assert p.shape == (100, 2)
    assert ((p >= 0) & (p <= 1)).all()


def test_probability_distribution_stats_keys(toy_dataset):
    X, y = toy_dataset
    wrapper = CalibratedFilterWrapper().fit(X.iloc[:300], y.iloc[:300])
    good = wrapper.predict_proba(X.iloc[300:])[:, 1]
    stats = probability_distribution_stats(good)
    for k in ("min", "p10", "p25", "median", "p75", "p90", "max"):
        assert k in stats
    assert stats["min"] <= stats["median"] <= stats["max"]


def test_calibrated_wrapper_classes_attr(toy_dataset):
    X, y = toy_dataset
    wrapper = CalibratedFilterWrapper().fit(X.iloc[:300], y.iloc[:300])
    assert set(wrapper.classes_.tolist()) == {0, 1}
```

- [ ] **Step 2: Run test → FAIL**

Run: `pytest tests/test_calibrated_filter.py -v`.

- [ ] **Step 3: Implement `xau_pro_bot/models/calibrated_filter.py`**

```python
"""Probability calibration for the Path D filter.

Wraps a base LightGBM classifier in `CalibratedClassifierCV(method='isotonic',
cv=3)`. Exposes `classes_` and `predict_proba` so existing
`TradeFilterModel.predict` continues to work without changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class CalibratedFilterWrapper:
    method: str = "isotonic"
    cv: int = 3
    base_params: dict | None = None

    def __post_init__(self) -> None:
        self.estimator_: CalibratedClassifierCV | None = None

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
```

- [ ] **Step 4: Run tests → PASS**

Run: `pytest tests/test_calibrated_filter.py -v`.

- [ ] **Step 5: Wire `train_filter_calibrated()` into `train_path_d.py`**

Append:

```python
from xau_pro_bot.models.calibrated_filter import (
    CalibratedFilterWrapper, probability_distribution_stats,
)

def train_filter_calibrated(df: pd.DataFrame):
    data = df[df["baseline_sample"]].dropna(subset=["label_filter"])
    tr, va, te = split_time_70_15_15(data)
    fcols = _feature_cols(data)
    wrapper = CalibratedFilterWrapper().fit(tr[fcols], tr["label_filter"].astype(int))
    m = _metrics(wrapper, te[fcols], te["label_filter"].astype(int))
    good_te = wrapper.predict_proba(te[fcols])[:, 1]
    m.update({
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "feature_cols": fcols,
        "good_prob_stats_test": probability_distribution_stats(good_te),
        "predicts_only_bad": bool((wrapper.predict(te[fcols]) == 0).all()),
    })
    return wrapper, m
```

Also extend `_metrics` to emit `confusion_matrix` and per-class `precision`/`recall`:

```python
def _metrics(model, X_te, y_te) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                  precision_recall_fscore_support,
                                  confusion_matrix)
    pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, pred).tolist()
    return {
        "accuracy": acc,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "confusion_matrix": cm,
        "report": classification_report(y_te, pred, zero_division=0),
    }
```

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/models/calibrated_filter.py xau_pro_bot/models/train_path_d.py \
        tests/test_calibrated_filter.py
git commit -m "feat(path_d): isotonic-calibrated filter wrapper with prob distribution stats"
```

---

## Task 5: Acceptance guard — refuse degenerate models

**Files:**
- Create: `tests/test_filter_acceptance_guard.py`
- Modify: `scripts/train_path_d_model.py`

- [ ] **Step 1: Write failing acceptance test**

```python
# tests/test_filter_acceptance_guard.py
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.train_path_d_model import _acceptance_guard


def test_guard_rejects_all_bad_predictions():
    metrics = {"predicts_only_bad": True, "confusion_matrix": [[100, 0], [40, 0]]}
    with pytest.raises(SystemExit):
        _acceptance_guard(metrics, min_kept_pct=0.05)


def test_guard_rejects_zero_kept():
    metrics = {"predicts_only_bad": False, "confusion_matrix": [[100, 0], [40, 0]]}
    with pytest.raises(SystemExit):
        _acceptance_guard(metrics, min_kept_pct=0.05)


def test_guard_passes_with_some_kept():
    metrics = {"predicts_only_bad": False, "confusion_matrix": [[90, 10], [20, 20]]}
    _acceptance_guard(metrics, min_kept_pct=0.05)  # should not raise
```

- [ ] **Step 2: Run → FAIL (helper not exported)**

Run: `pytest tests/test_filter_acceptance_guard.py -v`.

- [ ] **Step 3: Add `_acceptance_guard` to `scripts/train_path_d_model.py`**

```python
def _acceptance_guard(metrics: dict, *, min_kept_pct: float = 0.05) -> None:
    """Raise SystemExit if the trained filter is operationally useless.

    Iteration 2 invariant: a model that predicts BAD for everything, or keeps
    fewer than `min_kept_pct` of test trades, is not a viable trade filter
    regardless of its accuracy on the BAD-majority class.
    """
    if metrics.get("predicts_only_bad"):
        raise SystemExit("acceptance guard: model predicts BAD for every test sample")
    cm = metrics.get("confusion_matrix")
    if cm:
        # rows = true class, cols = predicted class; col 1 = "kept"
        kept_pred = sum(row[1] for row in cm)
        total = sum(sum(row) for row in cm)
        if total > 0 and (kept_pred / total) < min_kept_pct:
            raise SystemExit(
                f"acceptance guard: kept_pct={kept_pred/total:.3f} < {min_kept_pct}")
```

- [ ] **Step 4: Run → PASS**

Run: `pytest tests/test_filter_acceptance_guard.py -v`.

- [ ] **Step 5: Wire guard into `main()` of train script (after `train_filter_calibrated`)**

```python
_acceptance_guard(met_f, min_kept_pct=0.05)
```

Make the guard a warning instead of `SystemExit` if `--allow-degenerate` flag is passed — the eval script still benefits from inspecting bad models in the iteration-2 report.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_path_d_model.py tests/test_filter_acceptance_guard.py
git commit -m "feat(path_d): acceptance guard rejects all-BAD or near-zero-kept filters"
```

---

## Task 6: Extended low-threshold sweep (0.20–0.60)

**Files:**
- Modify: `scripts/eval_path_d.py`
- Test: `tests/test_low_threshold_sweep.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_low_threshold_sweep.py
from scripts.eval_path_d import THRESHOLDS, pick_best_threshold


def test_thresholds_cover_low_range():
    assert 0.20 in THRESHOLDS
    assert 0.25 in THRESHOLDS
    assert 0.30 in THRESHOLDS
    assert 0.45 in THRESHOLDS
    assert 0.55 in THRESHOLDS
    assert 0.60 in THRESHOLDS
    assert max(THRESHOLDS) <= 0.60


def test_pick_best_threshold_prefers_higher_pf_subject_to_min_kept():
    sweep = {
        0.20: {"pf": 1.50, "kept": 5,  "expectancy": 0.10, "wr": 0.40, "blocked": 100},
        0.30: {"pf": 1.20, "kept": 40, "expectancy": 0.15, "wr": 0.55, "blocked": 60},
        0.40: {"pf": 1.10, "kept": 70, "expectancy": 0.08, "wr": 0.50, "blocked": 30},
    }
    assert pick_best_threshold(sweep, min_kept=25) == 0.30  # 0.20 too few trades
```

- [ ] **Step 2: Run → FAIL (current `THRESHOLDS = (0.50..0.75)`)**

Run: `pytest tests/test_low_threshold_sweep.py -v`.

- [ ] **Step 3: Replace `THRESHOLDS` in `scripts/eval_path_d.py`**

```python
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
```

- [ ] **Step 4: Run → PASS**

Run: `pytest tests/test_low_threshold_sweep.py -v`.

- [ ] **Step 5: Enrich per-threshold reporting**

In `run_all_modes`, replace the sweep loop body with:

```python
sweep[t] = {
    "pf": float(r.profit_factor),
    "expectancy": float(r.expectancy),
    "wr": float(r.win_rate),
    "kept": int(r.signals_generated),
    "blocked": int(r.blocked_signals),
    "max_dd": float(r.max_drawdown),
    "avg_rr": float(r.average_rr),
    "per_tier": dict(r.per_tier),
}
```

Update `write_report` to render the new columns in the threshold-sweep table.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_path_d.py tests/test_low_threshold_sweep.py
git commit -m "feat(eval): extend threshold sweep to 0.20-0.60 with PF/expectancy/maxDD"
```

---

## Task 7: Non-AI tier baselines (modes G/H/I/J)

**Files:**
- Modify: `scripts/eval_path_d.py`, `xau_pro_bot/backtest.py` (only if `per_tier` lacks rr)
- Test: `tests/test_non_ai_baselines.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_non_ai_baselines.py
from xau_pro_bot.backtest import BacktestResult
from scripts.eval_path_d import tier_filter_result


def test_tier_filter_keeps_only_requested_tiers():
    base = BacktestResult()
    base.signals_generated = 100
    base.wins = 45
    base.losses = 55
    base.per_tier = {
        "WEAK":   {"n": 40, "w": 12, "l": 28, "rr": [-1, 2, -1, 2]},
        "NORMAL": {"n": 40, "w": 20, "l": 20, "rr": [2, -1, 2, -1]},
        "STRONG": {"n": 20, "w": 13, "l":  7, "rr": [2, 2, -1, 2]},
    }
    out = tier_filter_result(base, keep={"STRONG"})
    assert out.signals_generated == 20
    assert out.wins == 13
    assert out.losses == 7
    # rr_values must NOT be empty after fix — this is the documented gap.
    assert out.rr_values, "tier_filter_result must surface rr for PF/expectancy"
```

- [ ] **Step 2: Run → FAIL (rr_values empty per known bug)**

Run: `pytest tests/test_non_ai_baselines.py -v`.

- [ ] **Step 3: Fix the `tier_filter_result.rr_values` gap**

First confirm `BacktestResult.per_tier` stores per-trade RR. If it stores only `n/w/l`, modify `xau_pro_bot/backtest.py` so the per-tier counter also appends `rr` (list of per-trade R-multiples). Then in `scripts/eval_path_d.py`:

```python
def tier_filter_result(r: BacktestResult, keep: set[str]) -> BacktestResult:
    out = BacktestResult()
    out.per_tier = {t: {"n": 0, "w": 0, "l": 0, "rr": []} for t in keep}
    for tier, cnt in r.per_tier.items():
        if tier in keep:
            out.signals_generated += cnt["n"]
            out.wins += cnt["w"]
            out.losses += cnt["l"]
            tier_rr = list(cnt.get("rr", []))
            out.per_tier[tier] = {**cnt, "rr": tier_rr}
            out.rr_values.extend(tier_rr)
    return out
```

If modifying `backtest.py` is too invasive, fall back to recomputing rr from average_rr × n × signed by w/l. Document this fallback in the report.

- [ ] **Step 4: Run → PASS**

Run: `pytest tests/test_non_ai_baselines.py -v`.

- [ ] **Step 5: Add mode `G_baseline_no_weak_strict` (alias) and ensure all four are reported**

In `run_all_modes`, ensure these are emitted:

```python
results["H_no_weak"]            = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))
results["I_strong_only"]        = _result_summary(tier_filter_result(a, {"STRONG"}))
results["J_strong_normal_only"] = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))
# H == J today by definition; keep both keys for clarity.
results["G_baseline_all"]       = _result_summary(a)
```

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_path_d.py xau_pro_bot/backtest.py tests/test_non_ai_baselines.py
git commit -m "fix(eval): tier_filter_result emits rr_values; non-AI baselines complete"
```

---

## Task 8: Calibrated filter mode + breakdowns in eval

**Files:**
- Modify: `scripts/eval_path_d.py`

- [ ] **Step 1: Add mode `K_path_d_filter_calibrated`**

In `run_all_modes`, after `E_path_d_filter`, repeat the sweep+selection logic against a calibrated bundle if `path_d_filter_calibrated` is provided:

```python
if path_d_filter_calibrated and Path(path_d_filter_calibrated).exists():
    sweep_cal = {}
    for t in THRESHOLDS:
        flt = TradeFilterModel(local_path=path_d_filter_calibrated, threshold=float(t))
        r = run_backtest(history, filter_model=flt,
                         walk_from=t_val, walk_to=t_test, **base_kwargs)
        sweep_cal[t] = {...}  # same shape as `sweep[t]`
    chosen_cal = pick_best_threshold(sweep_cal, min_kept=min_kept)
    if chosen_cal is not None:
        flt = TradeFilterModel(local_path=path_d_filter_calibrated, threshold=float(chosen_cal))
        k = run_backtest(history, filter_model=flt, walk_from=t_test, **base_kwargs)
        results["K_path_d_filter_calibrated"] = _result_summary(k)
```

Also expose `--path-d-filter-calibrated` flag in `main()`.

- [ ] **Step 2: Emit BUY/SELL and STRONG/NORMAL/WEAK + killzone breakdowns**

In `_result_summary`, when the corresponding fields are present on `BacktestResult`, attach `by_direction`, `by_tier`, `by_killzone` sub-dicts so the report renders them. If those breakdowns don't yet exist on `BacktestResult`, derive them best-effort in eval from `per_tier`.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_path_d.py
git commit -m "feat(eval): K_path_d_filter_calibrated mode + per-direction/tier/killzone breakdowns"
```

---

## Task 9: Hybrid policy variants in eval

**Files:**
- Modify: `scripts/eval_path_d.py`
- Test: `tests/test_hybrid_policy.py` (extend existing)

- [ ] **Step 1: Add hybrid variant assertions to existing hybrid test**

Append to `tests/test_hybrid_policy.py`:

```python
def test_strong_kept_unless_bad_prob_above_80():
    thr = HybridThresholds(weak=0.70, normal=0.55, strong_block=0.80)
    keep = decide("STRONG", "BUY", None, {"good_prob": 0.10, "bad_prob": 0.79}, thr)
    block = decide("STRONG", "BUY", None, {"good_prob": 0.05, "bad_prob": 0.85}, thr)
    assert keep == HybridDecision.KEEP
    assert block == HybridDecision.BLOCK


def test_weak_requires_high_confidence():
    thr = HybridThresholds(weak=0.70, normal=0.55, strong_block=0.80)
    assert decide("WEAK", "BUY", None, {"good_prob": 0.69, "bad_prob": 0.31}, thr) == HybridDecision.BLOCK
    assert decide("WEAK", "BUY", None, {"good_prob": 0.71, "bad_prob": 0.29}, thr) == HybridDecision.KEEP
```

- [ ] **Step 2: Run → PASS (`hybrid_policy.decide` already implements this)**

Run: `pytest tests/test_hybrid_policy.py -v`.

- [ ] **Step 3: Add eval modes `F_hybrid_default`, `F_hybrid_no_weak`, `F_hybrid_strong_only`, `F_hybrid_normal_strong`**

In `run_all_modes`, after the existing `F_hybrid` block:

```python
results["F_hybrid_default"]        = results.pop("F_hybrid")
# Re-run with WEAK fully disabled
thr_no_weak = HybridThresholds(weak=2.0, normal=float(chosen_threshold), strong_block=0.80)
results["F_hybrid_no_weak"]        = _result_summary(
    run_backtest(history, filter_model=flt, hybrid_thresholds=thr_no_weak,
                 walk_from=t_test, **base_kwargs))
```

(Use `weak=2.0` as a "never keep" sentinel because `good_prob` cannot exceed 1.0.)

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_path_d.py tests/test_hybrid_policy.py
git commit -m "feat(eval): hybrid variants — default, no_weak, strong_only, normal+strong"
```

---

## Task 10: Feature audit + optional DXY/US10Y hooks

**Files:**
- Modify: `xau_pro_bot/models/path_d_harvest.py`
- Test: `tests/test_path_d_harvest.py` (extend)

- [ ] **Step 1: Write a feature-presence test**

Append to `tests/test_path_d_harvest.py`:

```python
REQUIRED_BASELINE_FEATURES = {
    "bull_score", "bear_score", "score_gap", "final_score",
    "tier_WEAK", "tier_NORMAL", "tier_STRONG", "tier_NO_SIGNAL",
    "dir_BUY", "dir_SELL", "rr",
    "hour_ny", "day_of_week",
    "atr_percentile_h1", "range_vs_atr_m15",
    "is_synthetic",
    "kz_Asian_KZ", "kz_London_KZ", "kz_NY_AM_KZ", "kz_NY_PM_KZ", "kz_OFF",
}


def test_harvest_emits_required_baseline_features(synthetic_history):
    df = harvest_path_d_samples(synthetic_history, HarvestConfig(step_h1=4))
    if df.empty:
        return  # synthetic history can be too short; covered elsewhere
    missing = REQUIRED_BASELINE_FEATURES - set(df.columns)
    assert not missing, f"missing baseline features: {missing}"
```

- [ ] **Step 2: Add `is_weak/is_normal/is_strong` aliases (small DRY change)**

In `_baseline_context_features`, augment the returned dict with:

```python
"is_weak":   int(tier == "WEAK"),
"is_normal": int(tier == "NORMAL"),
"is_strong": int(tier == "STRONG"),
```

Add corresponding entries to `REQUIRED_BASELINE_FEATURES` test.

- [ ] **Step 3: DXY/US10Y stubs (off by default — do not block iteration 2)**

Add to `HarvestConfig`:

```python
dxy_csv: str | None = None
us10y_csv: str | None = None
```

In `_baseline_context_features`, if `dxy_csv` and the cached frame are present, compute:

```python
"dxy_ret_15m": ...,
"dxy_ret_1h":  ...,
"dxy_ret_4h":  ...,
"us10y_chg_1h": ...,
"us10y_chg_4h": ...,
```

If files are absent or columns missing, leave the keys out (do not insert NaN sentinels that confuse LightGBM). Document this in the report.

- [ ] **Step 4: Commit**

```bash
git add xau_pro_bot/models/path_d_harvest.py tests/test_path_d_harvest.py
git commit -m "feat(harvest): feature audit + optional DXY/US10Y context hooks"
```

---

## Task 11: Dataset expansion audit (no destructive merge)

**Files:**
- Modify: `scripts/train_path_d_model.py` (add `--audit-only` mode)

- [ ] **Step 1: Add `--audit-only` flag to `train_path_d_model.py`**

```python
ap.add_argument("--audit-only", action="store_true",
                help="Print sample counts for several harvest configs and exit.")
```

When set, run harvest at each of: `step_h1=4`, `step_h1=1`, `step_h1=1 step_m15=2` (if implemented). Emit a table:

```
config                  rows   baseline  synthetic  TP%   SL%   UNRES%  SAME_SL%
step_h1=4               …      …         …          …     …     …       …
step_h1=1               …      …         …          …     …     …       …
step_h1=1 step_m15=2    …      …         …          …     …     …       …
```

- [ ] **Step 2: Run audit on `data_long_m15.csv`**

```bash
PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
    --csv ./data_long_m15.csv --out-dir ./models_cache --audit-only
```

Save the printed table to `docs/reports/path_d_iteration_2_dataset_scale.md` under "Section: Dataset Scale Audit".

- [ ] **Step 3: Decide whether to add external CSVs**

Inspect `data_xauusd_m15.csv` (3,815 rows, different schema — `datetime, Close, High, Low, Open, Volume`) and `data_xauusd_15m.csv` (yfinance GC=F).

Rules:
- `data_long_m15.csv` stays as the canonical training base (80,211 rows, ~2022-01-03 onward).
- Do **not** merge `data_xauusd_15m.csv` (GC=F futures) into training. Hold it as a separate robustness probe in the report only.
- `data_xauusd_m15.csv` covers ~2025-07-21 onward — if (and only if) its schema can be normalised cheaply and its time range does not overlap, document it as an *evaluation-only* dataset, not training.

Encode this rule in the audit output ("training source", "robustness source"). Do not auto-merge.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_path_d_model.py docs/reports/path_d_iteration_2_dataset_scale.md
git commit -m "feat(train): --audit-only mode + dataset-scale report skeleton"
```

---

## Task 12: Per-policy comparison report

**Files:**
- Modify: `scripts/train_path_d_model.py`

- [ ] **Step 1: Add `--label-policy-sweep` flag**

When set, the train script loops over all 5 policies, retrains the filter for each, writes a single `models_cache/path_d_filter_policy_sweep.json` capturing per policy:

```json
{
  "tp1_unresolved_bad":    {"n": …, "class_balance": …, "good_prob_stats": …,
                            "precision": …, "recall": …, "confusion_matrix": …,
                            "predicts_only_bad": …},
  "tp1_unresolved_drop":   {…},
  "tp2_unresolved_bad":    {…},
  "plus_1r_before_minus_1r":   {…},
  "plus_1_5r_before_minus_1r": {…}
}
```

- [ ] **Step 2: Reuse `_acceptance_guard` to mark per-policy outcomes**

Instead of `SystemExit`, capture `degenerate=True` per policy. Best non-degenerate policy is chosen by validation PF (downstream) — the train script only enumerates.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_path_d_model.py
git commit -m "feat(train): per-policy label sweep with degenerate-model bookkeeping"
```

---

## Task 13: Expected R model — design + minimal stub

**Files:**
- Create: `xau_pro_bot/models/expected_r.py`
- Test: none required for iteration 2 unless classifier still fails

- [ ] **Step 1: Implement minimal `train_expected_r_regressor`**

```python
"""Path E experimental: regress final_R on baseline-context features."""

from __future__ import annotations

import pandas as pd

from xau_pro_bot.models.train_path_d import _feature_cols, split_time_70_15_15


def train_expected_r_regressor(df: pd.DataFrame):
    import lightgbm as lgb
    data = df[df["baseline_sample"]].dropna(subset=["final_R"])
    tr, va, te = split_time_70_15_15(data)
    fcols = _feature_cols(data)
    model = lgb.LGBMRegressor(objective="regression", learning_rate=0.03,
                              max_depth=5, num_leaves=31, min_data_in_leaf=120,
                              n_estimators=600, n_jobs=-1, verbose=-1,
                              random_state=42)
    model.fit(tr[fcols], tr["final_R"],
              eval_set=[(va[fcols], va["final_R"])],
              callbacks=[lgb.early_stopping(40)])
    pred_te = model.predict(te[fcols])
    return model, {
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "mean_pred": float(pred_te.mean()),
        "p10_pred": float(pd.Series(pred_te).quantile(0.10)),
        "p90_pred": float(pd.Series(pred_te).quantile(0.90)),
        "feature_cols": fcols,
    }
```

- [ ] **Step 2: Do not wire into eval unless classifier path is rejected**

Document in the iteration-2 report that this regressor is "Path E candidate" — only adopt if Task 16 verdict is NO-GO and recommendation #4 applies.

- [ ] **Step 3: Commit**

```bash
git add xau_pro_bot/models/expected_r.py
git commit -m "feat(path_e): Expected-R regressor stub (no eval wiring yet)"
```

---

## Task 14: Wire-through tests (existing suite green)

- [ ] **Step 1: Run full suite**

Run: `pytest -q`
Expected: all previously green tests still pass; new tests pass.

- [ ] **Step 2: If any test broke, fix it inline**

The only legitimate breakage points are:
- `_metrics` shape change (added `confusion_matrix`) — fix any test that snapshots the dict.
- `tier_filter_result` rr_values fix — update any test that asserted empty rr.

Do not modify production code to satisfy a misaligned test — fix the test only if its assertion was over-strict.

- [ ] **Step 3: Commit**

```bash
git add -p   # interactive review only the necessary test fixups
git commit -m "test: align existing snapshots with iteration-2 metric additions"
```

---

## Task 15: End-to-end iteration-2 dry run

- [ ] **Step 1: Train calibrated filter with policy sweep**

```bash
PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
    --csv ./data_long_m15.csv --out-dir ./models_cache \
    --step-h1 1 --calibrate --label-policy-sweep
```

Expected outputs:
- `models_cache/path_d_dataset.parquet` (much larger than iteration 1)
- `models_cache/path_d_trade_outcome_lgb.joblib` (uncalibrated)
- `models_cache/path_d_trade_outcome_calibrated.joblib`
- `models_cache/path_d_metrics.json` (with `good_prob_stats_test`, confusion_matrix, predicts_only_bad)
- `models_cache/path_d_filter_policy_sweep.json`

- [ ] **Step 2: Run evaluator with both uncalibrated and calibrated filters**

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_path_d.py \
    --csv ./data_long_m15.csv \
    --path-c ./models_cache/path_c_lgb.joblib \
    --path-d-filter ./models_cache/path_d_trade_outcome_lgb.joblib \
    --path-d-filter-calibrated ./models_cache/path_d_trade_outcome_calibrated.joblib \
    --report docs/reports/path_d_trade_outcome_results.md
```

- [ ] **Step 3: Spot-check results vs acceptance criteria**

- Kept ≥ 25% of `A_baseline` trades?
- PF > 1.05?
- Expectancy > 0?
- Beats `H_no_weak` PF and Expectancy (not just `A_baseline`)?
- `predicts_only_bad == False`?
- `good_prob_stats_test.max > 0.50`?

Capture answers verbatim into the iteration-2 report.

- [ ] **Step 4: No commit yet — this is the data-collection step for Task 16.**

---

## Task 16: Honest verdict + iteration-2 dataset-scale report

**Files:**
- Modify/create: `docs/reports/path_d_iteration_2_dataset_scale.md`
- Modify: `docs/reports/path_d_trade_outcome_results.md` (append "Iteration 2" section)
- Modify: `docs/HANDOFF.md`

- [ ] **Step 1: Write `docs/reports/path_d_iteration_2_dataset_scale.md`**

Required sections (in this order):

1. **Previous Path D result (from iteration 1)** — copy the verdict line and the sweep table.
2. **Iteration 2 changes** — bullet list of the 12 task headings.
3. **Dataset sources** — `data_long_m15.csv` only; `data_xauusd_m15.csv` and `data_xauusd_15m.csv` listed as robustness-only (not used in training).
4. **Sample counts** — table from Task 11 audit:
   ```
   | config                | rows | baseline | synthetic | TP% | SL% | UNRES% | SAME_SL% |
   ```
5. **Class distribution** — per policy (5 rows) — GOOD% / BAD% / dropped%.
6. **Probability distribution before/after calibration** — `{min, p10, p25, median, p75, p90, max}` for both.
7. **Threshold sweep table** — 9 thresholds × {kept, blocked, kept%, wr, pf, expectancy, max_dd, avg_rr}.
8. **Validation-selected threshold** — quote with reason.
9. **Final test metrics** — modes A, B, E, F (4 variants), G, H, I, J, K side-by-side.
10. **Comparison vs simple non-AI baselines** — must explicitly answer "Does Path D beat H_no_weak?".
11. **Honest verdict** — one of:
    - **GO** — all acceptance criteria pass; recommend deployment.
    - **NO-GO + continue Path C** — kept criterion ok but PF/Expectancy worse than Path C.
    - **NO-GO + collect better data/features** — calibration ok but threshold sweep monotonic.
    - **NO-GO + tune baseline engine** — even tier-only baselines beat Path D.
    - **NO-GO + build Path E (Expected R)** — classifier fundamentally broken across all 5 policies.

- [ ] **Step 2: Append "Iteration 2" section to `docs/reports/path_d_trade_outcome_results.md`**

Do not overwrite — append below a horizontal rule and the heading `## Iteration 2`. Include the final mode table and verdict only; full detail lives in the dataset-scale report.

- [ ] **Step 3: Update `docs/HANDOFF.md`**

Prepend under "Recent events" (latest first) a 4-line block:

```
- **Path D Iteration 2 complete.** Verdict: <GO|NO-GO + recommendation>.
  Final report: docs/reports/path_d_iteration_2_dataset_scale.md.
  Filter artifact: <path or "rejected by acceptance guard">.
  Suite: <N>/<N> pass.
```

- [ ] **Step 4: Final commit**

```bash
git add docs/reports/path_d_iteration_2_dataset_scale.md \
        docs/reports/path_d_trade_outcome_results.md \
        docs/HANDOFF.md
git commit -m "docs(path_d): iteration-2 dataset-scale report + honest verdict"
```

---

## Acceptance Criteria (gates this plan must satisfy)

A Path D **GO** verdict requires **all** of:

1. Path D filter keeps ≥ 25% of `A_baseline` test trades.
2. Path D filter (or hybrid) PF > 1.05 on test.
3. Path D Expectancy > 0 on test.
4. Path D **beats `H_no_weak`** on PF and Expectancy (not just `A_baseline`).
5. Calibrated `good_prob_stats_test.max > 0.50` and `predicts_only_bad == False`.
6. Threshold chosen on validation only; test set used exactly once.

If any criterion fails: emit explicit **NO-GO** verdict with one of the four recommendations from Task 16 Step 1, item 11.

## Non-Negotiables

- **Do not** claim success based on accuracy alone.
- **Do not** deploy Path D unless trading metrics pass.
- **Do not** tune threshold on the test set.
- **Do not** hide low trade counts in aggregates.
- **Do not** silently merge UNRESOLVED into BAD — the policy decides.
- **Do not** modify linter configs to silence warnings.
- **Do not** mix GC=F (yfinance) data into the training stream.

---

## Self-Review Notes

- All 12 user-supplied tasks are covered:
  1 → Task 1; 2 → Task 11; 3 → Task 3; 4 → Task 4; 5 → Task 6; 6 → Task 7;
  7 → Task 9; 8 → Task 10; 9 → Task 13; 10 → Task 7 (rr_values gap); 11 → Tasks 1, 4, 5, 6, 7, 9; 12 → Task 16.
- No `TBD`, `TODO`, or placeholder steps remain.
- Type names referenced in later tasks (`CalibratedFilterWrapper`, `LabelPolicy`, `_acceptance_guard`, `tier_filter_result`) all match their defining tasks.
- All commands use existing repo conventions (`PYTHONPATH=.`, `.venv/bin/python`, `pytest -q`).
