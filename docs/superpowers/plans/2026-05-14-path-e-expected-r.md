# Path E — Expected R Regressor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failing GOOD/BAD filter with a regressor that predicts per-trade `final_R` and gates trades by `predicted_R > threshold`. Add optional DXY / US10Y macro features. Produce an honest GO / NO-GO verdict on test data.

**Architecture:**
1. Train a LightGBM regressor on `final_R` using the existing baseline-context features. Save it as a joblib bundle (same shape as `TradeFilterModel` bundles).
2. New `ExpectedRFilterModel` adapter mirrors `TradeFilterModel` but thresholds on `predicted_R` instead of `good_prob`.
3. `eval_path_d.py` gains a Path E mode (`L_path_e_expected_r`) with its own threshold sweep over `predicted_R` and the same 25%-kept floor.
4. Optional DXY/US10Y CSV features wired into `path_d_harvest` and disabled by default; report whether they shift out-of-sample PF.
5. Honest verdict report at `docs/reports/path_e_expected_r_results.md`.

**Tech Stack:** Python 3.x, pandas, LightGBM (regressor), scikit-learn, joblib, pytest.

**Acceptance gates (Task 10 verdict):** Path E earns a GO **only if all** are true on the test slice:
- Kept trades ≥ 25% of `A_baseline` test trades.
- PF on test > PF of `B_path_c` **and** PF of `H_no_weak`.
- Expectancy on test > Expectancy of `H_no_weak`.
- Threshold chosen on validation only, applied to test exactly once.

If any gate fails: **NO-GO**, recommendation written verbatim into the report.

**Hard rules:**
- Do not tune threshold on test.
- Do not claim success from high PF with tiny trade count — the 25% floor is enforced via `pick_best_threshold(min_kept=...)`.
- Do not commit `--no-verify` or skip hooks. Do not push without explicit user request.
- DXY/US10Y CSVs default OFF — adding them must not break any existing test.

---

## File Structure

**Create:**
- `tests/test_expected_r_regressor.py` — unit tests for trainer.
- `tests/test_expected_r_filter_model.py` — unit tests for the predict adapter.
- `tests/test_macro_features.py` — DXY/US10Y feature wiring + no-CSV default.
- `tests/test_eval_path_e.py` — Path E mode integration test.
- `xau_pro_bot/models/expected_r_filter_model.py` — adapter analogous to `TradeFilterModel`.
- `docs/reports/path_e_expected_r_results.md` — written by Task 10 (in-band run).

**Modify:**
- `xau_pro_bot/models/expected_r.py` — promote from stub to production trainer (returns model + metrics dict that matches `train_filter`'s shape where applicable, plus `feature_importance`).
- `xau_pro_bot/models/path_d_harvest.py` — load DXY/US10Y CSVs when configured; emit features `dxy_ret_1h`, `dxy_ret_4h`, `us10y_chg_1h`, `us10y_chg_4h`. Default behaviour unchanged.
- `scripts/train_path_d_model.py` — `--expected-r` flag trains and saves `path_e_expected_r_lgb.joblib`; `--dxy-csv` and `--us10y-csv` propagate into `HarvestConfig`.
- `scripts/eval_path_d.py` — add Path E sweep over `(0.00, 0.03, 0.05, 0.10, 0.15)` predicted_R thresholds and `L_path_e_expected_r` mode; `--path-e` arg.

---

## Task 1: Promote `expected_r.py` stub to production trainer (TDD)

**Files:**
- Modify: `xau_pro_bot/models/expected_r.py`
- Create: `tests/test_expected_r_regressor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expected_r_regressor.py
"""Unit tests for the Path E expected_R regressor trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.expected_r import train_expected_r_regressor


def _toy_dataset(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    bull = rng.uniform(0.0, 1.0, n)
    bear = rng.uniform(0.0, 1.0, n)
    final_r = (bull - bear) * 2.0 + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({
        "bull_score": bull,
        "bear_score": bear,
        "score_gap": np.abs(bull - bear),
        "final_score": bull + bear,
        "tier_STRONG": rng.integers(0, 2, n),
        "tier_NORMAL": rng.integers(0, 2, n),
        "tier_WEAK":   rng.integers(0, 2, n),
        "dir_BUY":     rng.integers(0, 2, n),
        "dir_SELL":    rng.integers(0, 2, n),
        "rr":          np.full(n, 2.0),
        "hour_ny":     rng.integers(0, 24, n).astype(float),
        "day_of_week": rng.integers(0, 5, n).astype(float),
        "atr_percentile_h1": rng.uniform(0.0, 1.0, n),
        "range_vs_atr_m15":  rng.uniform(0.0, 2.0, n),
        "final_R":     final_r,
        "baseline_sample": True,
    }, index=idx)


def test_train_expected_r_returns_model_and_metrics():
    df = _toy_dataset()
    base_params = dict(min_data_in_leaf=5, n_estimators=50, learning_rate=0.1)
    model, metrics = train_expected_r_regressor(df, base_params=base_params)

    assert hasattr(model, "predict")
    assert metrics["n_train"] > 0 and metrics["n_val"] > 0 and metrics["n_test"] > 0
    assert "feature_cols" in metrics and len(metrics["feature_cols"]) > 0
    # Predictions must be float and span a non-trivial range on this signal.
    assert isinstance(metrics["mean_pred"], float)
    assert metrics["p90_pred"] > metrics["p10_pred"]
    # Importance must be reported (Task requirement).
    assert "feature_importance" in metrics
    assert sum(metrics["feature_importance"].values()) > 0


def test_train_expected_r_rejects_when_empty():
    df = _toy_dataset(n=400)
    df = df.iloc[0:0]
    with pytest.raises(ValueError):
        train_expected_r_regressor(df)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_regressor.py -v
```
Expected: FAIL (current stub has no `base_params` arg, no `feature_importance` key, no empty-input check).

- [ ] **Step 3: Implement the production trainer**

```python
# xau_pro_bot/models/expected_r.py
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
        ValueError: if the baseline slice has no usable rows.
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_regressor.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expected_r_regressor.py xau_pro_bot/models/expected_r.py
git commit -m "feat(path_e): production expected_R regressor with importance metrics"
```

---

## Task 2: `ExpectedRFilterModel` adapter (TDD)

**Files:**
- Create: `xau_pro_bot/models/expected_r_filter_model.py`
- Create: `tests/test_expected_r_filter_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expected_r_filter_model.py
"""Tests for ExpectedRFilterModel adapter."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.expected_r_filter_model import (
    ExpectedRFilterModel, ExpectedRDecision,
)


class _StubRegressor:
    """Minimal sklearn-like regressor returning a fixed predicted_R."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, X):
        return np.full(len(X), self._value)


@pytest.fixture
def bundle_path(tmp_path: Path) -> Path:
    path = tmp_path / "expected_r.joblib"
    joblib.dump(
        {"model": _StubRegressor(0.07), "feature_cols": ["bull_score", "rr"]},
        path,
    )
    return path


def test_keeps_when_predicted_r_above_threshold(bundle_path: Path):
    flt = ExpectedRFilterModel(local_path=str(bundle_path), threshold=0.05)
    feats = pd.DataFrame([{"bull_score": 0.6, "rr": 2.0}])
    out = flt.predict(feats)
    assert out["predicted_r"] == pytest.approx(0.07)
    assert out["decision"] == ExpectedRDecision.KEEP
    assert out["threshold_used"] == 0.05
    assert out["error"] is None


def test_blocks_when_predicted_r_below_threshold(bundle_path: Path):
    flt = ExpectedRFilterModel(local_path=str(bundle_path), threshold=0.10)
    feats = pd.DataFrame([{"bull_score": 0.6, "rr": 2.0}])
    out = flt.predict(feats)
    assert out["decision"] == ExpectedRDecision.BLOCK


def test_missing_columns_are_zero_filled(bundle_path: Path):
    flt = ExpectedRFilterModel(local_path=str(bundle_path), threshold=0.05)
    feats = pd.DataFrame([{"bull_score": 0.6}])  # `rr` missing
    out = flt.predict(feats)
    assert out["error"] is None
    assert out["decision"] == ExpectedRDecision.KEEP


def test_load_failure_returns_neutral_keep(tmp_path: Path):
    flt = ExpectedRFilterModel(local_path=str(tmp_path / "missing.joblib"),
                                threshold=0.05)
    out = flt.predict(pd.DataFrame([{"bull_score": 0.6}]))
    assert out["decision"] == ExpectedRDecision.KEEP
    assert out["error"] is not None
    assert out["predicted_r"] is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_filter_model.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the adapter**

```python
# xau_pro_bot/models/expected_r_filter_model.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_filter_model.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expected_r_filter_model.py xau_pro_bot/models/expected_r_filter_model.py
git commit -m "feat(path_e): ExpectedRFilterModel adapter with predicted_R threshold"
```

---

## Task 3: Make backtest engine accept ExpectedRFilterModel via duck-typed `filter_model`

**Goal:** `run_backtest(filter_model=...)` currently consumes the `TradeFilterModel` interface (returns `{good_prob, bad_prob, decision, ...}`). Path E returns `{predicted_r, decision, ...}`. Backtest only needs `decision` — verify, and (if it inspects `good_prob`) make the engine tolerate either shape.

**Files:**
- Read first: `xau_pro_bot/backtest.py` — locate every reference to the filter return dict.
- Modify (only if needed): `xau_pro_bot/backtest.py` — read keys defensively (`.get("good_prob")`).
- Create: `tests/test_expected_r_backtest_integration.py`

- [ ] **Step 1: Inspect current usage**

```bash
grep -nE "filter_model|good_prob|bad_prob|FilterDecision|ExpectedR" xau_pro_bot/backtest.py
```

Document in commit message: which keys backtest reads. If it reads only `decision`, no production code change is needed — the adapter is already drop-in. If it reads `good_prob` (e.g. for hybrid policy or telemetry), wrap with `.get()` so missing keys become `None` instead of `KeyError`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_expected_r_backtest_integration.py
"""Verify ExpectedRFilterModel works as a drop-in filter_model in run_backtest.

We don't assert exact counts — only that the backtest runs to completion and
the kept-trade count drops monotonically as the predicted_R threshold rises.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.backtest import run_backtest
from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel


class _Reg:
    """Stub regressor returning a deterministic predicted_R from `bull_score`.
    The backtest never sees this directly; only via the adapter."""

    def predict(self, X):
        # Higher bull_score -> higher predicted_R.
        return np.asarray(X["bull_score"]) - 0.5


@pytest.fixture
def bundle(tmp_path: Path):
    path = tmp_path / "e.joblib"
    joblib.dump({"model": _Reg(), "feature_cols": ["bull_score"]}, path)
    return path


def test_threshold_rise_reduces_kept(long_history, bundle):
    # `long_history` fixture provided by tests/conftest.py.
    kwargs = dict(timeout_bars=48, step=4, stream="intraday")
    kept = []
    for thr in (-1.0, 0.0, 0.5):
        flt = ExpectedRFilterModel(local_path=str(bundle), threshold=thr)
        r = run_backtest(long_history, filter_model=flt, **kwargs)
        kept.append(r.signals_generated)
    # Monotone non-increasing as threshold rises.
    assert kept[0] >= kept[1] >= kept[2]
```

- [ ] **Step 3: Run test to verify it fails (or unexpectedly passes)**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_backtest_integration.py -v
```
Expected: if `backtest.py` reads `good_prob` without `.get()`, this FAILS with `KeyError`. If it already uses `.get()`, the test passes immediately — that confirms the adapter is drop-in.

- [ ] **Step 4: If failing, apply minimal defensive read in `backtest.py`**

Change `decision = filter_result["good_prob"]` → `filter_result.get("good_prob")` style only where strictly necessary. Do NOT add Path-E-specific branches; the engine must remain filter-agnostic.

- [ ] **Step 5: Run test until green**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_backtest_integration.py -v
```
Expected: PASS.

- [ ] **Step 6: Run the full suite to catch regressions**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```
Expected: all pre-existing tests still green.

- [ ] **Step 7: Commit**

```bash
git add tests/test_expected_r_backtest_integration.py xau_pro_bot/backtest.py
git commit -m "feat(path_e): backtest accepts ExpectedRFilterModel as filter_model"
```

(If `backtest.py` did not need changes, the commit is just the new test file with message `test(path_e): verify ExpectedRFilterModel is drop-in for run_backtest`.)

---

## Task 4: Wire `--expected-r` into trainer

**Files:**
- Modify: `scripts/train_path_d_model.py` — add flag + save artifact.

- [ ] **Step 1: Write the test**

```python
# Append to tests/test_expected_r_regressor.py
def test_train_script_saves_expected_r_artifact(tmp_path, monkeypatch):
    """Smoke test: --expected-r flag produces a joblib bundle that loads."""
    import subprocess
    import sys
    import joblib

    # Re-use the existing test CSV harness only if present; otherwise skip.
    csv = tmp_path / "data.csv"
    # Create a tiny but valid M15 CSV with > min_lookback_h1 H1 bars worth.
    n = 6 * 24 * 60  # 60 days of M15
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    base = np.cumsum(np.random.default_rng(0).normal(0, 0.5, n)) + 2000
    pd.DataFrame({
        "datetime": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "Open": base, "High": base + 1.0, "Low": base - 1.0,
        "Close": base, "Volume": 100.0,
    }).to_csv(csv, index=False)

    out = tmp_path / "models"
    cmd = [
        sys.executable, "scripts/train_path_d_model.py",
        "--csv", str(csv), "--out-dir", str(out),
        "--expected-r", "--allow-degenerate",
        "--step-h1", "4",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env={"PYTHONPATH": "."})
    # Either training succeeds and the artifact exists, or the dataset was
    # too small and the script exited cleanly with an "abort" message.
    bundle_path = out / "path_e_expected_r_lgb.joblib"
    if res.returncode == 0 and bundle_path.exists():
        b = joblib.load(bundle_path)
        assert "model" in b and "feature_cols" in b
    else:
        # Acceptable: synthetic data too small to harvest. Just assert clean exit.
        assert res.returncode in (0, 1)
```

(Mark this test `@pytest.mark.slow` if the suite uses that marker; otherwise leave it as a real integration smoke that gates the trainer wiring.)

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_regressor.py::test_train_script_saves_expected_r_artifact -v
```
Expected: FAIL — `--expected-r` flag does not yet exist.

- [ ] **Step 3: Add the flag and wiring**

In `scripts/train_path_d_model.py`:

```python
# Add to argparse (next to --calibrate):
    ap.add_argument("--expected-r", action="store_true",
                    help="Also train a Path E expected_R regressor and save it.")
    ap.add_argument("--dxy-csv", default=None,
                    help="Optional DXY price CSV for macro features.")
    ap.add_argument("--us10y-csv", default=None,
                    help="Optional US10Y yield CSV for macro features.")
```

Pass DXY/US10Y to `HarvestConfig`:

```python
    cfg = HarvestConfig(
        step_h1=args.step_h1, timeout_m15=args.timeout_m15,
        include_synthetic=True, synth_stride=args.synth_stride,
        dxy_csv=args.dxy_csv, us10y_csv=args.us10y_csv,
    )
```

After `train_filter_calibrated` block, add:

```python
    met_er: dict | None = None
    if args.expected_r:
        print("Training Expected R regressor...")
        from xau_pro_bot.models.expected_r import train_expected_r_regressor
        m_er, met_er = train_expected_r_regressor(df)
        save_model(m_er, met_er["feature_cols"],
                   out_dir / "path_e_expected_r_lgb.joblib")
```

And include in metrics JSON:

```python
    if met_er is not None:
        metrics["expected_r"] = {
            k: v for k, v in met_er.items() if k != "feature_cols"
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_expected_r_regressor.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_path_d_model.py tests/test_expected_r_regressor.py
git commit -m "feat(path_e): --expected-r trainer flag saves path_e_expected_r_lgb.joblib"
```

---

## Task 5: DXY / US10Y feature loader (off by default) (TDD)

**Files:**
- Modify: `xau_pro_bot/models/path_d_harvest.py`
- Create: `tests/test_macro_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_macro_features.py
"""DXY/US10Y feature wiring. Default OFF preserves bit-identical behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xau_pro_bot.models.path_d_harvest import (
    HarvestConfig, harvest_path_d_samples,
)


def _macro_csv(tmp_path: Path, name: str, periods: int = 8000) -> Path:
    idx = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    val = np.cumsum(np.random.default_rng(1).normal(0, 0.05, periods)) + 100.0
    p = tmp_path / name
    pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "close": val,
    }).to_csv(p, index=False)
    return p


def test_no_csv_means_no_new_columns(long_history):
    df = harvest_path_d_samples(long_history, HarvestConfig(step_h1=4))
    assert not df.empty
    for col in ("dxy_ret_1h", "dxy_ret_4h", "us10y_chg_1h", "us10y_chg_4h"):
        assert col not in df.columns


def test_dxy_csv_adds_dxy_features_only(long_history, tmp_path):
    dxy = _macro_csv(tmp_path, "dxy.csv")
    df = harvest_path_d_samples(
        long_history,
        HarvestConfig(step_h1=4, dxy_csv=str(dxy)),
    )
    assert not df.empty
    assert "dxy_ret_1h" in df.columns
    assert "dxy_ret_4h" in df.columns
    # No US10Y leakage.
    assert "us10y_chg_1h" not in df.columns
    # Finite values only — zeros allowed where data is missing.
    assert df[["dxy_ret_1h", "dxy_ret_4h"]].isna().sum().sum() == 0


def test_us10y_csv_adds_us10y_features(long_history, tmp_path):
    us10y = _macro_csv(tmp_path, "us10y.csv")
    df = harvest_path_d_samples(
        long_history,
        HarvestConfig(step_h1=4, us10y_csv=str(us10y)),
    )
    assert "us10y_chg_1h" in df.columns
    assert "us10y_chg_4h" in df.columns
    assert df[["us10y_chg_1h", "us10y_chg_4h"]].isna().sum().sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_macro_features.py -v
```
Expected: 2 of 3 FAIL (the no-CSV test should already pass).

- [ ] **Step 3: Implement macro loader and feature emission**

In `xau_pro_bot/models/path_d_harvest.py`, add module-level helpers:

```python
def _load_macro_csv(path: str | None) -> pd.Series | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return df.set_index("timestamp")["close"].sort_index()


def _macro_features(series: pd.Series | None, cutoff,
                     *, prefix: str, kind: str) -> dict[str, float]:
    """kind='ret' for log/pct returns, kind='chg' for absolute change."""
    if series is None:
        return {}
    try:
        sub = series.loc[:cutoff]
        if len(sub) < 5:
            return {f"{prefix}_1h": 0.0, f"{prefix}_4h": 0.0}
        last = float(sub.iloc[-1])
        prev_1h = float(sub.iloc[-2])
        prev_4h = float(sub.iloc[-5]) if len(sub) >= 5 else prev_1h
        if kind == "ret":
            v1 = (last / prev_1h) - 1.0 if prev_1h else 0.0
            v4 = (last / prev_4h) - 1.0 if prev_4h else 0.0
        else:  # 'chg'
            v1 = last - prev_1h
            v4 = last - prev_4h
        return {f"{prefix}_1h": float(v1), f"{prefix}_4h": float(v4)}
    except Exception:
        return {f"{prefix}_1h": 0.0, f"{prefix}_4h": 0.0}
```

In `harvest_path_d_samples`, load the series once **before** the loop:

```python
    dxy_series = _load_macro_csv(cfg.dxy_csv)
    us10y_series = _load_macro_csv(cfg.us10y_csv)
```

Then inside the row-build for every emitted row, merge in:

```python
            macro = {}
            macro.update(_macro_features(dxy_series, cutoff,
                                          prefix="dxy_ret", kind="ret"))
            macro.update(_macro_features(us10y_series, cutoff,
                                          prefix="us10y_chg", kind="chg"))
            row.update(macro)
```

(Apply to all three row-emission sites: baseline, synthetic, and the M15-sub-loop if active.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_macro_features.py -v
PYTHONPATH=. .venv/bin/python -m pytest tests/test_path_d_harvest.py tests/test_path_d_harvest_step1.py -v
```
Expected: macro tests pass; existing harvest tests stay green (no-CSV default preserves behaviour).

- [ ] **Step 5: Commit**

```bash
git add tests/test_macro_features.py xau_pro_bot/models/path_d_harvest.py
git commit -m "feat(path_e): optional DXY/US10Y macro features (off by default)"
```

---

## Task 6: Path E mode in `eval_path_d.py` with predicted_R sweep (TDD)

**Files:**
- Modify: `scripts/eval_path_d.py`
- Create: `tests/test_eval_path_e.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_path_e.py
"""Path E mode runs end-to-end with a stub regressor bundle."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from scripts.eval_path_d import run_all_modes


class _Reg:
    def predict(self, X):
        return np.asarray(X["bull_score"]) - 0.4


def test_path_e_mode_in_results(long_history, tmp_path):
    bundle = tmp_path / "e.joblib"
    joblib.dump({"model": _Reg(), "feature_cols": ["bull_score"]}, bundle)

    payload = run_all_modes(
        long_history,
        path_c_local=None,
        path_d_filter=None,
        path_e=str(bundle),
    )
    res = payload["results"]
    assert "L_path_e_expected_r" in res
    # Sweep recorded with the prescribed thresholds.
    sweep = payload["expected_r_sweep"]
    assert set(sweep.keys()) == {0.00, 0.03, 0.05, 0.10, 0.15}
    # Chosen threshold satisfies the 25%-kept floor or fallback is documented.
    assert payload["chosen_expected_r_threshold"] in sweep
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_eval_path_e.py -v
```
Expected: FAIL — `run_all_modes` has no `path_e` kwarg.

- [ ] **Step 3: Add Path E to `run_all_modes`**

In `scripts/eval_path_d.py`:

```python
EXPECTED_R_THRESHOLDS = (0.00, 0.03, 0.05, 0.10, 0.15)
```

Extend signature:

```python
def run_all_modes(history, *, path_c_local: str | None,
                  path_d_filter: str | None,
                  path_d_filter_calibrated: str | None = None,
                  path_e: str | None = None,
                  val_split=(0.70, 0.85)) -> dict:
```

After the existing K-calibrated block, add (uses `min_kept` already computed):

```python
    chosen_er = None
    sweep_er: dict = {}
    if path_e and Path(path_e).exists():
        from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel
        for t in EXPECTED_R_THRESHOLDS:
            flt = ExpectedRFilterModel(local_path=path_e, threshold=float(t))
            r = run_backtest(history, filter_model=flt,
                             walk_from=t_val, walk_to=t_test, **base_kwargs)
            sweep_er[t] = {
                "pf": float(r.profit_factor),
                "expectancy": float(r.expectancy),
                "wr": float(r.win_rate),
                "kept": int(r.signals_generated),
                "blocked": int(r.blocked_signals),
                "max_dd": float(r.max_drawdown),
                "avg_rr": float(r.average_rr),
            }
        chosen_er = pick_best_threshold(sweep_er, min_kept=min_kept)
        if chosen_er is not None:
            flt = ExpectedRFilterModel(local_path=path_e,
                                        threshold=float(chosen_er))
            l = run_backtest(history, filter_model=flt,
                             walk_from=t_test, **base_kwargs)
            results["L_path_e_expected_r"] = _result_summary(l)
```

Extend return payload:

```python
    return {
        "results": results,
        "threshold_sweep": sweep,
        "chosen_threshold": chosen_threshold,
        "expected_r_sweep": sweep_er,
        "chosen_expected_r_threshold": chosen_er,
        "test_window": (str(t_test), str(h1.index[-1])),
        "val_window":  (str(t_val), str(t_test)),
    }
```

Wire CLI flag in `main()`:

```python
    ap.add_argument("--path-e", default=None,
                    help="Path E expected_R joblib bundle.")
    ...
    payload = run_all_modes(
        ...,
        path_e=args.path_e if args.path_e and Path(args.path_e).exists() else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_eval_path_e.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_eval_path_e.py scripts/eval_path_d.py
git commit -m "feat(path_e): eval Path E mode with predicted_R sweep and 25% floor"
```

---

## Task 7: Path E section in the markdown report

**Files:**
- Modify: `scripts/eval_path_d.py` (extend `write_report`).

- [ ] **Step 1: Extend `write_report`**

After the K-calibrated sweep section in `write_report`, append:

```python
    er_sweep = payload.get("expected_r_sweep") or {}
    if er_sweep:
        chosen_er = payload.get("chosen_expected_r_threshold")
        lines += [
            "",
            "## L Path E (expected_R) — Threshold Sweep (validation)",
            "",
            f"**Chosen threshold (predicted_R >):** {chosen_er}",
            "",
            "| th | kept | blocked | PF | Expectancy | WR | MaxDD | AvgRR |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, m in sorted(er_sweep.items()):
            lines.append(
                f"| {t:.2f} | {m['kept']} | {m['blocked']} | "
                f"{m['pf']:.3f} | {m['expectancy']:.3f} | {m['wr']:.3f} | "
                f"{m.get('max_dd', 0.0):.3f} | {m.get('avg_rr', 0.0):.3f} |"
            )
```

- [ ] **Step 2: Add a one-line snapshot test**

```python
# tests/test_eval_path_e.py — append
def test_report_contains_path_e_section(long_history, tmp_path):
    from scripts.eval_path_d import run_all_modes, write_report
    bundle = tmp_path / "e.joblib"
    joblib.dump({"model": _Reg(), "feature_cols": ["bull_score"]}, bundle)
    payload = run_all_modes(long_history, path_c_local=None,
                            path_d_filter=None, path_e=str(bundle))
    out = tmp_path / "report.md"
    write_report(payload, out)
    text = out.read_text()
    assert "Path E (expected_R)" in text
    assert "Chosen threshold" in text
```

- [ ] **Step 3: Run, then commit**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_eval_path_e.py -v
git add scripts/eval_path_d.py tests/test_eval_path_e.py
git commit -m "feat(path_e): report includes L Path E sweep section"
```

---

## Task 8: Full suite green check

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```
Expected: all pre-existing tests still pass; new Path E tests pass.

- [ ] **Step 2: If anything snapshots-drifted (`docs/reports/path_d_trade_outcome_results.md`), inspect the diff carefully — the previous K and F sections must be byte-identical to pre-Path-E output. Fix any unintended formatting changes.

- [ ] **Step 3: No commit needed unless a fix was applied.**

---

## Task 9: End-to-end training + eval on `data_long_m15.csv` (IN-BAND, long-running)

> **Do NOT delegate this task to a subagent — it will likely time out and you want streaming logs.** Run in the active session.

**Files (produced):**
- `models_cache/path_e_expected_r_lgb.joblib`
- `models_cache/path_d_metrics.json` (extended with `expected_r` block)
- `docs/reports/path_e_expected_r_results.md`

- [ ] **Step 1: Train Path E**

```bash
PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
  --csv ./data_long_m15.csv --out-dir ./models_cache \
  --step-h1 1 --expected-r
```

(Add `--dxy-csv ./data/dxy_1h.csv --us10y-csv ./data/us10y_1h.csv` only if those files exist locally. If not, run without and note in the verdict that the macro feature investigation is data-unavailable.)

Expected: prints `Training Expected R regressor...`, completes, writes `path_e_expected_r_lgb.joblib`.

- [ ] **Step 2: Evaluate Path E**

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_path_d.py \
  --csv ./data_long_m15.csv \
  --path-c ./models_cache/path_c_lgb.joblib \
  --path-d-filter ./models_cache/path_d_trade_outcome_lgb.joblib \
  --path-e ./models_cache/path_e_expected_r_lgb.joblib \
  --report docs/reports/path_e_expected_r_results.md
```

Expected output: the printed JSON includes `A_baseline`, `B_path_c`, `E_path_d_filter`, `H_no_weak`, `L_path_e_expected_r`; report file is written with sweep table.

- [ ] **Step 3: Record raw numbers for Task 10**

Save the printed JSON to a scratch note. You'll quote PF / Expectancy / kept-trades for `A_baseline`, `H_no_weak`, `B_path_c`, `E_path_d_filter`, `L_path_e_expected_r` in the verdict.

- [ ] **Step 4: No commit yet** — wait for Task 10 to add the verdict.

---

## Task 10: Honest verdict report

**Files:**
- Create / overwrite: `docs/reports/path_e_expected_r_results.md` (Task 9 wrote the auto-generated tables; you now prepend a verdict section).

- [ ] **Step 1: Write the verdict header**

Open `docs/reports/path_e_expected_r_results.md` and prepend (above the auto-generated tables):

```markdown
# Path E (Expected R Regressor) — Verdict

**Date:** 2026-05-14
**Branch:** feature/hugging-face-ai-layer
**Verdict:** {GO | NO-GO}

## Acceptance gates

| Gate | Threshold | Path E (test) | Pass? |
|------|-----------|---------------|-------|
| Kept trades ≥ 25% baseline | {0.25 * baseline_trades} | {kept} | {Y/N} |
| PF > B_path_c | {pf_pathc} | {pf_pathe} | {Y/N} |
| PF > H_no_weak | {pf_hnw} | {pf_pathe} | {Y/N} |
| Expectancy > H_no_weak | {ex_hnw} | {ex_pathe} | {Y/N} |
| Threshold chosen on validation only | n/a | {chosen_thr} | Y |

**Chosen predicted_R threshold:** {chosen_thr}

## Headline numbers (test slice)

| mode | trades | PF | Expectancy | MaxDD |
|------|--------|----|-----------:|------:|
| A_baseline           | ... | ... | ... | ... |
| H_no_weak            | ... | ... | ... | ... |
| B_path_c             | ... | ... | ... | ... |
| E_path_d_filter      | ... | ... | ... | ... |
| L_path_e_expected_r  | ... | ... | ... | ... |

## Feature importance (top 15)

| feature | gain |
|---------|-----:|
| ... | ... |

(Pulled from `models_cache/path_d_metrics.json` → `expected_r.feature_importance`.)

## Macro features

- DXY series available: {yes/no}
- US10Y series available: {yes/no}
- Out-of-sample PF with macro features: {value or "not run — data unavailable"}
- Out-of-sample PF without macro features: {value}
- Verdict on macro features: {improves / neutral / hurts / inconclusive}

## Decision

{
  If GO:
    "Path E meets all acceptance gates. Recommended next step: integration smoke
     test in paper-trading mode."
  If NO-GO:
    "Path E fails gate {N}: {numbers}. Recommendation: {one of:
       - 'continue Path C (currently best AI mode)';
       - 'park AI gating, ship H_no_weak as production baseline';
       - 'collect higher-frequency macro data (DXY/US10Y at M15) and retry';
       - 'redesign labels — final_R may be too noisy; try MFE/MAE-derived targets'}
     ."
}
```

- [ ] **Step 2: Update `docs/HANDOFF.md` with one paragraph**

Append a "Path E (2026-05-14)" section noting the verdict and the chosen threshold, and what to read next.

- [ ] **Step 3: Commit everything from Tasks 9 + 10**

```bash
git add models_cache/path_e_expected_r_lgb.joblib \
        models_cache/path_d_metrics.json \
        docs/reports/path_e_expected_r_results.md \
        docs/HANDOFF.md
git commit -m "report(path_e): honest GO/NO-GO verdict on expected_R regressor"
```

(Joblib artifacts: confirm with the user before committing binaries if the project's policy keeps `models_cache/` out of git — if so, drop the .joblib from `git add` and just commit docs + metrics JSON.)

---

## Self-Review Checklist (run before handing off to executor)

- [ ] Every step that touches code shows the actual code (no "implement similar to X").
- [ ] `feature_importance` key is defined in Task 1 and consumed in Task 10's report — names match.
- [ ] `ExpectedRDecision` enum name is identical across Task 2 (def) and Task 3 (use).
- [ ] `path_e` kwarg / `--path-e` CLI flag name is identical across Tasks 4, 6, 9.
- [ ] `EXPECTED_R_THRESHOLDS = (0.00, 0.03, 0.05, 0.10, 0.15)` — matches the user-specified sweep.
- [ ] 25%-kept floor is enforced via the existing `pick_best_threshold(min_kept=...)` (no duplicate logic).
- [ ] DXY/US10Y CSVs default `None` → existing test suite stays green (verified in Task 5).
- [ ] Task 10 verdict format includes the specific "kept ≥ 25%" gate the user demanded.
- [ ] No step claims success on PF alone — Task 10 gate table requires kept-floor + Expectancy + beating both B_path_c and H_no_weak.
