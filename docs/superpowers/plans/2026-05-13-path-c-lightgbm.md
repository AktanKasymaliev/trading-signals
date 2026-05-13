# Path C: Train Own LightGBM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Train a LightGBM 3-class classifier (BUY/SELL/NO_TRADE) on our own 29 deterministic features, save locally, integrate via existing AI seam, compare against deterministic baseline.

**Architecture:**
- Reuse our `build_ai_features` (29 features, `internal` feature_set).
- Labels: forward 4h return classifier — `BUY=1` if `(close[t+16] - close[t]) / close[t] > +0.30%`, `SELL=-1` if `< -0.30%`, else `NO_TRADE=0` (4h on M15 = 16 bars).
- Training script: load long M15 history → build features+labels per bar → time-based train/test split (80/20) → LightGBM with class_weight balanced → save joblib.
- Adapter: add `AI_MODEL_LOCAL_PATH` env var. When set, `HFTradingModel` skips HF download and `joblib.load`s from disk.
- Use `AI_FEATURE_SET=internal` (default) — our existing builder.
- Existing calibration handles 3-class output natively.

**Tech Stack:** Add `lightgbm>=4.0,<5.0`. Use `data_long_m15.csv` (already downloaded, 80K M15 bars, 2022-2025).

---

## File Map

Create:
- `xau_pro_bot/models/train_lightgbm.py` — feature/label generation + walk-forward training
- `scripts/train_path_c_model.py` — CLI wrapper invoking the trainer
- `tests/test_train_lightgbm.py` — unit tests for label generator
- `tests/test_local_model_path.py` — adapter local-path tests

Modify:
- `xau_pro_bot/config.py` — add `AI_MODEL_LOCAL_PATH` to `load_ai_config`
- `xau_pro_bot/models/hf_model.py` — when `local_path` set, bypass HF download
- `xau_pro_bot/signals/engine.py` — plumb `local_path` to adapter
- `requirements.txt` — add `lightgbm`
- `.env.example` — document `AI_MODEL_LOCAL_PATH`

---

## Task 1: Config + adapter local-path support

**Files:** `xau_pro_bot/config.py`, `xau_pro_bot/models/hf_model.py`, `xau_pro_bot/signals/engine.py`, `tests/test_local_model_path.py`, `.env.example`.

- [ ] **Step 1: Failing test** — `tests/test_local_model_path.py`:

```python
from __future__ import annotations
import joblib
import pandas as pd
import pytest

from xau_pro_bot.models.hf_model import HFTradingModel


class StubModel:
    classes_ = [-1, 0, 1]
    def predict_proba(self, X):
        return [[0.1, 0.2, 0.7]]


def test_local_path_skips_hf_download(tmp_path, monkeypatch):
    p = tmp_path / "model.joblib"
    joblib.dump(StubModel(), p)
    called = []
    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download",
                        lambda **kw: called.append("nope") or "/x")
    m = HFTradingModel(model_id="", model_type="sklearn", local_path=str(p))
    pred = m.predict(pd.DataFrame([{"x": 1.0}]))
    assert called == []
    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.7


def test_local_path_overrides_revision_requirement(tmp_path):
    p = tmp_path / "model.joblib"
    joblib.dump(StubModel(), p)
    m = HFTradingModel(model_id="", model_type="sklearn",
                       local_path=str(p), revision="")
    pred = m.predict(pd.DataFrame([{"x": 1.0}]))
    assert pred["direction"] == "BUY"


def test_config_includes_local_path(monkeypatch):
    from xau_pro_bot import config
    monkeypatch.setenv("AI_MODEL_LOCAL_PATH", "/tmp/m.joblib")
    cfg = config.load_ai_config()
    assert cfg["local_path"] == "/tmp/m.joblib"
```

- [ ] **Step 2: Run, expect fail**: `.venv/bin/pytest tests/test_local_model_path.py -v`

- [ ] **Step 3: Add `local_path` to `load_ai_config`** — append after `feature_set` key:
```python
        "local_path": os.getenv("AI_MODEL_LOCAL_PATH", ""),
```

- [ ] **Step 4: Add `local_path` kwarg to `HFTradingModel.__init__`** with default `""`, store on `self.local_path`.

- [ ] **Step 5: Modify `_load_sklearn`** — at the top, before the SHA check, add:
```python
        if self.local_path:
            log.info("Loading local sklearn artifact from %s", self.local_path)
            return joblib.load(self.local_path)
```
(That short-circuits before the SHA check and HF download.)

- [ ] **Step 6: Plumb in engine** — in `MasterSignalEngine.__init__`, add `local_path=str(ai_cfg["local_path"])` to the `HFTradingModel(...)` call.

- [ ] **Step 7: Update `.env.example`** — append `AI_MODEL_LOCAL_PATH=`.

- [ ] **Step 8: Run tests** — `.venv/bin/pytest tests/test_local_model_path.py tests/test_hf_model.py tests/test_ai_revision_wiring.py tests/test_ai_model_filename.py -v` — all green.

- [ ] **Step 9: Commit**: `git add -A && git commit -m "feat(models): support AI_MODEL_LOCAL_PATH for locally-trained models"`

---

## Task 2: LightGBM dependency

- [ ] **Step 1:** Append `lightgbm>=4.0,<5.0` to `requirements.txt`.
- [ ] **Step 2:** `.venv/bin/pip install lightgbm`
- [ ] **Step 3:** Verify: `.venv/bin/python -c "import lightgbm; print(lightgbm.__version__)"`.
- [ ] **Step 4:** `.venv/bin/pytest -q` — no regressions.
- [ ] **Step 5:** Commit: `git add requirements.txt && git commit -m "build: add lightgbm for path C model training"`

---

## Task 3: Trainer module (feature+label builder + train func)

**Files:** `xau_pro_bot/models/train_lightgbm.py`, `tests/test_train_lightgbm.py`.

- [ ] **Step 1: Failing tests** — `tests/test_train_lightgbm.py`:

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.train_lightgbm import (
    build_training_dataset,
    label_forward_returns,
)


def test_label_forward_returns_classifies_correctly():
    closes = pd.Series([100.0] * 17, index=pd.date_range("2026-01-01", periods=17, freq="15min", tz="UTC"))
    # Bar 0: close=100, close[16]=100 → return=0 → 0
    # Modify bar 16 for BUY case
    closes_buy = closes.copy(); closes_buy.iloc[16] = 100.5  # +0.5% > 0.3
    labels_buy = label_forward_returns(closes_buy, horizon=16, threshold=0.003)
    assert labels_buy.iloc[0] == 1

    closes_sell = closes.copy(); closes_sell.iloc[16] = 99.5  # -0.5%
    labels_sell = label_forward_returns(closes_sell, horizon=16, threshold=0.003)
    assert labels_sell.iloc[0] == -1

    labels_neutral = label_forward_returns(closes, horizon=16, threshold=0.003)
    assert labels_neutral.iloc[0] == 0


def test_label_forward_returns_nan_at_tail():
    closes = pd.Series([100.0] * 20, index=pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC"))
    labels = label_forward_returns(closes, horizon=16, threshold=0.003)
    assert pd.isna(labels.iloc[-1])  # cannot label last bar


def test_build_training_dataset_returns_X_y_with_finite_features():
    np.random.seed(0)
    n = 500
    base = 2000.0 + np.cumsum(np.random.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base+2, "Low": base-2,
        "Close": base + np.random.normal(0, 0.3, n),
        "Volume": np.random.randint(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    history = {
        "M15": m15,
        "H1": m15.resample("1h").agg(agg).dropna(),
        "H4": m15.resample("4h").agg(agg).dropna(),
        "D1": m15.resample("1D").agg(agg).dropna(),
        "W1": m15.resample("1W").agg(agg).dropna(),
    }
    X, y = build_training_dataset(history, step=8, horizon=16, threshold=0.003)
    assert len(X) == len(y)
    assert len(X) > 0
    assert np.isfinite(X.values).all()
    assert set(y.unique()).issubset({-1, 0, 1})
```

- [ ] **Step 2:** Run — expect ImportError.

- [ ] **Step 3: Implement `xau_pro_bot/models/train_lightgbm.py`:**

```python
"""LightGBM trainer for the AI signal layer.

Builds a 3-class classifier (BUY=1, NO_TRADE=0, SELL=-1) using our internal
29-feature builder and forward-return labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xau_pro_bot.models.features import build_ai_features

log = logging.getLogger(__name__)


def label_forward_returns(close: pd.Series, horizon: int = 16,
                          threshold: float = 0.003) -> pd.Series:
    """Forward-return labels: +1 if return > threshold over `horizon` bars,
    -1 if < -threshold, 0 otherwise. Last `horizon` bars get NaN."""
    fwd = close.shift(-horizon)
    ret = (fwd - close) / close
    labels = pd.Series(0, index=close.index, dtype="float64")
    labels[ret > threshold] = 1
    labels[ret < -threshold] = -1
    labels[ret.isna()] = np.nan
    return labels


def build_training_dataset(history: dict[str, pd.DataFrame], *,
                           step: int = 8, horizon: int = 16,
                           threshold: float = 0.003) -> tuple[pd.DataFrame, pd.Series]:
    """Walk M15 bars, build features at each cutoff, label by forward return.

    Returns (X DataFrame, y Series) — aligned, NaN rows dropped.
    """
    m15 = history["M15"]
    labels_full = label_forward_returns(m15["Close"], horizon=horizon, threshold=threshold)

    feature_rows: list[pd.DataFrame] = []
    label_values: list[float] = []
    indices: list[pd.Timestamp] = []

    # Require enough lookback for indicators (200+ on H1 → ~800 M15)
    start = max(800, horizon)
    for i in range(start, len(m15) - horizon, step):
        cutoff = m15.index[i]
        slice_data = {
            "M15": m15.iloc[max(0, i - 720):i + 1],
            "H1":  history["H1"].loc[:cutoff].tail(720),
            "H4":  history["H4"].loc[:cutoff].tail(720),
            "D1":  history["D1"].loc[:cutoff].tail(720),
            "W1":  history["W1"].loc[:cutoff].tail(720),
        }
        try:
            features, complete = build_ai_features(slice_data)
        except Exception:
            continue
        if not complete:
            continue
        y_val = labels_full.iloc[i]
        if pd.isna(y_val):
            continue
        feature_rows.append(features)
        label_values.append(int(y_val))
        indices.append(cutoff)

    if not feature_rows:
        return pd.DataFrame(), pd.Series(dtype="int64")

    X = pd.concat(feature_rows, ignore_index=True)
    X.index = pd.Index(indices, name="datetime")
    y = pd.Series(label_values, index=X.index, name="label", dtype="int64")
    return X, y


def train_lightgbm(X: pd.DataFrame, y: pd.Series, *,
                   test_size: float = 0.2,
                   params: dict | None = None):
    """Time-based 80/20 train/test split, train LGB classifier, return (model, metrics)."""
    import lightgbm as lgb
    from sklearn.metrics import (accuracy_score, classification_report,
                                  precision_recall_fscore_support)

    n = len(X)
    cut = int(n * (1 - test_size))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]

    default_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(30)])

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="macro",
                                                  zero_division=0)
    report = classification_report(y_te, y_pred, zero_division=0)
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "report": report,
    }
    return model, metrics


def save_model(model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
```

- [ ] **Step 4:** Run tests — `.venv/bin/pytest tests/test_train_lightgbm.py -v`.

- [ ] **Step 5:** Commit: `git add xau_pro_bot/models/train_lightgbm.py tests/test_train_lightgbm.py && git commit -m "feat(models): add LightGBM trainer with forward-return labels"`

---

## Task 4: CLI training script + actual run

**Files:** `scripts/train_path_c_model.py`.

- [ ] **Step 1:** Create `scripts/train_path_c_model.py`:

```python
"""Train Path C LightGBM model on local M15 history.

Run:
    PYTHONPATH=. .venv/bin/python scripts/train_path_c_model.py \\
        --csv ./data_long_m15.csv \\
        --out ./models_cache/path_c_lgb.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from xau_pro_bot.models.train_lightgbm import (
    build_training_dataset, save_model, train_lightgbm,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.003)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    m15 = pd.read_csv(args.csv)
    m15["datetime"] = pd.to_datetime(m15["datetime"], utc=True)
    m15 = m15.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    history = {
        "M15": m15,
        "H1": m15.resample("1h").agg(agg).dropna(),
        "H4": m15.resample("4h").agg(agg).dropna(),
        "D1": m15.resample("1D").agg(agg).dropna(),
        "W1": m15.resample("1W").agg(agg).dropna(),
    }
    print(f"Loaded M15: {len(m15)} bars ({m15.index.min()} → {m15.index.max()})")
    print(f"Building dataset (step={args.step}, horizon={args.horizon}, "
          f"threshold={args.threshold})...")
    X, y = build_training_dataset(history, step=args.step, horizon=args.horizon,
                                   threshold=args.threshold)
    print(f"Dataset: X={X.shape}, y class distribution={y.value_counts().to_dict()}")
    if len(X) < 100:
        print("Not enough samples to train.")
        return 1

    print("Training LightGBM...")
    model, metrics = train_lightgbm(X, y)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print("\nClassification report:\n" + metrics["report"])

    save_model(model, args.out)
    print(f"Model saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2:** Run training:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_path_c_model.py \
    --csv ./data_long_m15.csv \
    --out ./models_cache/path_c_lgb.joblib
```

Expected: prints class distribution, training metrics, saves joblib.

- [ ] **Step 3:** Commit script: `git add scripts/train_path_c_model.py && git commit -m "chore(scripts): add Path C training CLI"`

---

## Task 5: Compare on long history

- [ ] **Step 1:** Reuse `scripts/poc_smc_v2_compare.py` style. Create `scripts/poc_path_c_compare.py`:

```python
"""Compare baseline vs Path C LightGBM on long M15 history."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from xau_pro_bot.backtest import compare_backtests
from xau_pro_bot.models.hf_model import HFTradingModel


def main() -> int:
    m15 = pd.read_csv("./data_long_m15.csv")
    m15["datetime"] = pd.to_datetime(m15["datetime"], utc=True)
    m15 = m15.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    history = {
        "M15": m15,
        "H1": m15.resample("1h").agg(agg).dropna(),
        "H4": m15.resample("4h").agg(agg).dropna(),
        "D1": m15.resample("1D").agg(agg).dropna(),
        "W1": m15.resample("1W").agg(agg).dropna(),
    }
    ai = HFTradingModel(model_id="", model_type="sklearn",
                        local_path="./models_cache/path_c_lgb.joblib")
    c = compare_backtests(history=history, timeout_bars=48, step=12,
                          stream="intraday", ai_model=ai)
    b, a, s = c["baseline"], c["ai"], c["summary"]
    print(f"BASE: sig={b.signals_generated} W/L={b.wins}/{b.losses} "
          f"WR={b.win_rate:.1%} E={b.expectancy:.2f}R PF={b.profit_factor:.2f}")
    print(f"AI  : sig={a.signals_generated} blocked={a.blocked_signals} "
          f"W/L={a.wins}/{a.losses} WR={a.win_rate:.1%} "
          f"E={a.expectancy:.2f}R PF={a.profit_factor:.2f}")
    print(f"DELTA: trades={s['trade_delta']} WR={s['win_rate_delta']:.1%} "
          f"E={s['expectancy_delta']:.2f}R PF={s['profit_factor_delta']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2:** Run the comparison:

```bash
AI_ENABLED=true AI_FEATURE_SET=internal \
PYTHONPATH=. .venv/bin/python scripts/poc_path_c_compare.py
```

(Note: `ai_model` is passed directly via Python — env vars only needed to flag `AI_ENABLED=true` for the engine construct path in `compare_backtests`.)

- [ ] **Step 3:** Record findings in `docs/HANDOFF.md` under "Path C: LightGBM" section. Capture WR/E/PF deltas.

- [ ] **Step 4:** Commit: `git add scripts/poc_path_c_compare.py docs/HANDOFF.md && git commit -m "docs(handoff): Path C LightGBM comparison results"`

---

## Self-Review

**Spec coverage:**
- Local-path adapter: Task 1.
- LightGBM dep: Task 2.
- Trainer + labeler: Task 3.
- Training run: Task 4.
- Real comparison: Task 5.

**Out of scope:**
- Hyperparam tuning beyond defaults.
- Walk-forward CV (using simple 80/20 time split for PoC).
- Feature importance analysis (can do post-hoc if needed).

**Risk:** Trained model may not beat baseline. Forward-return labels are a weak proxy for "good trade" (no SL/TP path-dependence). Acceptable as PoC — informs whether to iterate on labels.
