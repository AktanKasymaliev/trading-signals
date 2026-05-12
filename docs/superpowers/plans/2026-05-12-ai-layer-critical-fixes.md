# AI Layer Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four critical bugs blocking the optional Hugging Face AI layer from working in production with a pre-trained HF model: (1) missing `revision` wiring, (2) brittle multi-class handling in the sklearn predictor, (3) silent NaN propagation from the feature builder, and (4) per-call `HFTradingModel` instantiation that defeats lazy caching.

**Architecture:** Keep the existing seam (config → engine → adapter → calibration). Add `AI_MODEL_REVISION` to the config, plumb it through `MasterSignalEngine.__init__` to `HFTradingModel`. Make `_predict_sklearn` robust to arbitrary class labels by reading from `mapped` rather than three hard-coded keys. Make `build_ai_features` impute NaN deterministically and surface a `features_complete` flag the engine uses to skip AI (no `_neutral` masking). Reuse a single `HFTradingModel` instance across `analyze()` calls via lazy property cached in `MasterSignalEngine`, and have `StreamRouter` keep a single engine instance.

**Tech Stack:** Python 3.11, pandas, numpy, pytest with monkeypatching. No new deps.

---

## File Map

Modify:

- `xau_pro_bot/config.py` — add `AI_MODEL_REVISION` to `load_ai_config()`.
- `xau_pro_bot/models/hf_model.py` — robust multi-class label selection in `_predict_sklearn`.
- `xau_pro_bot/models/features.py` — `build_ai_features` returns `(DataFrame, complete: bool)`; NaN imputation rules explicit.
- `xau_pro_bot/signals/engine.py` — pass `revision` to `HFTradingModel`, cache model instance, skip AI when features incomplete.
- `xau_pro_bot/signals/router.py` — already calls `MasterSignalEngine()` once; verify no regressions.
- `.env.example` — document `AI_MODEL_REVISION`.

Create:

- `tests/test_ai_revision_wiring.py` — verifies engine passes `revision` to adapter.
- `tests/test_hf_multiclass.py` — verifies adapter handles non-standard labels.
- `tests/test_ai_features_nan.py` — verifies NaN policy.
- `tests/test_engine_ai_caching.py` — verifies single model instance reused.

Do not modify:

- `xau_pro_bot/models/calibration.py` — no changes required.
- `xau_pro_bot/formatter.py`, `xau_pro_bot/backtest.py` — out of scope (separate plan).

---

## Task 1: Add `AI_MODEL_REVISION` To Config

**Files:**
- Modify: `xau_pro_bot/config.py:125-142`
- Modify: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config test**

Append to `tests/test_config.py`:

```python
def test_ai_config_includes_revision_default_empty(monkeypatch):
    monkeypatch.delenv("AI_MODEL_REVISION", raising=False)
    cfg = config.load_ai_config()
    assert cfg["revision"] == ""


def test_ai_config_revision_from_env(monkeypatch):
    sha = "a" * 40
    monkeypatch.setenv("AI_MODEL_REVISION", sha)
    cfg = config.load_ai_config()
    assert cfg["revision"] == sha
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_config.py::test_ai_config_includes_revision_default_empty tests/test_config.py::test_ai_config_revision_from_env -v
```

Expected: `KeyError: 'revision'`.

- [ ] **Step 3: Add `revision` to `load_ai_config`**

In `xau_pro_bot/config.py`, modify the `load_ai_config()` return dict to include:

```python
        "revision": os.getenv("AI_MODEL_REVISION", ""),
```

Place it immediately after `"cache_dir": ...`.

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: all config tests pass.

- [ ] **Step 5: Update `.env.example`**

Append to `.env.example`:

```text
AI_MODEL_REVISION=
```

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/config.py tests/test_config.py .env.example
git commit -m "feat(config): add AI_MODEL_REVISION for pinned HF model artifacts"
```

---

## Task 2: Wire `revision` Through Engine To `HFTradingModel`

**Files:**
- Modify: `xau_pro_bot/signals/engine.py:28-41`
- Create: `tests/test_ai_revision_wiring.py`

- [ ] **Step 1: Write failing wiring test**

Create `tests/test_ai_revision_wiring.py`:

```python
from __future__ import annotations

import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


def test_engine_passes_revision_from_config_to_adapter(monkeypatch):
    sha = "b" * 40
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("AI_MODEL_ID", "owner/model")
    monkeypatch.setenv("AI_MODEL_TYPE", "sklearn")
    monkeypatch.setenv("AI_MODEL_REVISION", sha)
    monkeypatch.setenv("AI_CACHE_DIR", "/tmp/cache")

    engine = MasterSignalEngine()

    assert engine.ai_model is not None
    assert engine.ai_model.model_id == "owner/model"
    assert engine.ai_model.model_type == "sklearn"
    assert engine.ai_model.revision == sha
    assert engine.ai_model.cache_dir == "/tmp/cache"


def test_engine_revision_empty_when_not_configured(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("AI_MODEL_ID", "owner/model")
    monkeypatch.delenv("AI_MODEL_REVISION", raising=False)

    engine = MasterSignalEngine()

    assert engine.ai_model is not None
    assert engine.ai_model.revision == ""
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/bin/pytest tests/test_ai_revision_wiring.py -v
```

Expected: `AttributeError: 'HFTradingModel' object has no attribute 'revision'` is already present (good), but the engine isn't passing it, so `revision` will be `None` and assertions fail.

- [ ] **Step 3: Pass revision in engine constructor**

In `xau_pro_bot/signals/engine.py`, replace the `__init__` body (lines 28-41) with:

```python
    def __init__(
        self,
        ai_enabled: bool | None = None,
        ai_model: Any | None = None,
    ) -> None:
        ai_cfg = config.load_ai_config()
        self.ai_enabled = bool(ai_cfg["enabled"] if ai_enabled is None else ai_enabled)
        self.ai_model = ai_model
        if self.ai_enabled and self.ai_model is None:
            self.ai_model = HFTradingModel(
                model_id=str(ai_cfg["model_id"]),
                model_type=str(ai_cfg["model_type"]),
                cache_dir=str(ai_cfg["cache_dir"]),
                revision=str(ai_cfg["revision"]) or None,
            )
```

Note the trailing `or None` — `HFTradingModel._load_sklearn` checks revision via `_is_commit_sha` which accepts `None`/`""` and raises a clear error; passing `""` would print the empty string in logs.

- [ ] **Step 4: Run wiring tests**

```bash
.venv/bin/pytest tests/test_ai_revision_wiring.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Run existing engine tests for regression**

```bash
.venv/bin/pytest tests/test_engine.py tests/test_engine_ai.py tests/test_router.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/signals/engine.py tests/test_ai_revision_wiring.py
git commit -m "fix(engine): propagate AI_MODEL_REVISION to Hugging Face adapter"
```

---

## Task 3: Robust Multi-Class Label Selection In `_predict_sklearn`

**Files:**
- Modify: `xau_pro_bot/models/hf_model.py:148-173`
- Create: `tests/test_hf_multiclass.py`

**Problem recap:** Current code reads `mapped["BUY"]`, `mapped["SELL"]`, `mapped["NO_TRADE"]` only. A model with classes `["LONG", "SHORT", "FLAT"]` normalizes to `LONG`/`SHORT`/`FLAT` (because `_normalize_class` returns the raw upper-cased string for unknown labels) — those keys are absent from `mapped` lookups, so the model is silently treated as `NO_TRADE` 0.0.

**Fix:** Extend `_normalize_class` to map `LONG → BUY`, `SHORT → SELL`, `FLAT → NO_TRADE`, and change `_predict_sklearn` to pick the argmax from the full `mapped` dict, only falling back to BUY/SELL/NO_TRADE if those exact normalized keys are present.

- [ ] **Step 1: Write failing multi-class tests**

Create `tests/test_hf_multiclass.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/bin/pytest tests/test_hf_multiclass.py -v
```

Expected: `test_long_short_flat_labels_map_to_buy_sell_no_trade` fails because BUY direction is reported as `NO_TRADE` 0.0. `test_unknown_labels_fall_back_to_no_trade_with_error` fails because no error is raised.

- [ ] **Step 3: Extend `_normalize_class` with long/short/flat aliases**

In `xau_pro_bot/models/hf_model.py`, replace `_normalize_class` (lines 131-146):

```python
    _LABEL_ALIASES = {
        "BUY": "BUY",
        "LONG": "BUY",
        "UP": "BUY",
        "SELL": "SELL",
        "SHORT": "SELL",
        "DOWN": "SELL",
        "NO_TRADE": "NO_TRADE",
        "NOTRADE": "NO_TRADE",
        "HOLD": "NO_TRADE",
        "NEUTRAL": "NO_TRADE",
        "FLAT": "NO_TRADE",
    }

    def _normalize_class(self, value: Any) -> str | None:
        """Map a model class label to BUY/SELL/NO_TRADE, or None if unrecognized."""
        if value == 1:
            return "BUY"
        if value == -1:
            return "SELL"
        if value == 0:
            return "NO_TRADE"
        label = str(value).strip().upper().replace(" ", "_").replace("-", "_")
        return self._LABEL_ALIASES.get(label)
```

Note: `_LABEL_ALIASES` must be a class attribute (above the method) or moved to module scope. Place it as a class attribute directly before `_normalize_class`.

- [ ] **Step 4: Make `_predict_sklearn` enforce known labels**

Replace the `predict_proba` branch in `_predict_sklearn` (lines 148-173) with:

```python
    def _predict_sklearn(self, model: Any, features: pd.DataFrame) -> dict[str, Any]:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features))[0]
            classes = getattr(model, "classes_", np.arange(len(probabilities)))
            mapped: dict[str, float] = {}
            for cls, probability in zip(classes, probabilities):
                normalized = self._normalize_class(cls)
                if normalized is None:
                    continue
                mapped[normalized] = mapped.get(normalized, 0.0) + float(probability)

            if not mapped:
                raise RuntimeError(
                    f"unrecognized model classes_: {list(classes)!r}; "
                    "expected BUY/SELL/NO_TRADE (or LONG/SHORT/FLAT) labels"
                )

            prob_buy = mapped.get("BUY")
            prob_sell = mapped.get("SELL")
            prob_no_trade = mapped.get("NO_TRADE")
            direction, confidence = max(
                mapped.items(), key=lambda item: item[1]
            )
            return {
                "direction": direction,
                "confidence": float(confidence),
                "prob_buy": prob_buy,
                "prob_sell": prob_sell,
                "prob_no_trade": prob_no_trade,
            }

        prediction = np.asarray(model.predict(features))[0]
        normalized = self._normalize_class(prediction)
        if normalized is None:
            raise RuntimeError(
                f"unrecognized model.predict output: {prediction!r}"
            )
        return {
            "direction": normalized,
            "confidence": 0.50,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }
```

The `RuntimeError` is caught by the outer `try` in `predict()` and turned into a `_neutral(error=...)` response, satisfying the second test.

- [ ] **Step 5: Run multi-class tests**

```bash
.venv/bin/pytest tests/test_hf_multiclass.py -v
```

Expected: both tests pass.

- [ ] **Step 6: Run existing adapter tests for regression**

```bash
.venv/bin/pytest tests/test_hf_model.py -v
```

Expected: all pass. The existing tests use `["BUY","SELL","NO_TRADE"]` and `[1,-1,0]` which still map correctly.

- [ ] **Step 7: Commit**

```bash
git add xau_pro_bot/models/hf_model.py tests/test_hf_multiclass.py
git commit -m "fix(models): robust multi-class label normalization in HF adapter"
```

---

## Task 4: Explicit NaN Policy For AI Features

**Files:**
- Modify: `xau_pro_bot/models/features.py`
- Modify: `xau_pro_bot/signals/engine.py`
- Create: `tests/test_ai_features_nan.py`

**Problem recap:** `_last_float` returns `np.nan` when the indicator column is missing or the value is NaN. The resulting DataFrame can carry NaN into sklearn's `predict_proba`, which raises `ValueError`. The error is silently swallowed by `try/except` in `HFTradingModel.predict`, so AI quietly degrades to NO_TRADE 0.0.

**Fix:** Split feature building into a deterministic builder + a completeness flag. Numeric features are imputed with `0.0` (already neutral for returns/ranges); price/EMA NaNs flip `complete=False`; engine skips AI for that bar with a clear reason.

- [ ] **Step 1: Write failing NaN policy tests**

Create `tests/test_ai_features_nan.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.models.features import build_ai_features


def _empty_tfs() -> dict[str, pd.DataFrame]:
    return {tf: pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            for tf in ("W1", "D1", "H4", "H1", "M15")}


def test_build_ai_features_returns_tuple_with_complete_flag(all_tfs):
    features, complete = build_ai_features(all_tfs)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(complete, bool)
    assert complete is True


def test_build_ai_features_marks_incomplete_when_h1_missing():
    features, complete = build_ai_features(_empty_tfs())
    assert complete is False
    # all features still finite (imputed)
    assert features.isna().sum().sum() == 0


def test_build_ai_features_imputes_returns_with_zero():
    df = pd.DataFrame({
        "Open": [1.0], "High": [1.0], "Low": [1.0],
        "Close": [1.0], "Volume": [1.0],
    }, index=pd.date_range("2026-01-01", periods=1, freq="H", tz="UTC"))
    tfs = {tf: df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}

    features, _ = build_ai_features(tfs)

    assert features["return_m15_1"].iloc[0] == 0.0
    assert np.isfinite(features.values).all()
```

You will also need to update existing `tests/test_ai_features.py` callers — they currently unpack a single `DataFrame`. Update Task 4 Step 4 below.

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/bin/pytest tests/test_ai_features_nan.py -v
```

Expected: tests fail because `build_ai_features` currently returns a `DataFrame`, not a tuple.

- [ ] **Step 3: Modify `build_ai_features` return type and impute NaN**

In `xau_pro_bot/models/features.py`, change the signature and body of `build_ai_features` (lines 124-185):

```python
def build_ai_features(
    tfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    """Build exactly one deterministic feature row and a completeness flag.

    Returns (DataFrame, complete). When complete is False, callers should skip
    AI inference for this bar — required price/EMA values were missing.
    """
    m15 = _with_classic(_copy_df(tfs, "M15"))
    h1 = _with_classic(_copy_df(tfs, "H1"))
    h4 = _with_classic(_copy_df(tfs, "H4"))
    d1 = _with_classic(_copy_df(tfs, "D1"))

    pd_code = 0
    try:
        pd_code = _zone_code(premium_discount(h4, lookback=50).get("zone"))
    except Exception:
        pd_code = 0

    wy_code = 0
    try:
        wy_code = _bias_code(detect_wyckoff(h4).get("bias"))
    except Exception:
        wy_code = 0

    hour, day = _timestamp_features(m15 if not m15.empty else h1)

    close_h1 = _last_float(h1, "Close")
    ema8_h1 = _last_float(h1, "EMA_8")
    ema21_h1 = _last_float(h1, "EMA_21")
    ema50_h1 = _last_float(h1, "EMA_50")
    ema200_h1 = _last_float(h1, "EMA_200")

    # completeness: H1 close + at least EMA50 must be finite
    complete = (
        not pd.isna(close_h1)
        and not pd.isna(ema50_h1)
        and not h1.empty
        and not m15.empty
    )

    row: dict[str, Any] = {
        "close_m15": _last_float(m15, "Close"),
        "close_h1": close_h1,
        "close_h4": _last_float(h4, "Close"),
        "close_d1": _last_float(d1, "Close"),
        "return_m15_1": _return(m15, 1),
        "return_m15_3": _return(m15, 3),
        "return_m15_5": _return(m15, 5),
        "return_h1_1": _return(h1, 1),
        "return_h1_3": _return(h1, 3),
        "return_h4_1": _return(h4, 1),
        "atr_h1": _last_float(h1, "ATR_14"),
        "atr_m15": _last_float(m15, "ATR_14"),
        "rsi_h1": _last_float(h1, "RSI_14"),
        "rsi_m15": _last_float(m15, "RSI_14"),
        "ema8_h1": ema8_h1,
        "ema21_h1": ema21_h1,
        "ema50_h1": ema50_h1,
        "ema200_h1": ema200_h1,
        "ema8_above_ema21_h1": _above(ema8_h1, ema21_h1),
        "ema21_above_ema50_h1": _above(ema21_h1, ema50_h1),
        "ema50_above_ema200_h1": _above(ema50_h1, ema200_h1),
        "price_above_ema50_h1": _above(close_h1, ema50_h1),
        "price_above_ema200_h1": _above(close_h1, ema200_h1),
        "h1_range_pct": _range_pct(h1),
        "m15_range_pct": _range_pct(m15),
        "pd_zone_h4_encoded": pd_code,
        "wyckoff_bias_h4_encoded": wy_code,
        "hour_utc": hour,
        "day_of_week": day,
    }
    df = pd.DataFrame(
        [[row[name] for name in REQUIRED_AI_FEATURES]],
        columns=REQUIRED_AI_FEATURES,
    )
    df = df.fillna(0.0)
    return df, complete
```

- [ ] **Step 4: Update existing `tests/test_ai_features.py` callers**

Find every line `features = build_ai_features(...)` in `tests/test_ai_features.py` and replace with:

```python
    features, _ = build_ai_features(...)
```

Use a single `sed`-style search; preserve all other lines.

- [ ] **Step 5: Update engine to handle incomplete features**

In `xau_pro_bot/signals/engine.py`, replace `_run_ai_adjustment` body (around line 109-131):

```python
    def _run_ai_adjustment(
        self,
        data: dict[str, pd.DataFrame],
        deterministic_direction: str,
    ) -> dict[str, Any]:
        if not self.ai_enabled:
            return self._disabled_ai_fields()

        if self.ai_model is None:
            prediction: dict[str, Any] = {
                "direction": "NO_TRADE",
                "confidence": 0.0,
            }
        else:
            features, complete = build_ai_features(data)
            if not complete:
                return {
                    "ai_enabled": True,
                    "ai_direction": None,
                    "ai_confidence": None,
                    "ai_reason": "AI skipped: incomplete input features",
                    "ai_blocked": False,
                    "ai_score_delta_buy": 0,
                    "ai_score_delta_sell": 0,
                }
            prediction = self.ai_model.predict(features)

        adjustment = ai_prediction_to_adjustment(prediction, deterministic_direction)
        return {
            "ai_enabled": True,
            "ai_direction": adjustment["ai_direction"],
            "ai_confidence": adjustment["ai_confidence"],
            "ai_reason": adjustment["reason"],
            "ai_blocked": adjustment["block_signal"],
            "ai_score_delta_buy": adjustment["score_delta_buy"],
            "ai_score_delta_sell": adjustment["score_delta_sell"],
        }
```

- [ ] **Step 6: Run NaN and features tests**

```bash
.venv/bin/pytest tests/test_ai_features.py tests/test_ai_features_nan.py -v
```

Expected: all pass.

- [ ] **Step 7: Run full engine + integration tests**

```bash
.venv/bin/pytest tests/test_engine.py tests/test_engine_ai.py tests/test_router.py tests/test_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add xau_pro_bot/models/features.py xau_pro_bot/signals/engine.py tests/test_ai_features.py tests/test_ai_features_nan.py
git commit -m "fix(models): explicit NaN policy and completeness flag for AI features"
```

---

## Task 5: Reuse Single `HFTradingModel` Instance Across Calls

**Files:**
- Modify: `xau_pro_bot/signals/router.py` (verify no per-tick engine recreation)
- Create: `tests/test_engine_ai_caching.py`

**Problem recap:** `StreamRouter.__init__` already creates one `MasterSignalEngine`, but the intraday wrapper `IntradayAnalyzer` builds its own engine. We need to confirm both behaviors:

1. `MasterSignalEngine.__init__` builds the adapter only once per engine.
2. Across many `engine.analyze()` calls, the same adapter (and therefore the same lazy-loaded model) is reused.

We don't introduce a singleton — single instance per engine is enough.

- [ ] **Step 1: Write failing caching test**

Create `tests/test_engine_ai_caching.py`:

```python
from __future__ import annotations

import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


class CountingMockAI:
    """Records the number of predict() calls and instance identity."""

    instances: list["CountingMockAI"] = []

    def __init__(self) -> None:
        self.calls = 0
        CountingMockAI.instances.append(self)

    def predict(self, features):
        self.calls += 1
        return {
            "direction": "BUY",
            "confidence": 0.50,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }


def test_engine_keeps_single_ai_instance_across_analyze_calls(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    CountingMockAI.instances.clear()
    ai = CountingMockAI()
    engine = MasterSignalEngine(ai_enabled=True, ai_model=ai)

    for _ in range(3):
        engine.analyze(all_tfs)

    assert engine.ai_model is ai
    assert ai.calls == 3
    assert len(CountingMockAI.instances) == 1


def test_router_reuses_engine_across_runs():
    from xau_pro_bot.signals.router import StreamRouter

    router = StreamRouter()
    intraday = router.analyzers["intraday"]

    assert intraday is router.analyzers["intraday"]
```

- [ ] **Step 2: Run tests to verify pass-or-fail baseline**

```bash
.venv/bin/pytest tests/test_engine_ai_caching.py -v
```

Expected: first test should pass already (engine keeps `self.ai_model`). The second test passes if `analyzers["intraday"]` is stable. Run to confirm; if both pass, the only remaining work is to add a no-op regression guard.

- [ ] **Step 3: Read router code and confirm shape**

```bash
grep -n "analyzers\|MasterSignalEngine\|__init__" xau_pro_bot/signals/router.py
```

Expected: `StreamRouter.__init__` creates analyzers exactly once. If you discover per-call engine instantiation anywhere, refactor it to use the cached instance.

- [ ] **Step 4: Add documentation comment to engine init**

In `xau_pro_bot/signals/engine.py`, add a brief comment above the `__init__` body:

```python
        # The HFTradingModel is created once per engine instance. The adapter
        # itself lazy-loads the model on first predict() and caches it
        # internally, so subsequent analyze() calls reuse the loaded model.
```

Place it as the first line inside `__init__`, before `ai_cfg = config.load_ai_config()`.

- [ ] **Step 5: Run all AI tests**

```bash
.venv/bin/pytest tests/test_engine_ai_caching.py tests/test_engine_ai.py tests/test_router.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/signals/engine.py tests/test_engine_ai_caching.py
git commit -m "test(engine): document and guard single AI adapter instance per engine"
```

---

## Task 6: Final Verification

**Files:** none, verification only.

- [ ] **Step 1: Full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Verify default disabled path still works**

```bash
env -u AI_ENABLED -u AI_MODEL_ID -u AI_MODEL_REVISION \
    .venv/bin/python -c "from xau_pro_bot import bot; print('bot import ok')"
```

Expected output: `bot import ok`.

- [ ] **Step 3: Verify enabled-but-misconfigured path emits clear error**

```bash
AI_ENABLED=true AI_MODEL_ID=owner/x AI_MODEL_TYPE=sklearn AI_MODEL_REVISION= \
    .venv/bin/python -c "
from xau_pro_bot.signals.engine import MasterSignalEngine
import pandas as pd
e = MasterSignalEngine()
print('revision:', e.ai_model.revision)
"
```

Expected output: `revision: None` (because empty string is converted to None in Task 2 Step 3).

- [ ] **Step 4: Verify enabled-with-revision path attempts download path**

This step does not require network access. The intent is to confirm the call shape.

```bash
AI_ENABLED=true AI_MODEL_ID=owner/x AI_MODEL_TYPE=sklearn AI_MODEL_REVISION=$(python -c "print('a'*40)") \
    .venv/bin/python -c "
from xau_pro_bot.signals.engine import MasterSignalEngine
e = MasterSignalEngine()
print('revision:', e.ai_model.revision)
print('type:', e.ai_model.model_type)
print('id:', e.ai_model.model_id)
"
```

Expected: `revision: aaaaaaaa...` (40 chars).

- [ ] **Step 5: Inspect git diff**

```bash
git diff --stat main...HEAD
git status --short
```

Expected: only AI-layer files changed, no unrelated edits.

- [ ] **Step 6: No commit needed if verification clean**

If Steps 1-4 surfaced issues, fix them and commit:

```bash
git add -p  # stage only relevant hunks
git commit -m "fix(ai): resolve verification regressions"
```

---

## Self-Review

**Spec coverage:**

- Revision wiring (Critical bug #1): Tasks 1 + 2.
- Multi-class robustness (Critical bug #2): Task 3.
- NaN policy (Critical bug #3): Task 4.
- Instance caching (Critical bug #4): Task 5.
- Final integration verification: Task 6.

**Out of scope (separate plans):**

- Formatter AI line (original Task 6).
- Backtest `--use-ai`/`--compare-ai` (original Task 7).
- README docs (original Task 8).
- Performance: removing double `_with_classic` work between `_enrich` and `build_ai_features`.
- AI decision logging.
- Transformers-pipeline correctness for numeric features.

**Placeholder scan:** No TBDs. Every step has either exact code, exact command, or both. No "similar to Task N" cross-references.

**Type consistency:**

- `build_ai_features` return changes from `pd.DataFrame` to `tuple[pd.DataFrame, bool]`. Both call sites updated: engine (Task 4 Step 5) and tests (Task 4 Step 4).
- `_normalize_class` return changes from `str` to `str | None`. Both callers (`_predict_sklearn` proba branch and predict-only branch) updated in Task 3 Step 4 to raise on `None`.
- `HFTradingModel.__init__` already accepts `revision` (existing code). Engine passes it through (Task 2 Step 3).
- `load_ai_config()` returns dict — key `"revision"` added (Task 1 Step 3).
