# Hugging Face AI Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Hugging Face AI confirmation/filter layer to the existing deterministic XAU/USD signal bot without changing default deterministic behavior.

**Architecture:** The deterministic intraday `MasterSignalEngine` remains the source of baseline scores and levels. A new `xau_pro_bot.models` package prepares one-row deterministic feature frames, lazily loads optional Hugging Face models, converts predictions into score/block adjustments, and exposes injectable model seams for offline tests. Backtesting gets baseline-vs-AI comparison while Telegram formatting shows only a compact AI reason line when AI is enabled.

**Tech Stack:** Python 3.11, pandas, existing deterministic indicators, Hugging Face Hub, joblib/scikit-learn for trusted sklearn-style artifacts, optional lazy transformers support, pytest with mocked models and no network.

---

## File Map

Create:

- `xau_pro_bot/models/__init__.py` — package marker and public imports.
- `xau_pro_bot/models/features.py` — deterministic one-row AI feature builder.
- `xau_pro_bot/models/calibration.py` — converts model prediction dicts into score deltas and block decisions.
- `xau_pro_bot/models/hf_model.py` — lazy Hugging Face model adapter with sklearn, transformers, and custom modes.
- `tests/test_ai_features.py` — feature builder tests.
- `tests/test_ai_calibration.py` — prediction adjustment tests.
- `tests/test_hf_model.py` — mocked Hugging Face/joblib adapter tests.
- `tests/test_engine_ai.py` — engine integration tests with injected mock models.

Modify:

- `xau_pro_bot/config.py` — AI environment config helpers and constants.
- `xau_pro_bot/signals/engine.py` — optional AI adjustment in intraday deterministic scoring.
- `xau_pro_bot/formatter.py` — compact AI line in strong/weak signals.
- `xau_pro_bot/backtest.py` — `--use-ai`, `--ai-model-id`, `--ai-model-type`, `--compare-ai`, comparison metrics.
- `requirements.txt` — add `huggingface_hub`, `joblib`, `scikit-learn`.
- `.env.example` — document AI environment variables.
- `README.md` — AI usage, Railway notes, and trusted pickle/joblib warning.

Do not modify:

- `xau_pro_bot/signals/router.py`. The AI layer belongs inside `MasterSignalEngine`, so router-created intraday analysis picks it up automatically.
- Swing and scalp analyzers. They stay deterministic in this feature.

---

## Task 1: Add AI Configuration And Dependencies

**Files:**
- Modify: `xau_pro_bot/config.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Append to `tests/test_config.py`:

```python
def test_ai_defaults_disabled(monkeypatch):
    monkeypatch.delenv("AI_ENABLED", raising=False)
    monkeypatch.delenv("AI_MODEL_ID", raising=False)
    monkeypatch.delenv("AI_MODEL_TYPE", raising=False)
    monkeypatch.delenv("AI_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("AI_STRONG_CONFIDENCE", raising=False)
    monkeypatch.delenv("AI_NO_TRADE_THRESHOLD", raising=False)
    monkeypatch.delenv("AI_SCORE_BONUS", raising=False)
    monkeypatch.delenv("AI_STRONG_SCORE_BONUS", raising=False)
    monkeypatch.delenv("AI_CONFLICT_PENALTY", raising=False)
    monkeypatch.delenv("AI_CACHE_DIR", raising=False)

    cfg = config.load_ai_config()

    assert cfg["enabled"] is False
    assert cfg["model_id"] == ""
    assert cfg["model_type"] == "sklearn"
    assert cfg["min_confidence"] == 0.65
    assert cfg["strong_confidence"] == 0.75
    assert cfg["no_trade_threshold"] == 0.60
    assert cfg["score_bonus"] == 8
    assert cfg["strong_score_bonus"] == 12
    assert cfg["conflict_penalty"] == 10
    assert cfg["cache_dir"] == "./models_cache"


def test_ai_env_overrides(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("AI_MODEL_ID", "owner/xau-model")
    monkeypatch.setenv("AI_MODEL_TYPE", "transformers")
    monkeypatch.setenv("AI_MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("AI_STRONG_CONFIDENCE", "0.82")
    monkeypatch.setenv("AI_NO_TRADE_THRESHOLD", "0.64")
    monkeypatch.setenv("AI_SCORE_BONUS", "9")
    monkeypatch.setenv("AI_STRONG_SCORE_BONUS", "14")
    monkeypatch.setenv("AI_CONFLICT_PENALTY", "11")
    monkeypatch.setenv("AI_CACHE_DIR", "/tmp/hf-cache")

    cfg = config.load_ai_config()

    assert cfg["enabled"] is True
    assert cfg["model_id"] == "owner/xau-model"
    assert cfg["model_type"] == "transformers"
    assert cfg["min_confidence"] == 0.7
    assert cfg["strong_confidence"] == 0.82
    assert cfg["no_trade_threshold"] == 0.64
    assert cfg["score_bonus"] == 9
    assert cfg["strong_score_bonus"] == 14
    assert cfg["conflict_penalty"] == 11
    assert cfg["cache_dir"] == "/tmp/hf-cache"


def test_bool_parser_accepts_common_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("AI_ENABLED", value)
        assert config.load_ai_config()["enabled"] is True

    for value in ("0", "false", "FALSE", "no", "off", ""):
        monkeypatch.setenv("AI_ENABLED", value)
        assert config.load_ai_config()["enabled"] is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_config.py::test_ai_defaults_disabled tests/test_config.py::test_ai_env_overrides tests/test_config.py::test_bool_parser_accepts_common_values -v
```

Expected: fail with `AttributeError: module 'xau_pro_bot.config' has no attribute 'load_ai_config'`.

- [ ] **Step 3: Implement config helpers**

Append this block to `xau_pro_bot/config.py` after the stream constants:

```python
# ── Optional AI confirmation layer ────────────────────
AI_ENABLED = os.getenv("AI_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
AI_MODEL_ID = os.getenv("AI_MODEL_ID", "")
AI_MODEL_TYPE = os.getenv("AI_MODEL_TYPE", "sklearn")
AI_MIN_CONFIDENCE = float(os.getenv("AI_MIN_CONFIDENCE", "0.65"))
AI_STRONG_CONFIDENCE = float(os.getenv("AI_STRONG_CONFIDENCE", "0.75"))
AI_NO_TRADE_THRESHOLD = float(os.getenv("AI_NO_TRADE_THRESHOLD", "0.60"))
AI_SCORE_BONUS = int(os.getenv("AI_SCORE_BONUS", "8"))
AI_STRONG_SCORE_BONUS = int(os.getenv("AI_STRONG_SCORE_BONUS", "12"))
AI_CONFLICT_PENALTY = int(os.getenv("AI_CONFLICT_PENALTY", "10"))
AI_CACHE_DIR = os.getenv("AI_CACHE_DIR", "./models_cache")


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse common env bool values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_ai_config() -> dict[str, str | bool | float | int]:
    """Return current AI config using live env values.

    Tests mutate env after module import, so this function reads os.environ
    directly instead of returning only module-import constants.
    """
    return {
        "enabled": _env_bool("AI_ENABLED", False),
        "model_id": os.getenv("AI_MODEL_ID", ""),
        "model_type": os.getenv("AI_MODEL_TYPE", "sklearn"),
        "min_confidence": float(os.getenv("AI_MIN_CONFIDENCE", "0.65")),
        "strong_confidence": float(os.getenv("AI_STRONG_CONFIDENCE", "0.75")),
        "no_trade_threshold": float(os.getenv("AI_NO_TRADE_THRESHOLD", "0.60")),
        "score_bonus": int(os.getenv("AI_SCORE_BONUS", "8")),
        "strong_score_bonus": int(os.getenv("AI_STRONG_SCORE_BONUS", "12")),
        "conflict_penalty": int(os.getenv("AI_CONFLICT_PENALTY", "10")),
        "cache_dir": os.getenv("AI_CACHE_DIR", "./models_cache"),
    }
```

- [ ] **Step 4: Update dependency file**

Append to `requirements.txt`:

```text
huggingface_hub>=0.24,<1.0
joblib>=1.4,<2.0
scikit-learn>=1.5,<2.0
```

Do not add `torch` or `transformers` to base requirements. Transformer support imports them lazily and explains the optional dependency when missing.

- [ ] **Step 5: Update env example**

Append to `.env.example`:

```text
AI_ENABLED=false
AI_MODEL_ID=
AI_MODEL_TYPE=sklearn
AI_MIN_CONFIDENCE=0.65
AI_STRONG_CONFIDENCE=0.75
AI_NO_TRADE_THRESHOLD=0.60
AI_CACHE_DIR=./models_cache
```

- [ ] **Step 6: Install updated dependencies in the local venv**

Run:

```bash
.venv/bin/pip install -r requirements-dev.txt
```

Expected: installs `huggingface_hub`, `joblib`, and `scikit-learn` without adding `torch` or `transformers`.

- [ ] **Step 7: Run config tests**

Run:

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: all config tests pass.

- [ ] **Step 8: Commit**

```bash
git add xau_pro_bot/config.py requirements.txt .env.example tests/test_config.py
git commit -m "feat(config): add optional AI layer settings"
```

---

## Task 2: Add Deterministic AI Feature Builder

**Files:**
- Create: `xau_pro_bot/models/__init__.py`
- Create: `xau_pro_bot/models/features.py`
- Test: `tests/test_ai_features.py`

- [ ] **Step 1: Write failing feature tests**

Create `tests/test_ai_features.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.models.features import REQUIRED_AI_FEATURES, build_ai_features


def test_build_ai_features_returns_one_row(all_tfs):
    features = build_ai_features(all_tfs)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1
    assert list(features.columns) == REQUIRED_AI_FEATURES


def test_build_ai_features_does_not_mutate_input(all_tfs):
    before = {tf: df.copy(deep=True) for tf, df in all_tfs.items()}

    build_ai_features(all_tfs)

    for tf, df in all_tfs.items():
        pd.testing.assert_frame_equal(df, before[tf])


def test_build_ai_features_handles_short_dfs(short_df):
    tfs = {tf: short_df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}

    features = build_ai_features(tfs)

    assert len(features) == 1
    assert set(REQUIRED_AI_FEATURES).issubset(features.columns)
    assert np.isfinite(features["hour_utc"].iloc[0])
    assert np.isfinite(features["day_of_week"].iloc[0])


def test_build_ai_features_handles_missing_optional_indicator_columns(uptrend_df):
    tfs = {tf: uptrend_df[["Open", "High", "Low", "Close", "Volume"]].copy()
           for tf in ("W1", "D1", "H4", "H1", "M15")}

    features = build_ai_features(tfs)

    assert "rsi_h1" in features.columns
    assert "atr_h1" in features.columns
    assert features["ema8_above_ema21_h1"].iloc[0] in (-1, 0, 1)
    assert features["pd_zone_h4_encoded"].iloc[0] in (-1, 0, 1)
    assert features["wyckoff_bias_h4_encoded"].iloc[0] in (-1, 0, 1)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_ai_features.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'xau_pro_bot.models'`.

- [ ] **Step 3: Create models package marker**

Create `xau_pro_bot/models/__init__.py`:

```python
"""Optional AI model helpers for signal confirmation."""
```

- [ ] **Step 4: Implement `features.py`**

Create `xau_pro_bot/models/features.py`:

```python
"""Deterministic feature preparation for optional AI inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot.indicators import classic
from xau_pro_bot.indicators.smc import premium_discount
from xau_pro_bot.indicators.wyckoff import detect_wyckoff


REQUIRED_AI_FEATURES = [
    "close_m15",
    "close_h1",
    "close_h4",
    "close_d1",
    "return_m15_1",
    "return_m15_3",
    "return_m15_5",
    "return_h1_1",
    "return_h1_3",
    "return_h4_1",
    "atr_h1",
    "atr_m15",
    "rsi_h1",
    "rsi_m15",
    "ema8_h1",
    "ema21_h1",
    "ema50_h1",
    "ema200_h1",
    "ema8_above_ema21_h1",
    "ema21_above_ema50_h1",
    "ema50_above_ema200_h1",
    "price_above_ema50_h1",
    "price_above_ema200_h1",
    "h1_range_pct",
    "m15_range_pct",
    "pd_zone_h4_encoded",
    "wyckoff_bias_h4_encoded",
    "hour_utc",
    "day_of_week",
]


def _copy_df(tfs: dict[str, pd.DataFrame], tf: str) -> pd.DataFrame:
    df = tfs.get(tf)
    if df is None:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return df.copy(deep=True)


def _with_classic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy(deep=True)
    try:
        return classic.add_classic(df.copy(deep=True))
    except Exception:
        return df.copy(deep=True)


def _last_float(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    if df.empty or col not in df.columns:
        return default
    value = df[col].iloc[-1]
    return float(value) if not pd.isna(value) else default


def _return(df: pd.DataFrame, bars: int) -> float:
    if df.empty or "Close" not in df.columns or len(df) <= bars:
        return 0.0
    current = float(df["Close"].iloc[-1])
    previous = float(df["Close"].iloc[-1 - bars])
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return 0.0
    return (current - previous) / previous


def _above(left: float, right: float) -> int:
    if pd.isna(left) or pd.isna(right):
        return 0
    return 1 if left > right else -1


def _range_pct(df: pd.DataFrame) -> float:
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    high = _last_float(df, "High")
    low = _last_float(df, "Low")
    close = _last_float(df, "Close")
    if pd.isna(high) or pd.isna(low) or pd.isna(close) or close == 0:
        return 0.0
    return (high - low) / close


def _zone_code(zone: str | None) -> int:
    if zone == "discount":
        return 1
    if zone == "premium":
        return -1
    return 0


def _bias_code(bias: str | None) -> int:
    if bias == "bull":
        return 1
    if bias == "bear":
        return -1
    return 0


def _timestamp_features(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty or df.index.empty:
        return 0, 0
    ts = pd.Timestamp(df.index[-1])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    return int(ts.hour), int(ts.dayofweek)


def build_ai_features(tfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build exactly one deterministic feature row for AI inference."""
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
    return pd.DataFrame([[row[name] for name in REQUIRED_AI_FEATURES]],
                        columns=REQUIRED_AI_FEATURES)
```

- [ ] **Step 5: Run feature tests**

Run:

```bash
.venv/bin/pytest tests/test_ai_features.py -v
```

Expected: all feature tests pass.

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/models/__init__.py xau_pro_bot/models/features.py tests/test_ai_features.py
git commit -m "feat(models): add deterministic AI feature builder"
```

---

## Task 3: Add AI Calibration Rules

**Files:**
- Create: `xau_pro_bot/models/calibration.py`
- Test: `tests/test_ai_calibration.py`

- [ ] **Step 1: Write failing calibration tests**

Create `tests/test_ai_calibration.py`:

```python
from __future__ import annotations

from xau_pro_bot.models.calibration import ai_prediction_to_adjustment


def _pred(direction: str, confidence: float) -> dict:
    return {
        "direction": direction,
        "confidence": confidence,
        "prob_buy": None,
        "prob_sell": None,
        "prob_no_trade": None,
    }


def test_ai_agrees_buy_strong_confidence_adds_strong_bonus():
    adj = ai_prediction_to_adjustment(_pred("BUY", 0.76), "BUY")

    assert adj["score_delta_buy"] == 12
    assert adj["score_delta_sell"] == 0
    assert adj["block_signal"] is False
    assert adj["ai_direction"] == "BUY"
    assert adj["ai_confidence"] == 0.76
    assert "agrees" in adj["reason"]


def test_ai_agrees_sell_normal_confidence_adds_normal_bonus():
    adj = ai_prediction_to_adjustment(_pred("SELL", 0.66), "SELL")

    assert adj["score_delta_sell"] == 8
    assert adj["score_delta_buy"] == 0
    assert adj["block_signal"] is False


def test_ai_conflicts_with_buy_penalizes_buy():
    adj = ai_prediction_to_adjustment(_pred("SELL", 0.70), "BUY")

    assert adj["score_delta_buy"] == -10
    assert adj["score_delta_sell"] == 0
    assert adj["block_signal"] is False
    assert "conflicts" in adj["reason"]


def test_ai_no_trade_blocks_signal():
    adj = ai_prediction_to_adjustment(_pred("NO_TRADE", 0.65), "BUY")

    assert adj["block_signal"] is True
    assert adj["score_delta_buy"] == 0
    assert adj["score_delta_sell"] == 0
    assert "blocked" in adj["reason"]


def test_ai_low_confidence_does_nothing():
    adj = ai_prediction_to_adjustment(_pred("BUY", 0.40), "BUY")

    assert adj["score_delta_buy"] == 0
    assert adj["score_delta_sell"] == 0
    assert adj["block_signal"] is False
    assert "below" in adj["reason"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_ai_calibration.py -v
```

Expected: fail with `ModuleNotFoundError` or missing function.

- [ ] **Step 3: Implement `calibration.py`**

Create `xau_pro_bot/models/calibration.py`:

```python
"""Calibration rules converting AI predictions into deterministic score changes."""

from __future__ import annotations

from typing import Any

from xau_pro_bot import config


def _empty(direction: str, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "score_delta_buy": 0,
        "score_delta_sell": 0,
        "block_signal": False,
        "reason": reason,
        "ai_direction": direction,
        "ai_confidence": confidence,
    }


def ai_prediction_to_adjustment(
    prediction: dict,
    deterministic_direction: str | None,
) -> dict[str, Any]:
    """Convert a model prediction into score deltas and block decision."""
    direction = str(prediction.get("direction") or "NO_TRADE").upper()
    confidence = float(prediction.get("confidence") or 0.0)

    if direction == "NO_TRADE" and confidence >= config.AI_NO_TRADE_THRESHOLD:
        out = _empty(direction, confidence, "AI NO_TRADE confidence blocks signal")
        out["block_signal"] = True
        return out

    if confidence < config.AI_MIN_CONFIDENCE:
        return _empty(direction, confidence, "AI confidence below threshold")

    if deterministic_direction is None:
        return _empty(direction, confidence, "AI ignored because deterministic direction is missing")

    deterministic_direction = deterministic_direction.upper()
    if direction == deterministic_direction:
        bonus = (
            config.AI_STRONG_SCORE_BONUS
            if confidence >= config.AI_STRONG_CONFIDENCE
            else config.AI_SCORE_BONUS
        )
        out = _empty(direction, confidence, "AI agrees with setup")
        if direction == "BUY":
            out["score_delta_buy"] = bonus
        elif direction == "SELL":
            out["score_delta_sell"] = bonus
        return out

    if direction in {"BUY", "SELL"} and direction != deterministic_direction:
        out = _empty(direction, confidence, "AI conflicts with deterministic setup")
        if deterministic_direction == "BUY":
            out["score_delta_buy"] = -config.AI_CONFLICT_PENALTY
        elif deterministic_direction == "SELL":
            out["score_delta_sell"] = -config.AI_CONFLICT_PENALTY
        return out

    return _empty(direction, confidence, "AI prediction did not change score")
```

- [ ] **Step 4: Run calibration tests**

Run:

```bash
.venv/bin/pytest tests/test_ai_calibration.py -v
```

Expected: all calibration tests pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/models/calibration.py tests/test_ai_calibration.py
git commit -m "feat(models): add AI calibration adjustments"
```

---

## Task 4: Add Hugging Face Model Adapter

**Files:**
- Create: `xau_pro_bot/models/hf_model.py`
- Test: `tests/test_hf_model.py`

- [ ] **Step 1: Write failing Hugging Face adapter tests**

Create `tests/test_hf_model.py`:

```python
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.hf_model import HFTradingModel


class ProbModel:
    classes_ = np.array(["BUY", "SELL", "NO_TRADE"])

    def predict_proba(self, features):
        return np.array([[0.72, 0.18, 0.10]])


class NumericProbModel:
    classes_ = np.array([1, -1, 0])

    def predict_proba(self, features):
        return np.array([[0.20, 0.70, 0.10]])


class PredictOnlyModel:
    def predict(self, features):
        return np.array(["SELL"])


def test_sklearn_predict_proba_output(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: ProbModel(),
    )
    model = HFTradingModel("owner/model", "sklearn", cache_dir="/tmp/cache")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.72
    assert pred["prob_buy"] == 0.72
    assert pred["prob_sell"] == 0.18
    assert pred["prob_no_trade"] == 0.10


def test_sklearn_numeric_classes(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: NumericProbModel(),
    )
    model = HFTradingModel("owner/model", "sklearn")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "SELL"
    assert pred["confidence"] == 0.70


def test_predict_only_model_uses_default_confidence(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.hf_hub_download",
        lambda **kwargs: "/tmp/model.joblib",
    )
    monkeypatch.setattr(
        "xau_pro_bot.models.hf_model.joblib.load",
        lambda path: PredictOnlyModel(),
    )
    model = HFTradingModel("owner/model", "sklearn")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "SELL"
    assert pred["confidence"] == 0.50


def test_safe_fallback_on_exception(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", boom)
    model = HFTradingModel("owner/model", "sklearn")

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert pred["confidence"] == 0.0
    assert "error" in pred


def test_no_model_loaded_at_import_time(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/model.joblib"

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    model = HFTradingModel("owner/model", "sklearn")

    assert calls == []
    model.predict(pd.DataFrame([{"x": 1.0}]))
    assert calls != []


def test_transformers_missing_dependency_message(monkeypatch):
    model = HFTradingModel("owner/model", "transformers")
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name)))

    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert pred["direction"] == "NO_TRADE"
    assert "optional transformers dependency" in pred["error"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_hf_model.py -v
```

Expected: fail with missing `xau_pro_bot.models.hf_model`.

- [ ] **Step 3: Implement `hf_model.py`**

Create `xau_pro_bot/models/hf_model.py`:

```python
"""Lazy Hugging Face trading model adapter."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

_SKLEARN_FILENAMES = (
    "model.joblib",
    "model.pkl",
    "classifier.joblib",
    "trading_model.joblib",
)


class HFTradingModel:
    """Lazy-loaded optional model adapter for AI signal confirmation."""

    def __init__(self, model_id: str, model_type: str = "sklearn",
                 cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.model_type = model_type
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def _neutral(self, error: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "direction": "NO_TRADE",
            "confidence": 0.0,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }
        if error:
            out["error"] = error
        return out

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_id:
            raise RuntimeError("AI_MODEL_ID is required when AI is enabled")
        if self.model_type == "sklearn":
            self._model = self._load_sklearn()
            return self._model
        if self.model_type == "transformers":
            self._model = self._load_transformers()
            return self._model
        if self.model_type == "custom":
            raise NotImplementedError("custom AI model type requires an injected adapter")
        raise RuntimeError(f"Unsupported AI_MODEL_TYPE: {self.model_type}")

    def _load_sklearn(self) -> Any:
        last_exc: Exception | None = None
        for filename in _SKLEARN_FILENAMES:
            try:
                path = hf_hub_download(
                    repo_id=self.model_id,
                    filename=filename,
                    cache_dir=self.cache_dir,
                )
                log.info("Loaded sklearn AI model artifact %s from %s", filename, self.model_id)
                return joblib.load(path)
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def _load_transformers(self) -> Any:
        try:
            transformers = importlib.import_module("transformers")
            importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "optional transformers dependency is missing; install torch and transformers "
                "to use AI_MODEL_TYPE=transformers"
            ) from exc
        pipeline = transformers.pipeline(
            "text-classification",
            model=self.model_id,
            tokenizer=self.model_id,
        )
        return pipeline

    def _normalize_class(self, cls: Any) -> str:
        if cls == 1 or str(cls).upper() == "BUY":
            return "BUY"
        if cls == -1 or str(cls).upper() == "SELL":
            return "SELL"
        if cls == 0 or str(cls).upper() in {"NO_TRADE", "NO TRADE", "HOLD"}:
            return "NO_TRADE"
        return str(cls).upper()

    def _predict_sklearn(self, model: Any, features: pd.DataFrame) -> dict[str, Any]:
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(features))[0]
            classes = getattr(model, "classes_", np.arange(len(probs)))
            mapped = {self._normalize_class(cls): float(prob)
                      for cls, prob in zip(classes, probs)}
            prob_buy = mapped.get("BUY")
            prob_sell = mapped.get("SELL")
            prob_no_trade = mapped.get("NO_TRADE")
            direction, confidence = max(
                (
                    ("BUY", prob_buy or 0.0),
                    ("SELL", prob_sell or 0.0),
                    ("NO_TRADE", prob_no_trade or 0.0),
                ),
                key=lambda item: item[1],
            )
            return {
                "direction": direction,
                "confidence": float(confidence),
                "prob_buy": prob_buy,
                "prob_sell": prob_sell,
                "prob_no_trade": prob_no_trade,
                "raw": mapped,
            }

        pred = model.predict(features)[0]
        return {
            "direction": self._normalize_class(pred),
            "confidence": 0.50,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
            "raw": pred,
        }

    def _predict_transformers(self, model: Any, features: pd.DataFrame) -> dict[str, Any]:
        text = features.to_json(orient="records")
        raw = model(text)
        first = raw[0] if isinstance(raw, list) and raw else raw
        label = self._normalize_class(first.get("label", "NO_TRADE"))
        score = float(first.get("score", 0.0))
        return {
            "direction": label,
            "confidence": score,
            "prob_buy": score if label == "BUY" else None,
            "prob_sell": score if label == "SELL" else None,
            "prob_no_trade": score if label == "NO_TRADE" else None,
            "raw": raw,
        }

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """Return normalized prediction dict. Failures become neutral predictions."""
        try:
            model = self._load()
            if self.model_type == "sklearn":
                return self._predict_sklearn(model, features)
            if self.model_type == "transformers":
                return self._predict_transformers(model, features)
            raise NotImplementedError("custom AI model type requires an injected adapter")
        except Exception as exc:
            log.exception("AI model prediction failed")
            return self._neutral(str(exc))
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
.venv/bin/pytest tests/test_hf_model.py -v
```

Expected: all adapter tests pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/models/hf_model.py tests/test_hf_model.py
git commit -m "feat(models): add lazy Hugging Face trading model adapter"
```

---

## Task 5: Integrate AI Into MasterSignalEngine

**Files:**
- Modify: `xau_pro_bot/signals/engine.py`
- Test: `tests/test_engine_ai.py`

- [ ] **Step 1: Write failing engine AI tests**

Create `tests/test_engine_ai.py`:

```python
from __future__ import annotations

import copy

import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


class MockAIModel:
    def __init__(self, prediction: dict) -> None:
        self.prediction = prediction
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        assert len(features) == 1
        return self.prediction


def _prediction(direction: str, confidence: float) -> dict:
    return {
        "direction": direction,
        "confidence": confidence,
        "prob_buy": None,
        "prob_sell": None,
        "prob_no_trade": None,
    }


def test_ai_disabled_keeps_baseline_scores_and_adds_disabled_fields(all_tfs):
    baseline = MasterSignalEngine(ai_enabled=False).analyze(all_tfs)
    result = MasterSignalEngine(ai_enabled=False).analyze(all_tfs)

    assert result["direction"] == baseline["direction"]
    assert result["tier"] == baseline["tier"]
    assert result["score"] == baseline["score"]
    assert result["ai_enabled"] is False
    assert result["ai_direction"] is None
    assert result["ai_confidence"] is None
    assert result["ai_reason"] is None
    assert result["ai_blocked"] is False
    assert result["ai_score_delta_buy"] == 0
    assert result["ai_score_delta_sell"] == 0


def test_ai_buy_high_confidence_increases_buy_score(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    baseline = MasterSignalEngine(ai_enabled=False).analyze(all_tfs)
    ai = MockAIModel(_prediction("BUY", 0.76))

    result = MasterSignalEngine(ai_enabled=True, ai_model=ai).analyze(all_tfs)

    if baseline["direction"] == "BUY":
        assert result["score"] >= baseline["score"] + 12
    assert result["ai_enabled"] is True
    assert result["ai_direction"] == "BUY"
    assert result["ai_confidence"] == 0.76
    assert ai.calls == 1


def test_ai_no_trade_high_confidence_blocks_signal(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    ai = MockAIModel(_prediction("NO_TRADE", 0.65))

    result = MasterSignalEngine(ai_enabled=True, ai_model=ai).analyze(all_tfs)

    assert result["tier"] == "NO_SIGNAL"
    assert result["ai_blocked"] is True
    assert result["ai_reason"] is not None
    assert "blocks" in result["ai_reason"]


def test_ai_conflict_penalizes_deterministic_direction(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    baseline = MasterSignalEngine(ai_enabled=False).analyze(all_tfs)
    conflict_direction = "SELL" if baseline["direction"] == "BUY" else "BUY"
    ai = MockAIModel(_prediction(conflict_direction, 0.70))

    result = MasterSignalEngine(ai_enabled=True, ai_model=ai).analyze(all_tfs)

    assert result["score"] <= baseline["score"]
    assert result["ai_enabled"] is True
    assert result["ai_direction"] == conflict_direction
    assert result["ai_reason"] is not None
    assert "conflicts" in result["ai_reason"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_engine_ai.py -v
```

Expected: fail with `TypeError: MasterSignalEngine() takes no arguments`.

- [ ] **Step 3: Add imports and constructor to engine**

Modify the imports at the top of `xau_pro_bot/signals/engine.py`:

```python
from xau_pro_bot.models.calibration import ai_prediction_to_adjustment
from xau_pro_bot.models.features import build_ai_features
from xau_pro_bot.models.hf_model import HFTradingModel
```

Add this constructor inside `class MasterSignalEngine` before `_tier`:

```python
    def __init__(self, ai_enabled: bool | None = None,
                 ai_model: Any | None = None) -> None:
        ai_cfg = config.load_ai_config()
        self.ai_enabled = bool(ai_cfg["enabled"] if ai_enabled is None else ai_enabled)
        self.ai_model = ai_model
        if self.ai_enabled and self.ai_model is None:
            self.ai_model = HFTradingModel(
                model_id=str(ai_cfg["model_id"]),
                model_type=str(ai_cfg["model_type"]),
                cache_dir=str(ai_cfg["cache_dir"]),
            )
```

- [ ] **Step 4: Add helper methods to engine**

Add these methods inside `MasterSignalEngine` after `_macro_penalty`:

```python
    def _disabled_ai_fields(self) -> dict[str, Any]:
        return {
            "ai_enabled": False,
            "ai_direction": None,
            "ai_confidence": None,
            "ai_reason": None,
            "ai_blocked": False,
            "ai_score_delta_buy": 0,
            "ai_score_delta_sell": 0,
        }

    def _run_ai_adjustment(self, original_data: dict[str, pd.DataFrame],
                           deterministic_direction: str) -> dict[str, Any]:
        if not self.ai_enabled or self.ai_model is None:
            return self._disabled_ai_fields()
        features = build_ai_features(original_data)
        prediction = self.ai_model.predict(features)
        adjustment = ai_prediction_to_adjustment(
            prediction,
            deterministic_direction=deterministic_direction,
        )
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

- [ ] **Step 5: Apply AI adjustment inside `analyze`**

In `MasterSignalEngine.analyze`, after macro penalty is applied and before `final_score = max(bull_score, bear_score)`, insert:

```python
        ai_fields = self._run_ai_adjustment(data, direction)
        bull_score += ai_fields["ai_score_delta_buy"]
        bear_score += ai_fields["ai_score_delta_sell"]
        if ai_fields["ai_reason"]:
            reasons["ai"] = [ai_fields["ai_reason"]]
```

Then replace:

```python
        final_score = max(bull_score, bear_score)
        tier = self._tier(final_score)
```

with:

```python
        final_score = max(bull_score, bear_score)
        tier = "NO_SIGNAL" if ai_fields["ai_blocked"] else self._tier(final_score)
```

In the `NO_SIGNAL` return dict, add:

```python
                **ai_fields,
```

In the signal return dict, add:

```python
            **ai_fields,
```

The surrounding `reasons` dict must be created before this inserted AI block. The final shape should be:

```python
        reasons = {
            "macro": macro_reasons,
            "smc": smc_reasons,
            "ict": ict_reasons,
            "classic": cls_reasons,
            "penalties": [pen_reason] if pen_reason else [],
        }

        ai_fields = self._run_ai_adjustment(data, direction)
        bull_score += ai_fields["ai_score_delta_buy"]
        bear_score += ai_fields["ai_score_delta_sell"]
        if ai_fields["ai_reason"]:
            reasons["ai"] = [ai_fields["ai_reason"]]

        final_score = max(bull_score, bear_score)
        tier = "NO_SIGNAL" if ai_fields["ai_blocked"] else self._tier(final_score)
```

- [ ] **Step 6: Run engine AI tests**

Run:

```bash
.venv/bin/pytest tests/test_engine_ai.py -v
```

Expected: all engine AI tests pass.

- [ ] **Step 7: Run existing engine/router tests**

Run:

```bash
.venv/bin/pytest tests/test_engine.py tests/test_router.py tests/test_pipeline.py -v
```

Expected: all pass. Existing tests should tolerate the new AI fields.

- [ ] **Step 8: Commit**

```bash
git add xau_pro_bot/signals/engine.py tests/test_engine_ai.py
git commit -m "feat(engine): apply optional AI signal adjustments"
```

---

## Task 6: Add Compact AI Line To Formatter

**Files:**
- Modify: `xau_pro_bot/formatter.py`
- Test: `tests/test_formatter.py`

- [ ] **Step 1: Write failing formatter tests**

Append to `tests/test_formatter.py`:

```python
def test_strong_signal_shows_compact_ai_line_when_enabled():
    sig = _sig()
    sig.update({
        "ai_enabled": True,
        "ai_direction": "BUY",
        "ai_confidence": 0.72,
        "ai_reason": "AI agrees with setup",
        "ai_blocked": False,
    })

    text = format_strong_signal(sig)

    assert "AI: BUY 0.72 confidence — AI agrees with setup" in text


def test_weak_signal_shows_compact_ai_line_when_enabled():
    sig = _sig(tier="WEAK")
    sig.update({
        "ai_enabled": True,
        "ai_direction": "SELL",
        "ai_confidence": 0.66,
        "ai_reason": "AI conflicts with deterministic setup",
        "ai_blocked": False,
    })

    text = format_weak_signal(sig)

    assert "AI: SELL 0.66 confidence — AI conflicts with deterministic setup" in text
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_formatter.py::test_strong_signal_shows_compact_ai_line_when_enabled tests/test_formatter.py::test_weak_signal_shows_compact_ai_line_when_enabled -v
```

Expected: fail because the AI line is absent.

- [ ] **Step 3: Add formatter helper**

Add this helper after `_direction_header` in `xau_pro_bot/formatter.py`:

```python
def _ai_line(sig: dict) -> str | None:
    if not sig.get("ai_enabled"):
        return None
    direction = sig.get("ai_direction") or "NO_TRADE"
    confidence = sig.get("ai_confidence")
    reason = sig.get("ai_reason") or "AI checked"
    conf_text = f"{float(confidence):.2f}" if confidence is not None else "0.00"
    return f"AI: {direction} {conf_text} confidence — {reason}"
```

- [ ] **Step 4: Insert AI line into strong and weak messages**

In `format_strong_signal`, after the horizon label block and before the separator, add:

```python
    ai_line = _ai_line(sig)
    if ai_line:
        parts.append(ai_line)
```

In `format_weak_signal`, after the horizon label block and before `return "\n".join(parts)`, add:

```python
    ai_line = _ai_line(sig)
    if ai_line:
        parts.append(ai_line)
```

- [ ] **Step 5: Run formatter tests**

Run:

```bash
.venv/bin/pytest tests/test_formatter.py -v
```

Expected: all formatter tests pass.

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/formatter.py tests/test_formatter.py
git commit -m "feat(formatter): show compact AI confirmation line"
```

---

## Task 7: Add AI Backtest And Comparison Mode

**Files:**
- Modify: `xau_pro_bot/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write failing backtest tests**

Append to `tests/test_backtest.py`:

```python
class BacktestMockAI:
    def predict(self, features):
        return {
            "direction": "NO_TRADE",
            "confidence": 0.65,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }


def test_run_backtest_accepts_ai_model():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    history = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}

    result = run_backtest(
        history=history,
        timeout_bars=48,
        stream="intraday",
        use_ai=True,
        ai_model=BacktestMockAI(),
    )

    assert isinstance(result, BacktestResult)
    assert result.blocked_signals >= 0


def test_compare_backtests_returns_baseline_and_ai():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    history = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}

    comparison = compare_backtests(
        history=history,
        timeout_bars=48,
        step=24,
        stream="intraday",
        ai_model=BacktestMockAI(),
    )

    assert set(comparison) == {"baseline", "ai", "summary"}
    assert isinstance(comparison["baseline"], BacktestResult)
    assert isinstance(comparison["ai"], BacktestResult)
    assert "trade_delta" in comparison["summary"]
```

Update the import block at the top of `tests/test_backtest.py`:

```python
from xau_pro_bot.backtest import (
    load_csv_history, run_backtest, BacktestResult, compare_backtests,
)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_backtest.py::test_run_backtest_accepts_ai_model tests/test_backtest.py::test_compare_backtests_returns_baseline_and_ai -v
```

Expected: fail because `run_backtest` lacks `use_ai`/`ai_model` and `compare_backtests` is missing.

- [ ] **Step 3: Add backtest result metrics**

Modify `BacktestResult` in `xau_pro_bot/backtest.py`:

```python
@dataclass
class BacktestResult:
    signals_generated: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    blocked_signals: int = 0
    pnl_r: list[float] = field(default_factory=list)
    rr_values: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    per_tier: dict[str, dict[str, int]] = field(
        default_factory=lambda: {t: {"n": 0, "w": 0, "l": 0}
                                  for t in ("STRONG", "NORMAL", "WEAK")})

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def expectancy(self) -> float:
        return float(np.mean(self.pnl_r)) if self.pnl_r else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(x for x in self.pnl_r if x > 0)
        losses = -sum(x for x in self.pnl_r if x < 0)
        if losses > 0:
            return gains / losses
        return float("inf") if gains > 0 else 0.0

    @property
    def average_rr(self) -> float:
        return float(np.mean(self.rr_values)) if self.rr_values else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            max_dd = min(max_dd, value - peak)
        return abs(max_dd)
```

- [ ] **Step 4: Add analyzer factory and update `run_backtest` signature**

In `xau_pro_bot/backtest.py`, add imports:

```python
from typing import Any

from xau_pro_bot.models.hf_model import HFTradingModel
from xau_pro_bot.signals.engine import MasterSignalEngine
```

Add this helper before `run_backtest`:

```python
def _build_analyzer(stream: str, use_ai: bool = False,
                    ai_model: Any | None = None,
                    ai_model_id: str = "",
                    ai_model_type: str = "sklearn"):
    router = StreamRouter()
    if stream not in router.analyzers:
        raise ValueError(f"Unknown stream: {stream}")
    if stream != "intraday" or not use_ai:
        return router.analyzers[stream]
    model = ai_model
    if model is None and ai_model_id:
        model = HFTradingModel(ai_model_id, ai_model_type)
    return MasterSignalEngine(ai_enabled=True, ai_model=model)
```

Change `run_backtest` signature:

```python
def run_backtest(history: dict[str, pd.DataFrame],
                 timeout_bars: int = 48,
                 step: int = 4,
                 stream: str = "intraday",
                 use_ai: bool = False,
                 ai_model: Any | None = None,
                 ai_model_id: str = "",
                 ai_model_type: str = "sklearn") -> BacktestResult:
```

Replace the current router/analyzer initialization in `run_backtest` with:

```python
    analyzer = _build_analyzer(
        stream=stream,
        use_ai=use_ai,
        ai_model=ai_model,
        ai_model_id=ai_model_id,
        ai_model_type=ai_model_type,
    )
```

- [ ] **Step 5: Track blocked signals, RR, and equity**

Inside the backtest loop, replace:

```python
        if sig is None or sig["tier"] == "NO_SIGNAL" or sig.get("tp1") is None:
            continue
```

with:

```python
        if sig is None:
            continue
        if sig["tier"] == "NO_SIGNAL":
            if sig.get("ai_blocked"):
                res.blocked_signals += 1
            continue
        if sig.get("tp1") is None:
            continue
```

After `res.signals_generated += 1`, add:

```python
        if sig.get("rr") is not None:
            res.rr_values.append(float(sig["rr"]))
```

After `res.pnl_r.append(r)`, add:

```python
        previous = res.equity_curve[-1] if res.equity_curve else 0.0
        res.equity_curve.append(previous + r)
```

- [ ] **Step 6: Add comparison helper**

Add this function after `run_backtest`:

```python
def compare_backtests(history: dict[str, pd.DataFrame],
                      timeout_bars: int = 48,
                      step: int = 4,
                      stream: str = "intraday",
                      ai_model: Any | None = None,
                      ai_model_id: str = "",
                      ai_model_type: str = "sklearn") -> dict[str, Any]:
    baseline = run_backtest(
        history=history,
        timeout_bars=timeout_bars,
        step=step,
        stream=stream,
        use_ai=False,
    )
    ai = run_backtest(
        history=history,
        timeout_bars=timeout_bars,
        step=step,
        stream=stream,
        use_ai=True,
        ai_model=ai_model,
        ai_model_id=ai_model_id,
        ai_model_type=ai_model_type,
    )
    return {
        "baseline": baseline,
        "ai": ai,
        "summary": {
            "trade_delta": ai.signals_generated - baseline.signals_generated,
            "win_rate_delta": ai.win_rate - baseline.win_rate,
            "expectancy_delta": ai.expectancy - baseline.expectancy,
            "profit_factor_delta": ai.profit_factor - baseline.profit_factor,
            "blocked_signals": ai.blocked_signals,
        },
    }
```

- [ ] **Step 7: Add CLI flags and side-by-side output**

In `_cli`, add arguments:

```python
    p.add_argument("--use-ai", action="store_true")
    p.add_argument("--ai-model-id", default="")
    p.add_argument("--ai-model-type", default="sklearn",
                   choices=["sklearn", "transformers", "custom"])
    p.add_argument("--compare-ai", action="store_true")
```

Add this printer helper inside `_cli` before the stream loop:

```python
    def print_result(label: str, res: BacktestResult) -> None:
        print(f"\n=== {label} ===")
        print(f"Signals:      {res.signals_generated}")
        print(f"Blocked AI:   {res.blocked_signals}")
        print(f"Wins/Loss:    {res.wins} / {res.losses} (timeouts {res.timeouts})")
        print(f"Win rate:     {res.win_rate:.1%}")
        print(f"Expectancy:   {res.expectancy:.2f} R")
        print(f"Profit f.:    {res.profit_factor:.2f}")
        print(f"Max DD:       {res.max_drawdown:.2f} R")
        print(f"Avg RR:       {res.average_rr:.2f}")
```

Replace the body of the stream loop with:

```python
        if args.compare_ai:
            comparison = compare_backtests(
                history=history,
                timeout_bars=timeout,
                step=args.step,
                stream=s,
                ai_model_id=args.ai_model_id,
                ai_model_type=args.ai_model_type,
            )
            print(f"\n=== Stream: {s} comparison ===")
            print_result("baseline", comparison["baseline"])
            print_result("baseline + AI", comparison["ai"])
            summary = comparison["summary"]
            print("\nSummary:")
            print(f"Trade delta:       {summary['trade_delta']}")
            print(f"Win-rate delta:    {summary['win_rate_delta']:.1%}")
            print(f"Expectancy delta:  {summary['expectancy_delta']:.2f} R")
            print(f"Profit factor Δ:   {summary['profit_factor_delta']:.2f}")
            print(f"Blocked signals:   {summary['blocked_signals']}")
            export_rows.extend({"stream": s, "mode": "baseline", "R": r}
                               for r in comparison["baseline"].pnl_r)
            export_rows.extend({"stream": s, "mode": "ai", "R": r}
                               for r in comparison["ai"].pnl_r)
            continue

        res = run_backtest(
            history,
            timeout_bars=timeout,
            step=args.step,
            stream=s,
            use_ai=args.use_ai,
            ai_model_id=args.ai_model_id,
            ai_model_type=args.ai_model_type,
        )
        print_result(f"Stream: {s}", res)
        export_rows.extend({"stream": s, "mode": "ai" if args.use_ai else "baseline", "R": r}
                           for r in res.pnl_r)
```

- [ ] **Step 8: Run backtest tests**

Run:

```bash
.venv/bin/pytest tests/test_backtest.py -v
```

Expected: all backtest tests pass.

- [ ] **Step 9: Commit**

```bash
git add xau_pro_bot/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): compare deterministic and AI-enhanced signals"
```

---

## Task 8: Update README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update feature description**

Replace the opening description in `README.md`:

```markdown
Deterministic Telegram signal bot for XAU/USD using ICT, SMC, Wyckoff (soft bias), and classic TA confluence. No broker execution.

The bot can optionally run a Hugging Face AI confirmation layer after deterministic scoring. AI is disabled by default and is never required for local tests or Railway startup.
```

- [ ] **Step 2: Add AI environment variables to Quickstart note**

Replace:

```markdown
# Fill TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVE_DATA_API_KEY
```

with:

```markdown
# Fill TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVE_DATA_API_KEY
# Optional AI remains disabled unless AI_ENABLED=true
```

- [ ] **Step 3: Add AI section after Backtesting section**

Insert after the Backtesting section:

```markdown
## Optional Hugging Face AI layer

AI is an optional confirmation/filter layer for the intraday deterministic engine. It can add score bonuses, apply conflict penalties, or block a signal when the model returns `NO_TRADE` with sufficient confidence. Swing and scalp streams remain deterministic.

Environment variables:

```bash
AI_ENABLED=false
AI_MODEL_ID=
AI_MODEL_TYPE=sklearn
AI_MIN_CONFIDENCE=0.65
AI_STRONG_CONFIDENCE=0.75
AI_NO_TRADE_THRESHOLD=0.60
AI_CACHE_DIR=./models_cache
```

Supported model types:

- `sklearn`: downloads a trusted Hugging Face artifact such as `model.joblib` or `model.pkl`.
- `transformers`: lazy optional mode; install `torch` and `transformers` yourself before enabling it.
- `custom`: adapter seam for injected models in tests or future code.

Security warning: sklearn `joblib` and pickle artifacts can execute code during loading. Only use models from sources you trust and pin the exact Hugging Face repo/revision operationally.

Backtest comparison:

```bash
python -m xau_pro_bot.backtest --csv history_h1.csv --compare-ai --ai-model-id owner/xau-model
```
```

- [ ] **Step 4: Update module map**

Add to the module map:

```markdown
- `models/` — optional AI features, Hugging Face adapter, and calibration rules.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document optional Hugging Face AI layer"
```

---

## Task 9: Final Verification And Regression Fixes

**Files:**
- Modify only files required by verification failures.

- [ ] **Step 1: Run targeted AI suite**

Run:

```bash
.venv/bin/pytest tests/test_ai_features.py tests/test_ai_calibration.py tests/test_hf_model.py tests/test_engine_ai.py -v
```

Expected: all targeted AI tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
.venv/bin/pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Verify bot imports without AI env or model download**

Run:

```bash
env -u AI_ENABLED -u AI_MODEL_ID .venv/bin/python -c "from xau_pro_bot import bot; print('bot import ok')"
```

Expected output:

```text
bot import ok
```

This verifies AI disabled default does not load Hugging Face models during import.

- [ ] **Step 4: Verify backtest CLI help includes AI flags**

Run:

```bash
.venv/bin/python -m xau_pro_bot.backtest --help
```

Expected output contains:

```text
--use-ai
--ai-model-id
--ai-model-type
--compare-ai
```

- [ ] **Step 5: Inspect git diff**

Run:

```bash
git diff --stat
git status --short
```

Expected: only AI-layer files are modified, no unrelated files.

- [ ] **Step 6: Commit verification fixes if any were made**

If Step 1-5 required fixes, commit them:

```bash
git add xau_pro_bot tests README.md requirements.txt .env.example
git commit -m "fix(ai): resolve AI layer verification regressions"
```

If no files changed after verification, do not create an empty commit.

---

## Self-Review

Spec coverage:

- AI model loading from Hugging Face: Task 4.
- Deterministic AI feature preparation: Task 2.
- Integration into `MasterSignalEngine`: Task 5.
- Config flags and env defaults: Task 1.
- Backtest baseline vs AI comparison: Task 7.
- Mocked tests with no network: Tasks 2-7.
- Railway deployability and disabled default: Tasks 1, 4, 8, 9.
- Formatter compact AI line: Task 6.
- README security warning for joblib/pickle: Task 8.
- No live trading/execution logic: all tasks are analysis, formatting, docs, and tests only.

Placeholder scan:

- The plan contains no open-ended implementation steps.
- Each code-changing task includes concrete file paths, code blocks, commands, and expected outcomes.
- The only `NotImplementedError` is the required custom adapter failure path.

Type consistency:

- `build_ai_features(tfs: dict[str, pd.DataFrame]) -> pd.DataFrame` is used by engine only.
- `ai_prediction_to_adjustment(prediction, deterministic_direction)` returns the exact `ai_*` fields mapped in engine.
- `HFTradingModel.predict(features)` returns the required normalized prediction dict.
- `run_backtest(..., use_ai=False, ai_model=None, ai_model_id="", ai_model_type="sklearn")` supports tests and CLI.
