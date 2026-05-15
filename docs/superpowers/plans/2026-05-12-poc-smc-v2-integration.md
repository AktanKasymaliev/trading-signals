# PoC: Integrate JonusNattapong SMC v2 Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the AI layer infrastructure end-to-end by integrating the pre-trained `JonusNattapong/xauusd-trading-ai-smc-v2` model (15m variant) locally as a Proof-of-Concept. The success criterion is *infrastructure correctness*, not "AI beats baseline" — we want a clean pipeline: HF download with revision pin → custom filename support → SMC-v2 feature builder → adapter prediction → calibration → engine output → `--compare-ai` runs and prints both columns.

**Architecture:**

- Extend `HFTradingModel` with optional `filename` override; downstream this comes from a new `AI_MODEL_FILENAME` env var. The default `_SKLEARN_FILENAMES` list does not include `trading_model_15m.pkl`.
- Add `AI_FEATURE_SET` env var with values `internal` (default, our existing 29 features) and `smc_v2` (the model's 23 features). Engine routes to the correct feature builder.
- Create `xau_pro_bot/models/smc_v2_features.py` that produces exactly the 23 features documented in the model README in the documented order. Some are already computable from existing indicators; MACD / Bollinger Bands / SMA / lag-features are computed inline.
- The model is **binary** (class 1 = BUY, class 0 = NOT_BUY). With our existing `_normalize_class` rule (`0 → NO_TRADE`, `1 → BUY`), `_predict_sklearn` will produce `{"BUY": p, "NO_TRADE": 1-p}` and argmax-pick direction. Calibration treats `NO_TRADE` confidence ≥ `AI_NO_TRADE_THRESHOLD` as a block; the model effectively acts as a BUY-only filter. No SELL signals are ever boosted by this model — that's expected for a binary classifier.
- A one-shot smoke script downloads the model, builds a synthetic 15m DataFrame, runs `engine.analyze()` and prints the AI-augmented signal. After smoke passes, manually download the model's own training data (`data_xauusd_v3/15m_data_v3.csv` from the HF repo), resample to H1, and run `python -m xau_pro_bot.backtest --compare-ai` for the head-to-head comparison.

**Tech Stack:** Existing — `huggingface_hub`, `joblib`, `scikit-learn`, `pandas`. **One new dep:** `xgboost` (the pkl is an `XGBClassifier`; loading it via joblib requires xgboost importable).

---

## File Map

Create:

- `xau_pro_bot/models/smc_v2_features.py` — 23-feature builder for SMC v2 model
- `tests/test_smc_v2_features.py` — feature builder tests
- `tests/test_ai_model_filename.py` — adapter filename-override tests
- `tests/test_ai_feature_set.py` — engine feature-set selection tests
- `scripts/poc_smc_v2_smoke.py` — one-shot integration smoke check

Modify:

- `xau_pro_bot/config.py` — add `AI_MODEL_FILENAME` and `AI_FEATURE_SET` to `load_ai_config`
- `xau_pro_bot/models/hf_model.py` — `_load_sklearn` honors a single override filename
- `xau_pro_bot/signals/engine.py` — pick feature builder based on `feature_set`
- `.env.example` — document the two new env vars
- `requirements.txt` — add `xgboost>=2.0,<3.0`

Do not modify:

- `xau_pro_bot/models/calibration.py` — works as-is for binary {BUY, NO_TRADE} models.
- `xau_pro_bot/models/features.py` — existing builder remains the default (`internal`).

---

## Task 1: Add `AI_MODEL_FILENAME` And `AI_FEATURE_SET` To Config

**Files:**
- Modify: `xau_pro_bot/config.py:load_ai_config`
- Modify: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Append to `tests/test_config.py`:

```python
def test_ai_config_includes_model_filename_default_empty(monkeypatch):
    monkeypatch.delenv("AI_MODEL_FILENAME", raising=False)
    cfg = config.load_ai_config()
    assert cfg["model_filename"] == ""


def test_ai_config_model_filename_from_env(monkeypatch):
    monkeypatch.setenv("AI_MODEL_FILENAME", "trading_model_15m.pkl")
    cfg = config.load_ai_config()
    assert cfg["model_filename"] == "trading_model_15m.pkl"


def test_ai_config_feature_set_default_internal(monkeypatch):
    monkeypatch.delenv("AI_FEATURE_SET", raising=False)
    cfg = config.load_ai_config()
    assert cfg["feature_set"] == "internal"


def test_ai_config_feature_set_smc_v2(monkeypatch):
    monkeypatch.setenv("AI_FEATURE_SET", "smc_v2")
    cfg = config.load_ai_config()
    assert cfg["feature_set"] == "smc_v2"
```

- [ ] **Step 2: Run — expect KeyError**

```bash
.venv/bin/pytest tests/test_config.py -v
```

- [ ] **Step 3: Add to `load_ai_config`**

In `xau_pro_bot/config.py`, modify the `load_ai_config()` return dict, immediately after `"revision": os.getenv("AI_MODEL_REVISION", "")` add:

```python
        "model_filename": os.getenv("AI_MODEL_FILENAME", ""),
        "feature_set": os.getenv("AI_FEATURE_SET", "internal"),
```

- [ ] **Step 4: Run — expect all pass**

```bash
.venv/bin/pytest tests/test_config.py -v
```

- [ ] **Step 5: Update `.env.example`**

Append:

```text
AI_MODEL_FILENAME=
AI_FEATURE_SET=internal
```

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/config.py tests/test_config.py .env.example
git commit -m "feat(config): add AI_MODEL_FILENAME and AI_FEATURE_SET options"
```

---

## Task 2: Adapter Filename Override

**Files:**
- Modify: `xau_pro_bot/models/hf_model.py`
- Create: `tests/test_ai_model_filename.py`

- [ ] **Step 1: Write failing adapter test**

Create `tests/test_ai_model_filename.py`:

```python
from __future__ import annotations

import pandas as pd

from xau_pro_bot.models.hf_model import HFTradingModel


def test_load_sklearn_uses_custom_filename(monkeypatch):
    downloaded: list[str] = []

    def fake_download(*, repo_id, filename, cache_dir, revision):
        downloaded.append(filename)
        return "/tmp/dummy.pkl"

    class DummyModel:
        classes_ = [0, 1]

        def predict_proba(self, features):
            return [[0.4, 0.6]]

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr("xau_pro_bot.models.hf_model.joblib.load", lambda path: DummyModel())

    sha = "a" * 40
    model = HFTradingModel(
        model_id="owner/m",
        model_type="sklearn",
        revision=sha,
        filename="trading_model_15m.pkl",
    )
    pred = model.predict(pd.DataFrame([{"x": 1.0}]))

    assert downloaded == ["trading_model_15m.pkl"]
    assert pred["direction"] == "BUY"
    assert pred["confidence"] == 0.6


def test_load_sklearn_falls_back_to_defaults_when_filename_empty(monkeypatch):
    downloaded: list[str] = []

    def fake_download(*, repo_id, filename, cache_dir, revision):
        downloaded.append(filename)
        if filename == "model.joblib":
            return "/tmp/dummy.joblib"
        raise FileNotFoundError(filename)

    class DummyModel:
        classes_ = [1]

        def predict(self, features):
            return [1]

    monkeypatch.setattr("xau_pro_bot.models.hf_model.hf_hub_download", fake_download)
    monkeypatch.setattr("xau_pro_bot.models.hf_model.joblib.load", lambda path: DummyModel())

    sha = "b" * 40
    model = HFTradingModel(
        model_id="owner/m",
        model_type="sklearn",
        revision=sha,
        filename="",
    )
    model.predict(pd.DataFrame([{"x": 1.0}]))

    assert downloaded[0] == "model.joblib"
```

- [ ] **Step 2: Run — expect failure (HFTradingModel has no `filename` parameter)**

```bash
.venv/bin/pytest tests/test_ai_model_filename.py -v
```

- [ ] **Step 3: Add `filename` to `HFTradingModel.__init__`**

In `xau_pro_bot/models/hf_model.py`, modify `__init__`:

```python
    def __init__(
        self,
        model_id: str,
        model_type: str = "sklearn",
        cache_dir: str | None = None,
        revision: str | None = None,
        filename: str = "",
    ) -> None:
        self.model_id = model_id
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.revision = revision
        self.filename = filename
        self._model: Any | None = None
```

- [ ] **Step 4: Update `_load_sklearn` to honor `filename`**

Replace the body of `_load_sklearn` (around lines 73-112):

```python
    def _load_sklearn(self) -> Any:
        if not _is_commit_sha(self.revision):
            raise RuntimeError(
                "sklearn/joblib models require a pinned 40-character commit SHA "
                "for Hugging Face artifacts"
            )

        if self.filename:
            candidates = (self.filename,)
        else:
            candidates = _SKLEARN_FILENAMES

        last_error: Exception | None = None
        for filename in candidates:
            try:
                log.info(
                    "Attempting to download Hugging Face sklearn artifact %s from %s",
                    filename,
                    self.model_id,
                )
                path = hf_hub_download(
                    repo_id=self.model_id,
                    filename=filename,
                    cache_dir=self.cache_dir,
                    revision=self.revision,
                )
                model = joblib.load(path)
                log.info(
                    "Loaded Hugging Face sklearn model %s from artifact %s",
                    self.model_id,
                    filename,
                )
                return model
            except Exception as exc:
                last_error = exc
                log.debug(
                    "Unable to load Hugging Face sklearn artifact %s from %s",
                    filename,
                    self.model_id,
                    exc_info=True,
                )

        if last_error is None:
            raise RuntimeError("no sklearn artifact filenames were configured")
        raise last_error
```

- [ ] **Step 5: Plumb `filename` through engine**

In `xau_pro_bot/signals/engine.py`, modify the `HFTradingModel(...)` construction in `__init__` to pass `filename`:

```python
        if self.ai_enabled and self.ai_model is None:
            self.ai_model = HFTradingModel(
                model_id=str(ai_cfg["model_id"]),
                model_type=str(ai_cfg["model_type"]),
                cache_dir=str(ai_cfg["cache_dir"]),
                revision=str(ai_cfg["revision"]),
                filename=str(ai_cfg["model_filename"]),
            )
```

- [ ] **Step 6: Run new + existing adapter tests**

```bash
.venv/bin/pytest tests/test_ai_model_filename.py tests/test_hf_model.py tests/test_hf_multiclass.py tests/test_ai_revision_wiring.py -v
```

All should pass.

- [ ] **Step 7: Commit**

```bash
git add xau_pro_bot/models/hf_model.py xau_pro_bot/signals/engine.py tests/test_ai_model_filename.py
git commit -m "feat(models): support custom HF artifact filename"
```

---

## Task 3: Discover SMC v2 Feature Encoding

This is a read-only investigation task to avoid guessing.

- [ ] **Step 1: Download the model's feature dataset for inspection**

Run:

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='JonusNattapong/xauusd-trading-ai-smc-v2',
    filename='smc_features_dataset.csv',
    revision='d1ee87d058bf714af1b6f4b3979646dd0024b726',
    cache_dir='./models_cache',
)
print(path)
"
```

- [ ] **Step 2: Inspect column names + unique values for the encoded columns**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('<path-from-step-1>')
print('Columns:', list(df.columns))
print()
for col in ['FVG_Type_Encoded', 'OB_Type_Encoded', 'Recovery_Type_Encoded']:
    if col in df.columns:
        print(col, '→ unique:', sorted(df[col].dropna().unique().tolist()))
print()
print('FVG_Size summary:', df['FVG_Size'].describe() if 'FVG_Size' in df.columns else 'missing')
"
```

- [ ] **Step 3: Write findings into a comment block at the top of `xau_pro_bot/models/smc_v2_features.py` in Task 4**

No commit at this step — purely investigation. The findings inform encoding choices in Task 4.

---

## Task 4: SMC v2 Feature Builder

**Files:**
- Create: `xau_pro_bot/models/smc_v2_features.py`
- Test: `tests/test_smc_v2_features.py`

**Feature spec — verified against `model.feature_names_in_` (Task 3):**

```
Close, High, Low, Open,
SMA_20, SMA_50, EMA_12, EMA_26,
RSI, MACD, MACD_signal, MACD_hist,
BB_upper, BB_middle, BB_lower,
FVG_Size, FVG_Type, OB_Type,
Close_lag1, Close_lag2, Close_lag3
```

**21 features (not 23).** No Volume. No Recovery_Type. Column names are `FVG_Type`/`OB_Type` without `_Encoded` suffix (despite being numeric).

Encoding from Task 3 inspection of X_features.csv (sklearn LabelEncoder alphabetical):
- FVG_Type, OB_Type: `bearish=0, bullish=1, none=2`

All features are computed from the **M15 timeframe**. Indicators use standard defaults (RSI 14, MACD 12/26/9, BB 20/2, SMA 20/50, EMA 12/26). Raw prices work — trees are scale-robust.

- [ ] **Step 1: Write failing tests**

Create `tests/test_smc_v2_features.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.smc_v2_features import (
    REQUIRED_SMC_V2_FEATURES,
    build_smc_v2_features,
)


@pytest.fixture
def long_m15_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    base = 2000.0 + np.cumsum(np.random.normal(0, 1.0, n))
    return pd.DataFrame({
        "Open": base,
        "High": base + 2,
        "Low": base - 2,
        "Close": base + np.random.normal(0, 0.5, n),
        "Volume": np.random.randint(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))


def test_build_returns_dataframe_with_23_features(long_m15_df):
    df, complete = build_smc_v2_features({"M15": long_m15_df})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == REQUIRED_SMC_V2_FEATURES
    assert len(REQUIRED_SMC_V2_FEATURES) == 23
    assert complete is True


def test_required_features_order_matches_model_readme():
    expected = [
        "Close", "High", "Low", "Open", "Volume",
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_middle", "BB_lower",
        "FVG_Size", "FVG_Type_Encoded",
        "OB_Type_Encoded", "Recovery_Type_Encoded",
        "Close_lag1", "Close_lag2", "Close_lag3",
    ]
    assert REQUIRED_SMC_V2_FEATURES == expected


def test_lag_features_match_previous_closes(long_m15_df):
    df, _ = build_smc_v2_features({"M15": long_m15_df})
    closes = long_m15_df["Close"].iloc[-4:].tolist()
    assert df["Close_lag1"].iloc[0] == pytest.approx(closes[-2])
    assert df["Close_lag2"].iloc[0] == pytest.approx(closes[-3])
    assert df["Close_lag3"].iloc[0] == pytest.approx(closes[-4])


def test_incomplete_when_too_few_bars():
    short = pd.DataFrame({
        "Open": [1.0] * 20, "High": [1.0] * 20, "Low": [1.0] * 20,
        "Close": [1.0] * 20, "Volume": [1.0] * 20,
    }, index=pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC"))
    df, complete = build_smc_v2_features({"M15": short})
    assert complete is False
    assert df.isna().sum().sum() == 0  # imputed


def test_no_nan_in_output(long_m15_df):
    df, _ = build_smc_v2_features({"M15": long_m15_df})
    assert np.isfinite(df.values).all()
```

- [ ] **Step 2: Run — expect import failure**

```bash
.venv/bin/pytest tests/test_smc_v2_features.py -v
```

- [ ] **Step 3: Implement `smc_v2_features.py`**

Create `xau_pro_bot/models/smc_v2_features.py`:

```python
"""Feature builder matching JonusNattapong/xauusd-trading-ai-smc-v2 contract.

Model expects 23 features in this exact order computed from M15 bars.
See the model README for the complete feature spec.

Encoding choices (FVG_Type_Encoded, OB_Type_Encoded, Recovery_Type_Encoded)
follow the values observed in the model's `smc_features_dataset.csv`. If those
values differ, update the helpers below. Defaults assume {-1, 0, 1} mapping
for bear / neutral / bull.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SMC_V2_FEATURES = [
    "Close", "High", "Low", "Open", "Volume",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_middle", "BB_lower",
    "FVG_Size", "FVG_Type_Encoded",
    "OB_Type_Encoded", "Recovery_Type_Encoded",
    "Close_lag1", "Close_lag2", "Close_lag3",
]

_MIN_BARS_FOR_COMPLETE = 60  # need >= 50 for SMA_50 + lookback for SMC


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    middle = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = middle + mult * std
    lower = middle - mult * std
    return upper, middle, lower


def _last_fvg(df: pd.DataFrame, lookback: int = 50) -> tuple[float, int]:
    """Return (FVG_Size, FVG_Type_Encoded) for the most recent FVG.

    FVG (Fair Value Gap, 3-candle imbalance):
      Bull FVG at bar i: low[i] > high[i-2]  → gap_size = low[i] - high[i-2]
      Bear FVG at bar i: high[i] < low[i-2]  → gap_size = low[i-2] - high[i]
    Returns the size and {-1, 0, +1} encoding.
    """
    if len(df) < 3:
        return 0.0, 0
    window = df.iloc[-lookback:] if len(df) > lookback else df
    high = window["High"].values
    low = window["Low"].values
    for i in range(len(window) - 1, 1, -1):
        if low[i] > high[i - 2]:
            return float(low[i] - high[i - 2]), 1
        if high[i] < low[i - 2]:
            return float(low[i - 2] - high[i]), -1
    return 0.0, 0


def _last_ob_encoded(df: pd.DataFrame, lookback: int = 50) -> int:
    """{-1, 0, +1}: most recent Order Block direction (bearish / none / bullish).

    Heuristic: last candle in the lookback window whose body is opposite to
    the next 3-bar move >= 1.5 * body size is treated as an OB.
    """
    if len(df) < 5:
        return 0
    window = df.iloc[-lookback:] if len(df) > lookback else df
    o = window["Open"].values
    c = window["Close"].values
    for i in range(len(window) - 4, -1, -1):
        body = abs(c[i] - o[i])
        if body <= 0:
            continue
        move = c[i + 3] - c[i]
        if c[i] < o[i] and move > 1.5 * body:
            return 1
        if c[i] > o[i] and move < -1.5 * body:
            return -1
    return 0


def _recovery_encoded(df: pd.DataFrame, lookback: int = 20) -> int:
    """Recovery / pullback in trending move.
    +1 if recent close above 50% retracement of last leg up
    -1 if recent close below 50% retracement of last leg down
    0 otherwise (or insufficient data).
    """
    if len(df) < lookback:
        return 0
    window = df["Close"].iloc[-lookback:]
    high = window.max()
    low = window.min()
    last = window.iloc[-1]
    mid = (high + low) / 2.0
    if window.iloc[0] < window.iloc[-1] and last > mid:
        return 1
    if window.iloc[0] > window.iloc[-1] and last < mid:
        return -1
    return 0


def build_smc_v2_features(
    tfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    """Build a 1-row DataFrame of the 23 SMC v2 features from M15 bars."""
    m15 = tfs.get("M15")
    if m15 is None or m15.empty:
        empty = pd.DataFrame(
            [[0.0] * len(REQUIRED_SMC_V2_FEATURES)],
            columns=REQUIRED_SMC_V2_FEATURES,
        )
        return empty, False

    df = m15.copy(deep=True)
    complete = len(df) >= _MIN_BARS_FOR_COMPLETE

    close = df["Close"]
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    rsi = _rsi(close, 14)
    macd, macd_signal, macd_hist = _macd(close)
    bb_upper, bb_middle, bb_lower = _bollinger(close)
    fvg_size, fvg_type = _last_fvg(df)
    ob_encoded = _last_ob_encoded(df)
    recovery_encoded = _recovery_encoded(df)

    last_close = float(close.iloc[-1])
    lag1 = float(close.iloc[-2]) if len(close) >= 2 else last_close
    lag2 = float(close.iloc[-3]) if len(close) >= 3 else last_close
    lag3 = float(close.iloc[-4]) if len(close) >= 4 else last_close

    row: dict[str, Any] = {
        "Close": last_close,
        "High": float(df["High"].iloc[-1]),
        "Low": float(df["Low"].iloc[-1]),
        "Open": float(df["Open"].iloc[-1]),
        "Volume": float(df["Volume"].iloc[-1]),
        "SMA_20": float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else last_close,
        "SMA_50": float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else last_close,
        "EMA_12": float(ema_12.iloc[-1]),
        "EMA_26": float(ema_26.iloc[-1]),
        "RSI": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
        "MACD": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
        "MACD_signal": float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
        "MACD_hist": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0.0,
        "BB_upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else last_close,
        "BB_middle": float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else last_close,
        "BB_lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else last_close,
        "FVG_Size": fvg_size,
        "FVG_Type_Encoded": fvg_type,
        "OB_Type_Encoded": ob_encoded,
        "Recovery_Type_Encoded": recovery_encoded,
        "Close_lag1": lag1,
        "Close_lag2": lag2,
        "Close_lag3": lag3,
    }
    out = pd.DataFrame(
        [[row[name] for name in REQUIRED_SMC_V2_FEATURES]],
        columns=REQUIRED_SMC_V2_FEATURES,
    )
    return out.fillna(0.0), complete
```

- [ ] **Step 4: Run feature tests**

```bash
.venv/bin/pytest tests/test_smc_v2_features.py -v
```

All should pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/models/smc_v2_features.py tests/test_smc_v2_features.py
git commit -m "feat(models): add SMC v2 feature builder for HF model PoC"
```

---

## Task 5: Engine Routes To Selected Feature Builder

**Files:**
- Modify: `xau_pro_bot/signals/engine.py`
- Create: `tests/test_ai_feature_set.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_ai_feature_set.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


class CapturingAI:
    def __init__(self) -> None:
        self.last_features_columns: list[str] | None = None

    def predict(self, features):
        self.last_features_columns = list(features.columns)
        return {
            "direction": "NO_TRADE",
            "confidence": 0.0,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }


def test_engine_uses_internal_feature_set_by_default(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    monkeypatch.delenv("AI_FEATURE_SET", raising=False)
    ai = CapturingAI()

    engine = MasterSignalEngine(ai_enabled=True, ai_model=ai)
    engine.analyze(all_tfs)

    assert ai.last_features_columns is not None
    # internal builder has 29 features starting with close_m15
    assert "close_m15" in ai.last_features_columns
    assert "rsi_h1" in ai.last_features_columns


def test_engine_uses_smc_v2_feature_set_when_configured(monkeypatch, all_tfs):
    monkeypatch.setattr("xau_pro_bot.signals.engine.get_killzone", lambda: "London KZ")
    monkeypatch.setenv("AI_FEATURE_SET", "smc_v2")
    ai = CapturingAI()

    engine = MasterSignalEngine(ai_enabled=True, ai_model=ai)
    engine.analyze(all_tfs)

    assert ai.last_features_columns is not None
    # smc_v2 builder has 23 features starting with Close (uppercase)
    assert ai.last_features_columns[0] == "Close"
    assert "FVG_Size" in ai.last_features_columns
    assert len(ai.last_features_columns) == 23
```

- [ ] **Step 2: Run — expect failure (engine ignores feature_set)**

```bash
.venv/bin/pytest tests/test_ai_feature_set.py -v
```

- [ ] **Step 3: Wire feature_set into engine**

In `xau_pro_bot/signals/engine.py`:

(a) Add an import next to the existing models imports:

```python
from xau_pro_bot.models.smc_v2_features import build_smc_v2_features
```

(b) In `MasterSignalEngine.__init__`, after the `ai_cfg = config.load_ai_config()` line, capture the feature set:

```python
        self.ai_feature_set = str(ai_cfg.get("feature_set", "internal"))
```

(c) Replace the inner branch of `_run_ai_adjustment` that calls `build_ai_features`:

Find:

```python
        else:
            features, complete = build_ai_features(data)
            if not complete:
                return {
                    ...
                }
            prediction = self.ai_model.predict(features)
```

Replace with:

```python
        else:
            if self.ai_feature_set == "smc_v2":
                features, complete = build_smc_v2_features(data)
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
```

(Keep the existing dict return shape exactly as it was — only the branch selecting the builder is new.)

- [ ] **Step 4: Run new + regression tests**

```bash
.venv/bin/pytest tests/test_ai_feature_set.py tests/test_engine_ai.py tests/test_engine_ai_caching.py tests/test_engine.py -v
```

All should pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/signals/engine.py tests/test_ai_feature_set.py
git commit -m "feat(engine): route AI features via AI_FEATURE_SET"
```

---

## Task 6: Add `xgboost` Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append xgboost**

In `requirements.txt`, append:

```text
xgboost>=2.0,<3.0
```

- [ ] **Step 2: Install**

```bash
.venv/bin/pip install -r requirements-dev.txt
```

- [ ] **Step 3: Verify xgboost importable**

```bash
.venv/bin/python -c "import xgboost; print(xgboost.__version__)"
```

Expected: a version >= 2.0.

- [ ] **Step 4: Run full suite — no regressions**

```bash
.venv/bin/pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "build: add xgboost runtime dep for HF model loading"
```

---

## Task 7: Smoke Script

**Files:**
- Create: `scripts/poc_smc_v2_smoke.py`

Goal: a single-command sanity check that downloads the model with revision pin, builds a synthetic 200-bar M15 DataFrame, runs `engine.analyze`, prints the resulting signal dict.

- [ ] **Step 1: Create directory if needed**

```bash
mkdir -p scripts
```

- [ ] **Step 2: Write the script**

Create `scripts/poc_smc_v2_smoke.py`:

```python
"""Smoke test: download SMC v2 model, run engine.analyze() on synthetic data.

Run:
    AI_ENABLED=true \
    AI_MODEL_ID=JonusNattapong/xauusd-trading-ai-smc-v2 \
    AI_MODEL_REVISION=d1ee87d058bf714af1b6f4b3979646dd0024b726 \
    AI_MODEL_FILENAME=trading_model_15m.pkl \
    AI_FEATURE_SET=smc_v2 \
    .venv/bin/python scripts/poc_smc_v2_smoke.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from xau_pro_bot.signals.engine import MasterSignalEngine


def _synthetic_tfs(n: int = 300, seed: int = 42) -> dict[str, pd.DataFrame]:
    np.random.seed(seed)
    base = 2000.0 + np.cumsum(np.random.normal(0, 1.5, n))
    m15 = pd.DataFrame({
        "Open": base,
        "High": base + 2.5,
        "Low": base - 2.5,
        "Close": base + np.random.normal(0, 0.8, n),
        "Volume": np.random.randint(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    # naive resamples for higher TFs
    h1 = m15.resample("1h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    h4 = m15.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    d1 = m15.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    w1 = d1.copy()
    return {"M15": m15, "H1": h1, "H4": h4, "D1": d1, "W1": w1}


def main() -> int:
    if os.getenv("AI_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
        print("AI_ENABLED is not true — set the env vars from the docstring at the top.")
        return 1
    tfs = _synthetic_tfs()
    engine = MasterSignalEngine()
    print("AI enabled:", engine.ai_enabled)
    print("AI feature set:", engine.ai_feature_set)
    print("AI adapter:", type(engine.ai_model).__name__ if engine.ai_model else None)
    sig = engine.analyze(tfs)
    serializable = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in sig.items() if k != "reasons"}
    print(json.dumps(serializable, indent=2, default=str))
    print("\nreasons:", sig["reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the smoke**

```bash
AI_ENABLED=true \
AI_MODEL_ID=JonusNattapong/xauusd-trading-ai-smc-v2 \
AI_MODEL_REVISION=d1ee87d058bf714af1b6f4b3979646dd0024b726 \
AI_MODEL_FILENAME=trading_model_15m.pkl \
AI_FEATURE_SET=smc_v2 \
AI_CACHE_DIR=./models_cache \
.venv/bin/python scripts/poc_smc_v2_smoke.py
```

Expected:

- First run downloads the model to `./models_cache/` (one-time).
- Prints `AI enabled: True`, `AI feature set: smc_v2`, `AI adapter: HFTradingModel`.
- Prints a signal dict containing `ai_enabled: true` and either `ai_direction` set or `ai_reason="AI skipped: incomplete input features"`.

If the model fails to load (e.g., XGBoost version mismatch), the output will show `ai_reason` containing the error from `_neutral`. Report and triage rather than guessing.

- [ ] **Step 4: Commit the script**

```bash
git add scripts/poc_smc_v2_smoke.py
git commit -m "chore(scripts): add SMC v2 PoC smoke check"
```

---

## Task 8: End-to-End Compare-AI On Real Data

This is a **manual** task — no source code changes, just running.

- [ ] **Step 1: Download model's training history**

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download(
    repo_id='JonusNattapong/xauusd-trading-ai-smc-v2',
    filename='15m_data.csv',
    revision='d1ee87d058bf714af1b6f4b3979646dd0024b726',
    cache_dir='./models_cache',
)
shutil.copy(src, './data_xauusd_15m.csv')
print('Wrote ./data_xauusd_15m.csv')
"
```

- [ ] **Step 2: Inspect the CSV header and resample to H1**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('./data_xauusd_15m.csv')
print('Columns:', list(df.columns))
print('Head:')
print(df.head(3))
print('Tail:')
print(df.tail(3))
"
```

The backtester expects a `datetime` column and columns `Open, High, Low, Close, Volume`. If the CSV uses different column names (e.g., `Date`, `Time`, `Price`), normalize first. Document the exact transformation in this step.

- [ ] **Step 3: Build an H1 CSV for the backtester**

```bash
.venv/bin/python -c "
import pandas as pd

df = pd.read_csv('./data_xauusd_15m.csv')
# Adjust column names to whatever the CSV uses (Step 2 told us)
# Standardize:
df = df.rename(columns={'Datetime': 'datetime'}) if 'Datetime' in df.columns else df
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.set_index('datetime').sort_index()
h1 = df.resample('1h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
}).dropna()
h1.reset_index().to_csv('./data_xauusd_h1.csv', index=False)
print('H1 bars:', len(h1))
"
```

If the CSV provides daily-only data or a different shape, adjust accordingly.

- [ ] **Step 4: Run baseline-only backtest**

```bash
.venv/bin/python -m xau_pro_bot.backtest --csv ./data_xauusd_h1.csv --stream intraday
```

Save the printed metrics.

- [ ] **Step 5: Run compare-ai backtest**

```bash
AI_ENABLED=true \
AI_MODEL_ID=JonusNattapong/xauusd-trading-ai-smc-v2 \
AI_MODEL_REVISION=d1ee87d058bf714af1b6f4b3979646dd0024b726 \
AI_MODEL_FILENAME=trading_model_15m.pkl \
AI_FEATURE_SET=smc_v2 \
AI_CACHE_DIR=./models_cache \
.venv/bin/python -m xau_pro_bot.backtest \
    --csv ./data_xauusd_h1.csv \
    --stream intraday \
    --compare-ai \
    --ai-model-id JonusNattapong/xauusd-trading-ai-smc-v2 \
    --ai-model-revision d1ee87d058bf714af1b6f4b3979646dd0024b726 \
    --ai-model-type sklearn
```

Note that the `--compare-ai` path passes only some args via CLI; `AI_MODEL_FILENAME` and `AI_FEATURE_SET` are picked up from the env. Both must be set.

- [ ] **Step 6: Record findings**

Capture: trade delta, win-rate delta, expectancy delta, profit-factor delta, blocked signals count. Add to `docs/HANDOFF.md` under a new "PoC: SMC v2" section with one paragraph of interpretation.

- [ ] **Step 7: Commit findings**

```bash
git add docs/HANDOFF.md
git commit -m "docs(handoff): record SMC v2 PoC comparison results"
```

If the model produced an `ai_reason` containing an error (XGBoost incompatibility, feature shape mismatch, etc.), the PoC has revealed an infrastructure gap. Document it in HANDOFF.md and decide next step (fix or pivot to Path C).

---

## Self-Review

**Spec coverage:**

- Custom HF filename support: Task 2.
- Feature-set routing config: Task 1, Task 5.
- SMC v2 23-feature builder (with documented encoding choices): Tasks 3, 4.
- xgboost dependency: Task 6.
- Local smoke script: Task 7.
- Real-data comparison: Task 8.

**Out of scope (separate work):**

- Path C — training our own LightGBM. After Task 8 we'll write a separate plan informed by what the PoC revealed.
- Performance optimization (eliminating double-`_with_classic` work in the internal builder).
- AI decision logging.

**Placeholder scan:**

- No "TODO" / "implement later" steps. Every code block is concrete.
- Tasks 3 and 8 are intentionally read-only / manual; both have explicit commands and explicit success criteria.

**Type consistency:**

- `build_smc_v2_features(tfs) -> tuple[pd.DataFrame, bool]` mirrors the contract of `build_ai_features` (Task 4 of the previous critical-fixes plan), so engine branching is symmetric.
- `HFTradingModel(filename: str = "")` is keyword-only-equivalent; existing positional usage in tests is unaffected because all existing callers use keyword arguments.
- `engine.analyze(...)` return shape unchanged.

**Known PoC limitations (not bugs in the plan):**

- FVG / OB / Recovery encodings in `smc_v2_features.py` are educated guesses. Task 3 inspects the model's own dataset; if encodings differ from {-1, 0, +1}, update Task 4 implementations before running Task 7.
- Model is binary BUY-only filter. No SELL signal is ever boosted; this is acknowledged in the architecture section, not a defect.
- A failing smoke test (e.g., xgboost version pickle incompatibility) is a **valid PoC outcome** — it tells us the model isn't shippable as-is and we should pivot to Path C.
