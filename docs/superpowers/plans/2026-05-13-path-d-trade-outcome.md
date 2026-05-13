# Path D — Trade Outcome LightGBM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Path C's weak forward-return labels with TP/SL trade-outcome labels; train a Directional model (BUY/SELL/NO_TRADE, A1 + A2 variants) and a Filter model (GOOD/BAD on baseline setups); wire an opt-in filter+hybrid policy seam into the engine; evaluate across 6 AI modes and 4 non-AI baselines on the existing M15 history.

**Architecture:** New modules under `xau_pro_bot/models/` for outcome labeler, sample harvester, trainer, and filter adapter. New `hybrid_policy` module under `xau_pro_bot/signals/`. Engine and backtest gain optional, backwards-compatible knobs. Path C left fully intact. Time-based 70/15/15 split; threshold sweep on validation only; test set evaluated once.

**Tech Stack:** Python 3.11, pandas, numpy, lightgbm>=4.0, scikit-learn, joblib, pytest. Reuse existing `build_ai_features` (29 features, `internal` set) and `MasterSignalEngine` for baseline harvest.

**Spec:** `docs/superpowers/specs/2026-05-13-path-d-trade-outcome-design.md`

---

## File Map

### Create
- `xau_pro_bot/models/trade_outcome.py` — `Outcome` dataclass + `resolve_outcome_m15()` + `OutcomeClass` enum
- `xau_pro_bot/models/path_d_harvest.py` — `harvest_baseline_samples()`, `add_synthetic_samples()`, `build_path_d_features()`
- `xau_pro_bot/models/train_path_d.py` — time-split, train Directional A1/A2 + Filter, save artifacts
- `xau_pro_bot/models/trade_filter_model.py` — `TradeFilterModel` adapter class
- `xau_pro_bot/signals/hybrid_policy.py` — `HybridPolicy`, `HybridDecision`, `HybridThresholds` dataclasses; pure `decide()` function
- `scripts/train_path_d_model.py` — CLI wrapper
- `scripts/eval_path_d.py` — backtest harness for 10 modes + threshold sweep + report writer
- `tests/test_trade_outcome_labeler.py`
- `tests/test_path_d_harvest.py`
- `tests/test_trade_filter_model.py`
- `tests/test_hybrid_policy.py`
- `tests/test_path_d_eval.py`

### Modify
- `xau_pro_bot/signals/engine.py` — add `bull_score`/`bear_score` to return dict (lines 277–302); accept `filter_model`, `hybrid_policy` in `__init__` (lines ~30–48); invoke filter+hybrid after tier decision
- `xau_pro_bot/config.py` — add Path D env vars to `load_ai_config()`
- `xau_pro_bot/backtest.py` — pipe `filter_model` and `hybrid_mode` through `_build_analyzer` / `run_backtest` / `compare_backtests`
- `.env.example` — document new env vars
- `docs/HANDOFF.md` — append Path D summary after run

### Final report (written by `scripts/eval_path_d.py`)
- `docs/reports/path_d_trade_outcome_results.md`

---

## Task 1: Outcome labeler (TP/SL on M15)

**Files:**
- Create: `xau_pro_bot/models/trade_outcome.py`
- Test: `tests/test_trade_outcome_labeler.py`

- [ ] **Step 1: Write the failing test**

`tests/test_trade_outcome_labeler.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from xau_pro_bot.models.trade_outcome import (
    Outcome,
    OutcomeClass,
    resolve_outcome_m15,
)


def _bars(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])


def test_tp_hit_buy_returns_positive_R():
    future = _bars([
        [100.0, 101.0, 99.5, 100.5],   # not yet TP/SL
        [100.5, 103.0, 100.4, 102.5],  # TP=102 hit
    ])
    out = resolve_outcome_m15(entry=100.0, sl=99.0, tp=102.0,
                               direction="BUY", m15_future=future,
                               timeout_bars=192)
    assert out.hit_tp and not out.hit_sl
    assert out.final_R == pytest.approx(2.0)
    assert out.outcome_class == OutcomeClass.TP
    assert out.bars_to_outcome == 2
    assert out.tp_used == 102.0


def test_sl_hit_buy_returns_minus_one_R():
    future = _bars([[100.0, 100.2, 98.5, 99.0]])  # SL=99 hit
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 192)
    assert out.hit_sl and not out.hit_tp
    assert out.final_R == -1.0
    assert out.outcome_class == OutcomeClass.SL


def test_same_candle_tp_and_sl_resolves_to_SL_first():
    future = _bars([[100.0, 103.0, 98.5, 99.5]])  # both TP=102 AND SL=99
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 192)
    assert out.outcome_class == OutcomeClass.SAME_CANDLE_SL_FIRST
    assert out.same_candle_conflict is True
    assert out.final_R == -1.0


def test_unresolved_at_timeout():
    future = _bars([[100.0, 100.5, 99.5, 100.1]] * 5)
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 5)
    assert out.outcome_class == OutcomeClass.UNRESOLVED
    assert out.final_R == 0.0
    assert out.bars_to_outcome == 5


def test_sell_outcome_tp_hit():
    future = _bars([[100.0, 100.5, 97.5, 98.0]])  # TP=98 for SELL
    out = resolve_outcome_m15(100.0, 101.0, 98.0, "SELL", future, 192)
    assert out.hit_tp
    assert out.final_R == pytest.approx(2.0)


def test_mfe_mae_tracked_in_R_units():
    future = _bars([
        [100.0, 100.5, 99.7, 100.2],
        [100.2, 101.5, 99.2, 99.5],   # MFE up to +1.5R, MAE down to -0.8R, no exit yet
        [99.5, 102.5, 99.0, 102.2],   # TP hit on bar 3
    ])
    out = resolve_outcome_m15(100.0, 99.0, 102.0, "BUY", future, 192)
    assert out.mfe_R >= 2.0      # eventually >=2 because TP hit
    assert out.mae_R <= -0.8 + 1e-9


def test_zero_risk_raises():
    with pytest.raises(ValueError):
        resolve_outcome_m15(100.0, 100.0, 102.0, "BUY",
                            _bars([[100.0, 101.0, 99.0, 100.5]]), 192)
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_trade_outcome_labeler.py -v
```
Expected: `ImportError: cannot import name 'Outcome' from 'xau_pro_bot.models.trade_outcome'`.

- [ ] **Step 3: Implement `xau_pro_bot/models/trade_outcome.py`**

```python
"""TP/SL outcome resolver on M15 future bars for Path D labeling."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import pandas as pd


class OutcomeClass(str, enum.Enum):
    TP = "TP"
    SL = "SL"
    UNRESOLVED = "UNRESOLVED"
    SAME_CANDLE_SL_FIRST = "SAME_CANDLE_SL_FIRST"


@dataclass(frozen=True)
class Outcome:
    hit_tp: bool
    hit_sl: bool
    unresolved: bool
    same_candle_conflict: bool
    final_R: float
    mfe_R: float
    mae_R: float
    bars_to_outcome: int
    tp_used: float
    outcome_class: OutcomeClass


def resolve_outcome_m15(entry: float, sl: float, tp: float,
                        direction: str, m15_future: pd.DataFrame,
                        timeout_bars: int = 192) -> Outcome:
    """Resolve a hypothetical trade on M15 future bars.

    Conservative rule: if a single candle's range touches both TP and SL,
    treat it as SL-first (and record same_candle_conflict).
    """
    risk = abs(entry - sl)
    if risk <= 0:
        raise ValueError("zero-risk trade: entry == sl")
    R_tp = abs(tp - entry) / risk
    mfe = 0.0
    mae = 0.0
    bars = m15_future.iloc[:timeout_bars]

    for k, (_, bar) in enumerate(bars.iterrows(), start=1):
        if direction == "BUY":
            mfe = max(mfe, (bar.High - entry) / risk)
            mae = min(mae, (bar.Low - entry) / risk)
            hit_sl = bar.Low <= sl
            hit_tp = bar.High >= tp
        else:
            mfe = max(mfe, (entry - bar.Low) / risk)
            mae = min(mae, (entry - bar.High) / risk)
            hit_sl = bar.High >= sl
            hit_tp = bar.Low <= tp

        if hit_sl and hit_tp:
            return Outcome(False, True, False, True, -1.0, mfe, mae, k, tp,
                           OutcomeClass.SAME_CANDLE_SL_FIRST)
        if hit_sl:
            return Outcome(False, True, False, False, -1.0, mfe, mae, k, tp,
                           OutcomeClass.SL)
        if hit_tp:
            return Outcome(True, False, False, False, R_tp, mfe, mae, k, tp,
                           OutcomeClass.TP)

    return Outcome(False, False, True, False, 0.0, mfe, mae,
                   min(timeout_bars, len(bars)), tp, OutcomeClass.UNRESOLVED)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
.venv/bin/pytest tests/test_trade_outcome_labeler.py -v
```
Expected: all 7 pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/models/trade_outcome.py tests/test_trade_outcome_labeler.py
git commit -m "feat(models): TP/SL outcome resolver on M15 future bars"
```

---

## Task 2: Engine emits `bull_score`/`bear_score`

**Files:**
- Modify: `xau_pro_bot/signals/engine.py:277-302`
- Test: `tests/test_engine_score_breakdown.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/test_engine_score_breakdown.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.signals.engine import MasterSignalEngine


def _synthetic_history(n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2,
        "Close": base + rng.normal(0, 0.3, n),
        "Volume": rng.integers(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def test_engine_return_includes_bull_and_bear_scores():
    eng = MasterSignalEngine()
    sig = eng.analyze(_synthetic_history())
    assert "bull_score" in sig
    assert "bear_score" in sig
    assert isinstance(sig["bull_score"], (int, float))
    assert isinstance(sig["bear_score"], (int, float))
    # final_score must equal max of the two (post macro penalties + AI)
    assert sig["score"] == int(max(sig["bull_score"], sig["bear_score"]))
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_engine_score_breakdown.py -v
```
Expected: `KeyError: 'bull_score'`.

- [ ] **Step 3: Modify `xau_pro_bot/signals/engine.py`**

Inside `analyze()`, locate both return dicts (the `NO_SIGNAL` branch around line 278 and the success branch around line 293). Add the two keys to **both**:

```python
        if tier == "NO_SIGNAL":
            return {
                "direction": direction,
                "tier": tier,
                "score": int(final_score),
                "bull_score": float(bull_score),
                "bear_score": float(bear_score),
                "entry": float(m15["Close"].iloc[-1]),
                "sl": None, "tp1": None, "tp2": None, "tp3": None,
                "rr": None,
                "killzone": get_killzone(),
                "reasons": reasons,
                "tp2_unavailable": False,
                "ts_utc": datetime.now(timezone.utc),
                **ai_fields,
            }

        levels = self._compute_levels(direction, h1, m15, d1)
        return {
            "direction": direction,
            "tier": tier,
            "score": int(final_score),
            "bull_score": float(bull_score),
            "bear_score": float(bear_score),
            **levels,
            "killzone": get_killzone(),
            "reasons": reasons,
            "ts_utc": datetime.now(timezone.utc),
            **ai_fields,
        }
```

- [ ] **Step 4: Run new test + full engine tests to confirm no regression**

```bash
.venv/bin/pytest tests/test_engine_score_breakdown.py tests/ -q -k "engine or signal or backtest"
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/signals/engine.py tests/test_engine_score_breakdown.py
git commit -m "feat(engine): expose bull_score/bear_score in analyze() return"
```

---

## Task 3: Sample harvester (baseline-only + synthetic NO_TRADE)

**Files:**
- Create: `xau_pro_bot/models/path_d_harvest.py`
- Test: `tests/test_path_d_harvest.py`

- [ ] **Step 1: Write the failing test**

`tests/test_path_d_harvest.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.path_d_harvest import (
    HarvestConfig,
    harvest_path_d_samples,
)


@pytest.fixture
def long_history():
    rng = np.random.default_rng(7)
    n = 4000
    base = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2,
        "Close": base + rng.normal(0, 0.3, n),
        "Volume": rng.integers(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def test_harvest_baseline_only_emits_rows_with_outcome_metadata(long_history):
    cfg = HarvestConfig(step_h1=4, timeout_m15=192, include_synthetic=False)
    df = harvest_path_d_samples(long_history, cfg)
    if df.empty:
        pytest.skip("synthetic history did not produce baseline signals")
    required = {"entry", "sl", "tp_used", "direction", "tier",
                "bull_score", "bear_score", "score_gap",
                "outcome_class", "final_R", "mfe_R", "mae_R",
                "bars_to_outcome", "baseline_sample", "is_synthetic"}
    assert required.issubset(df.columns)
    assert df["baseline_sample"].all()
    assert (~df["is_synthetic"]).all()


def test_harvest_with_synthetic_adds_synthetic_rows(long_history):
    cfg = HarvestConfig(step_h1=4, timeout_m15=192,
                        include_synthetic=True, synth_stride=8,
                        synth_atr_sl=1.5, synth_rr=2.0)
    df = harvest_path_d_samples(long_history, cfg)
    if df.empty:
        pytest.skip("no rows harvested")
    assert df["is_synthetic"].any()
    synth = df[df["is_synthetic"]]
    assert (~synth["baseline_sample"]).all()
    # synthetic rows still carry outcome metadata
    assert synth["outcome_class"].notna().all()


def test_same_candle_conflicts_are_counted(long_history):
    cfg = HarvestConfig(step_h1=4, timeout_m15=192, include_synthetic=False)
    df = harvest_path_d_samples(long_history, cfg)
    if df.empty:
        pytest.skip("empty")
    # column always present, even if zero
    assert "outcome_class" in df.columns
    counts = df["outcome_class"].value_counts().to_dict()
    # at least TP/SL/UNRESOLVED keys may appear; conflict key may be 0
    assert set(counts.keys()).issubset({"TP", "SL", "UNRESOLVED",
                                         "SAME_CANDLE_SL_FIRST"})
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_path_d_harvest.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `xau_pro_bot/models/path_d_harvest.py`**

```python
"""Harvest training samples for Path D.

Walks history H1-bar by H1-bar, asks baseline MasterSignalEngine for a
setup, resolves the TP/SL outcome on M15 future bars, and optionally
appends synthetic ATR-based NO_TRADE samples for Mode A2.

The output is a flat DataFrame: one row per (cutoff, sample) with
features + labels + outcome bookkeeping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xau_pro_bot.models.features import build_ai_features
from xau_pro_bot.models.trade_outcome import (
    OutcomeClass,
    resolve_outcome_m15,
)
from xau_pro_bot.signals.engine import MasterSignalEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarvestConfig:
    step_h1: int = 4
    timeout_m15: int = 192          # ≈ 48 H1 bars
    label_tp_target: str = "tp1"    # tp1 | tp2
    include_synthetic: bool = False
    synth_stride: int = 8           # every Nth no-signal cutoff
    synth_atr_sl: float = 1.5       # SL = entry ± k*ATR
    synth_rr: float = 2.0           # TP = entry ± k*RR*ATR
    min_lookback_h1: int = 250


_KILLZONES = ("Asian KZ", "London KZ", "NY AM KZ", "NY PM KZ", "OFF")


def _killzone_onehot(label: str | None) -> dict[str, int]:
    label = label if label in _KILLZONES else "OFF"
    return {f"kz_{k.replace(' ', '_')}": int(label == k) for k in _KILLZONES}


def _atr(series: pd.DataFrame, n: int = 14) -> float:
    high = series["High"]; low = series["Low"]; close = series["Close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def _baseline_context_features(sig: dict, m15: pd.DataFrame,
                               h1: pd.DataFrame) -> dict:
    bull = float(sig.get("bull_score", 0.0))
    bear = float(sig.get("bear_score", 0.0))
    tier = sig.get("tier", "NO_SIGNAL")
    direction = sig.get("direction", "BUY")
    atr_h1 = _atr(h1.tail(50))
    atr_pct = float((h1["High"] - h1["Low"]).tail(100).rank(pct=True).iloc[-1])
    range_m15 = float(m15["High"].iloc[-1] - m15["Low"].iloc[-1])
    range_vs_atr = range_m15 / atr_h1 if atr_h1 > 0 else 0.0
    ts = m15.index[-1].tz_convert("America/New_York")
    return {
        "bull_score": bull,
        "bear_score": bear,
        "score_gap": abs(bull - bear),
        "final_score": float(sig.get("score", 0.0)),
        "tier_WEAK":   int(tier == "WEAK"),
        "tier_NORMAL": int(tier == "NORMAL"),
        "tier_STRONG": int(tier == "STRONG"),
        "tier_NO_SIGNAL": int(tier == "NO_SIGNAL"),
        "dir_BUY":  int(direction == "BUY"),
        "dir_SELL": int(direction == "SELL"),
        "rr": float(sig.get("rr") or 0.0),
        "hour_ny": float(ts.hour),
        "day_of_week": float(ts.dayofweek),
        "atr_percentile_h1": atr_pct,
        "range_vs_atr_m15": range_vs_atr,
        **_killzone_onehot(sig.get("killzone")),
    }


def _directional_label(direction: str, outcome_class: OutcomeClass) -> int:
    """Mode A: TP → BUY (+1) / SELL (-1); SL/UNRESOLVED/SAME → NO_TRADE (0)."""
    if outcome_class == OutcomeClass.TP:
        return 1 if direction == "BUY" else -1
    return 0


def _filter_label(outcome_class: OutcomeClass,
                  unresolved_policy: str = "bad") -> int:
    """Mode B: TP=GOOD(1); SL/SAME=BAD(0); UNRESOLVED follows policy."""
    if outcome_class == OutcomeClass.TP:
        return 1
    if outcome_class == OutcomeClass.UNRESOLVED:
        return 0 if unresolved_policy == "bad" else 1
    return 0


def harvest_path_d_samples(history: dict[str, pd.DataFrame],
                           cfg: HarvestConfig = HarvestConfig(),
                           ) -> pd.DataFrame:
    h1 = history["H1"]; m15 = history["M15"]
    if len(h1) < cfg.min_lookback_h1:
        return pd.DataFrame()

    engine = MasterSignalEngine(ai_enabled=False)
    rows: list[dict] = []

    for i in range(cfg.min_lookback_h1, len(h1) - 1, cfg.step_h1):
        cutoff = h1.index[i]
        slice_data = {tf: df.loc[:cutoff].tail(720) for tf, df in history.items()}
        try:
            sig = engine.analyze(slice_data)
        except Exception:
            continue
        if sig is None:
            continue

        m15_future = m15.loc[m15.index > cutoff]
        if len(m15_future) < 10:
            break

        # Existing 29 features
        try:
            feats_29, complete = build_ai_features(slice_data)
        except Exception:
            continue
        if not complete:
            continue
        feats_29_row = feats_29.iloc[0].to_dict()

        base_ctx = _baseline_context_features(sig, slice_data["M15"], slice_data["H1"])

        tier = sig.get("tier", "NO_SIGNAL")
        tp = (sig.get("tp1") if cfg.label_tp_target == "tp1"
              else (sig.get("tp2") or sig.get("tp1")))
        if tier in {"WEAK", "NORMAL", "STRONG"} and tp is not None and sig.get("sl") is not None:
            try:
                out = resolve_outcome_m15(
                    entry=float(sig["entry"]), sl=float(sig["sl"]),
                    tp=float(tp), direction=str(sig["direction"]),
                    m15_future=m15_future, timeout_bars=cfg.timeout_m15,
                )
            except ValueError:
                continue
            rows.append({
                **feats_29_row, **base_ctx,
                "is_synthetic": 0,
                "baseline_sample": True,
                "entry": float(sig["entry"]),
                "sl": float(sig["sl"]),
                "tp_used": float(tp),
                "direction": sig["direction"],
                "tier": tier,
                "outcome_class": out.outcome_class.value,
                "final_R": out.final_R,
                "mfe_R": out.mfe_R,
                "mae_R": out.mae_R,
                "bars_to_outcome": out.bars_to_outcome,
                "label_directional": _directional_label(sig["direction"], out.outcome_class),
                "label_filter": _filter_label(out.outcome_class),
                "cutoff": cutoff,
            })
        elif cfg.include_synthetic and (i % cfg.synth_stride == 0):
            atr = _atr(slice_data["H1"].tail(50))
            if atr <= 0:
                continue
            entry = float(slice_data["M15"]["Close"].iloc[-1])
            for direction in ("BUY", "SELL"):
                if direction == "BUY":
                    sl = entry - cfg.synth_atr_sl * atr
                    tp = entry + cfg.synth_atr_sl * cfg.synth_rr * atr
                else:
                    sl = entry + cfg.synth_atr_sl * atr
                    tp = entry - cfg.synth_atr_sl * cfg.synth_rr * atr
                try:
                    out = resolve_outcome_m15(entry, sl, tp, direction,
                                              m15_future, cfg.timeout_m15)
                except ValueError:
                    continue
                # synthetic ctx: tier=NO_SIGNAL, direction=this hypothetical
                synth_ctx = dict(base_ctx)
                synth_ctx["dir_BUY"]  = int(direction == "BUY")
                synth_ctx["dir_SELL"] = int(direction == "SELL")
                rows.append({
                    **feats_29_row, **synth_ctx,
                    "is_synthetic": 1,
                    "baseline_sample": False,
                    "entry": entry, "sl": sl, "tp_used": tp,
                    "direction": direction,
                    "tier": "NO_SIGNAL",
                    "outcome_class": out.outcome_class.value,
                    "final_R": out.final_R,
                    "mfe_R": out.mfe_R,
                    "mae_R": out.mae_R,
                    "bars_to_outcome": out.bars_to_outcome,
                    "label_directional": _directional_label(direction, out.outcome_class),
                    "label_filter": np.nan,   # filter ignores synthetics
                    "cutoff": cutoff,
                })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("cutoff").sort_index()
    return df
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_path_d_harvest.py -v
```
Expected: 3 pass (or skip on empty).

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/models/path_d_harvest.py tests/test_path_d_harvest.py
git commit -m "feat(models): Path D sample harvester (baseline + synthetic, M15 outcomes)"
```

---

## Task 4: Trainer (Directional A1, A2, Filter)

**Files:**
- Create: `xau_pro_bot/models/train_path_d.py`
- Test: `tests/test_train_path_d.py`

- [ ] **Step 1: Write the failing test**

`tests/test_train_path_d.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.train_path_d import (
    split_time_70_15_15,
    train_directional,
    train_filter,
)


def _synthetic_dataset(n=600, seed=0, with_synth=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    feats = {f"f{i}": rng.normal(0, 1, n) for i in range(20)}
    df = pd.DataFrame(feats, index=idx)
    df["is_synthetic"] = (rng.random(n) < 0.3).astype(int) if with_synth else 0
    df["baseline_sample"] = df["is_synthetic"] == 0
    df["label_directional"] = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
    df["label_filter"] = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    return df


def test_split_70_15_15_time_based_preserves_order():
    df = _synthetic_dataset()
    tr, va, te = split_time_70_15_15(df)
    assert tr.index.max() < va.index.min()
    assert va.index.max() < te.index.min()
    n = len(df)
    assert len(tr) == int(n * 0.70)
    assert len(va) == int(n * 0.15)
    assert len(te) == n - len(tr) - len(va)


def test_train_directional_a1_uses_baseline_only_rows():
    df = _synthetic_dataset(with_synth=True)
    model, metrics = train_directional(df, variant="A1")
    # A1 must drop synthetic rows from training
    assert metrics["n_train"] + metrics["n_val"] + metrics["n_test"] <= (df["baseline_sample"]).sum()
    assert set(model.classes_).issubset({-1, 0, 1})


def test_train_directional_a2_includes_synthetic_rows():
    df = _synthetic_dataset(with_synth=True)
    _, metrics_a1 = train_directional(df, variant="A1")
    _, metrics_a2 = train_directional(df, variant="A2")
    a1_total = metrics_a1["n_train"] + metrics_a1["n_val"] + metrics_a1["n_test"]
    a2_total = metrics_a2["n_train"] + metrics_a2["n_val"] + metrics_a2["n_test"]
    assert a2_total >= a1_total


def test_train_filter_binary_classes():
    df = _synthetic_dataset()
    model, metrics = train_filter(df)
    assert set(model.classes_).issubset({0, 1})
    assert metrics["n_train"] > 0 and metrics["n_test"] > 0
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_train_path_d.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `xau_pro_bot/models/train_path_d.py`**

```python
"""Path D trainer: time-split + LightGBM with conservative anti-overfit params.

Trains 3 artifacts:
- directional_a1 (baseline-only samples, 3 classes BUY/SELL/NO_TRADE)
- directional_a2 (baseline + synthetic, 3 classes)
- filter         (baseline-only samples, 2 classes GOOD/BAD)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_NON_FEATURE_COLS = {
    "entry", "sl", "tp_used", "direction", "tier",
    "outcome_class", "final_R", "mfe_R", "mae_R", "bars_to_outcome",
    "label_directional", "label_filter", "baseline_sample",
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _NON_FEATURE_COLS and df[c].dtype.kind in "fiub"]


def split_time_70_15_15(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    n = len(df)
    i_tr = int(n * 0.70)
    i_va = i_tr + int(n * 0.15)
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]


def _lgb_params(num_class: int | None) -> dict:
    p = dict(
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_data_in_leaf=120,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",
        n_estimators=600,
        n_jobs=-1, verbose=-1,
        random_state=42,
    )
    if num_class is None:
        p["objective"] = "binary"
    else:
        p["objective"] = "multiclass"
        p["num_class"] = num_class
    return p


def _fit_lgb(X_tr, y_tr, X_va, y_va, params: dict):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(40)])
    return model


def _metrics(model, X_te, y_te) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                  precision_recall_fscore_support)
    pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "report": classification_report(y_te, pred, zero_division=0),
    }


def train_directional(df: pd.DataFrame, *, variant: Literal["A1", "A2"]):
    """Train 3-class BUY/SELL/NO_TRADE."""
    data = df if variant == "A2" else df[df["baseline_sample"]]
    data = data.dropna(subset=["label_directional"])
    tr, va, te = split_time_70_15_15(data)
    fcols = _feature_cols(data)
    X_tr, y_tr = tr[fcols], tr["label_directional"].astype(int)
    X_va, y_va = va[fcols], va["label_directional"].astype(int)
    X_te, y_te = te[fcols], te["label_directional"].astype(int)
    model = _fit_lgb(X_tr, y_tr, X_va, y_va, _lgb_params(num_class=3))
    m = _metrics(model, X_te, y_te)
    m.update({"n_train": len(tr), "n_val": len(va), "n_test": len(te),
              "feature_cols": fcols, "variant": variant})
    return model, m


def train_filter(df: pd.DataFrame):
    """Train 2-class GOOD/BAD on baseline-only rows."""
    data = df[df["baseline_sample"]].dropna(subset=["label_filter"])
    tr, va, te = split_time_70_15_15(data)
    fcols = _feature_cols(data)
    X_tr, y_tr = tr[fcols], tr["label_filter"].astype(int)
    X_va, y_va = va[fcols], va["label_filter"].astype(int)
    X_te, y_te = te[fcols], te["label_filter"].astype(int)
    model = _fit_lgb(X_tr, y_tr, X_va, y_va, _lgb_params(num_class=None))
    m = _metrics(model, X_te, y_te)
    m.update({"n_train": len(tr), "n_val": len(va), "n_test": len(te),
              "feature_cols": fcols})
    return model, m


def save_model(model, feature_cols: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_train_path_d.py -v
```
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/models/train_path_d.py tests/test_train_path_d.py
git commit -m "feat(models): Path D trainer (Directional A1/A2 + Filter) with time-split"
```

---

## Task 5: TradeFilterModel adapter

**Files:**
- Create: `xau_pro_bot/models/trade_filter_model.py`
- Test: `tests/test_trade_filter_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_trade_filter_model.py`:
```python
from __future__ import annotations

import joblib
import pandas as pd
import pytest

from xau_pro_bot.models.trade_filter_model import (
    FilterDecision,
    TradeFilterModel,
)


class _StubBinary:
    classes_ = [0, 1]

    def __init__(self, good_prob: float):
        self._p = good_prob

    def predict_proba(self, X):
        return [[1 - self._p, self._p]]


def _dump(tmp_path, good_prob: float, fcols=("f0",)):
    p = tmp_path / "f.joblib"
    joblib.dump({"model": _StubBinary(good_prob), "feature_cols": list(fcols)}, p)
    return p


def test_keep_when_good_prob_above_threshold(tmp_path):
    p = _dump(tmp_path, good_prob=0.80)
    m = TradeFilterModel(local_path=str(p), threshold=0.55)
    pred = m.predict(pd.DataFrame([{"f0": 0.1}]))
    assert pred["good_prob"] == pytest.approx(0.80)
    assert pred["bad_prob"]  == pytest.approx(0.20)
    assert pred["decision"]  == FilterDecision.KEEP
    assert pred["threshold_used"] == 0.55


def test_block_when_good_prob_below_threshold(tmp_path):
    p = _dump(tmp_path, good_prob=0.30)
    m = TradeFilterModel(local_path=str(p), threshold=0.55)
    pred = m.predict(pd.DataFrame([{"f0": 0.0}]))
    assert pred["decision"] == FilterDecision.BLOCK


def test_missing_features_filled_with_zero(tmp_path):
    p = _dump(tmp_path, good_prob=0.7, fcols=("f0", "f1", "f2"))
    m = TradeFilterModel(local_path=str(p), threshold=0.5)
    pred = m.predict(pd.DataFrame([{"f0": 1.0}]))  # f1/f2 absent
    assert pred["decision"] == FilterDecision.KEEP


def test_load_failure_yields_neutral_keep(tmp_path):
    m = TradeFilterModel(local_path=str(tmp_path / "missing.joblib"),
                          threshold=0.55)
    pred = m.predict(pd.DataFrame([{"f0": 0.0}]))
    assert pred["decision"] == FilterDecision.KEEP   # neutral fallback
    assert pred["error"] is not None
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_trade_filter_model.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `xau_pro_bot/models/trade_filter_model.py`**

```python
"""Path D filter model adapter (GOOD/BAD → KEEP/BLOCK)."""

from __future__ import annotations

import enum
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FilterDecision(str, enum.Enum):
    KEEP = "KEEP"
    BLOCK = "BLOCK"


class TradeFilterModel:
    """Loads a Path D filter joblib bundle `{model, feature_cols}` and
    returns `{good_prob, bad_prob, decision, threshold_used}`.

    On load/predict failure returns a neutral KEEP with `error` populated —
    callers can decide to ignore (default: keep the baseline behaviour).
    """

    def __init__(self, local_path: str, threshold: float = 0.55) -> None:
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
            probs = np.asarray(model.predict_proba(X))[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            good_idx = classes.index(1) if 1 in classes else 1
            bad_idx = classes.index(0) if 0 in classes else 0
            good = float(probs[good_idx])
            bad = float(probs[bad_idx])
            decision = (FilterDecision.KEEP if good >= self.threshold
                        else FilterDecision.BLOCK)
            return {
                "good_prob": good, "bad_prob": bad,
                "decision": decision, "threshold_used": self.threshold,
                "error": None,
            }
        except Exception as exc:
            log.exception("TradeFilterModel.predict failed")
            return {
                "good_prob": None, "bad_prob": None,
                "decision": FilterDecision.KEEP,
                "threshold_used": self.threshold, "error": str(exc),
            }
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_trade_filter_model.py -v
```
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/models/trade_filter_model.py tests/test_trade_filter_model.py
git commit -m "feat(models): TradeFilterModel adapter for Path D filter"
```

---

## Task 6: HybridPolicy (pure)

**Files:**
- Create: `xau_pro_bot/signals/hybrid_policy.py`
- Test: `tests/test_hybrid_policy.py`

- [ ] **Step 1: Write the failing test**

`tests/test_hybrid_policy.py`:
```python
from __future__ import annotations

import pytest

from xau_pro_bot.models.trade_filter_model import FilterDecision
from xau_pro_bot.signals.hybrid_policy import (
    HybridDecision,
    HybridThresholds,
    decide,
)


T = HybridThresholds(weak=0.70, normal=0.55, strong_block=0.80,
                      directional_conflict=0.65)


def _filter(good_prob: float, decision=None):
    return {
        "good_prob": good_prob,
        "bad_prob": 1.0 - good_prob,
        "decision": decision or (FilterDecision.KEEP if good_prob >= 0.5
                                  else FilterDecision.BLOCK),
        "threshold_used": 0.55,
        "error": None,
    }


def test_no_signal_passthrough():
    d = decide(tier="NO_SIGNAL", baseline_dir="BUY",
               ai_directional=None, ai_filter=None, thresholds=T)
    assert d == HybridDecision.KEEP


def test_strong_keep_by_default():
    d = decide("STRONG", "BUY", None, _filter(0.5), T)
    assert d == HybridDecision.KEEP


def test_strong_blocked_only_when_filter_very_confident_bad():
    d = decide("STRONG", "BUY", None, _filter(0.10), T)  # bad_prob=0.90
    assert d == HybridDecision.BLOCK


def test_normal_requires_filter_approval():
    assert decide("NORMAL", "BUY", None, _filter(0.40), T) == HybridDecision.BLOCK
    assert decide("NORMAL", "BUY", None, _filter(0.60), T) == HybridDecision.KEEP


def test_weak_high_bar():
    assert decide("WEAK", "BUY", None, _filter(0.60), T) == HybridDecision.BLOCK
    assert decide("WEAK", "BUY", None, _filter(0.75), T) == HybridDecision.KEEP


def test_directional_conflict_blocks_normal():
    ai = {"direction": "SELL", "confidence": 0.70}
    assert decide("NORMAL", "BUY", ai, _filter(0.80), T) == HybridDecision.BLOCK


def test_directional_low_confidence_does_not_block():
    ai = {"direction": "SELL", "confidence": 0.50}
    assert decide("NORMAL", "BUY", ai, _filter(0.80), T) == HybridDecision.KEEP


def test_works_without_filter():
    # If filter missing, mode degrades gracefully: only directional gate.
    assert decide("NORMAL", "BUY", None, None, T) == HybridDecision.KEEP
    assert decide("WEAK",   "BUY", None, None, T) == HybridDecision.KEEP
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_hybrid_policy.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `xau_pro_bot/signals/hybrid_policy.py`**

```python
"""Pure hybrid-mode policy combining baseline tier, directional model,
and filter model into KEEP/BLOCK. No I/O, no globals — easy to unit-test."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class HybridDecision(str, enum.Enum):
    KEEP = "KEEP"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class HybridThresholds:
    weak: float = 0.70
    normal: float = 0.55
    strong_block: float = 0.80
    directional_conflict: float = 0.65


def decide(*, tier: str, baseline_dir: str,
           ai_directional: dict | None,
           ai_filter: dict | None,
           thresholds: HybridThresholds) -> HybridDecision:
    if tier == "NO_SIGNAL":
        return HybridDecision.KEEP

    if (ai_directional and
            ai_directional.get("direction") and
            ai_directional["direction"] != baseline_dir and
            float(ai_directional.get("confidence", 0.0)) > thresholds.directional_conflict):
        return HybridDecision.BLOCK

    if tier == "STRONG":
        if ai_filter and ai_filter.get("bad_prob") is not None:
            if float(ai_filter["bad_prob"]) >= thresholds.strong_block:
                return HybridDecision.BLOCK
        return HybridDecision.KEEP

    if tier == "NORMAL":
        if ai_filter and ai_filter.get("good_prob") is not None:
            if float(ai_filter["good_prob"]) < thresholds.normal:
                return HybridDecision.BLOCK
        return HybridDecision.KEEP

    if tier == "WEAK":
        if ai_filter and ai_filter.get("good_prob") is not None:
            if float(ai_filter["good_prob"]) < thresholds.weak:
                return HybridDecision.BLOCK
        return HybridDecision.KEEP

    return HybridDecision.KEEP
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_hybrid_policy.py -v
```
Expected: 8 pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/signals/hybrid_policy.py tests/test_hybrid_policy.py
git commit -m "feat(signals): hybrid policy combining tier + directional + filter"
```

---

## Task 7: Config + engine wiring for filter+hybrid

**Files:**
- Modify: `xau_pro_bot/config.py:125-146`
- Modify: `xau_pro_bot/signals/engine.py:30-48` and end of `analyze()`
- Modify: `.env.example`
- Test: `tests/test_engine_filter_integration.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/test_engine_filter_integration.py`:
```python
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.config import load_ai_config
from xau_pro_bot.models.trade_filter_model import TradeFilterModel
from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.hybrid_policy import HybridThresholds


class _StubFilter:
    classes_ = [0, 1]
    def __init__(self, good): self._g = good
    def predict_proba(self, X): return [[1 - self._g, self._g]]


@pytest.fixture
def history():
    rng = np.random.default_rng(3)
    n = 500
    base = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2,
        "Close": base + rng.normal(0, 0.3, n),
        "Volume": rng.integers(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def test_config_includes_path_d_keys(monkeypatch):
    monkeypatch.setenv("AI_PATH_D_FILTER_PATH", "/tmp/f.joblib")
    monkeypatch.setenv("AI_HYBRID_MODE", "filter")
    monkeypatch.setenv("AI_FILTER_THRESHOLD_NORMAL", "0.60")
    cfg = load_ai_config()
    assert cfg["path_d_filter_path"] == "/tmp/f.joblib"
    assert cfg["hybrid_mode"] == "filter"
    assert cfg["filter_threshold_normal"] == 0.60


def test_filter_block_marks_signal_as_blocked(tmp_path, history):
    fp = tmp_path / "f.joblib"
    joblib.dump({"model": _StubFilter(good=0.10),
                 "feature_cols": ["dummy"]}, fp)
    filt = TradeFilterModel(local_path=str(fp), threshold=0.55)
    eng = MasterSignalEngine(filter_model=filt,
                             hybrid_thresholds=HybridThresholds())
    sig = eng.analyze(history)
    if sig["tier"] == "NO_SIGNAL":
        pytest.skip("baseline produced no signal in this synthetic slice")
    # filter very confident BAD → any non-strong tier should be blocked
    if sig["tier"] in {"WEAK", "NORMAL"}:
        assert sig.get("ai_blocked") is True
        assert sig["tier"] == "NO_SIGNAL"


def test_engine_without_filter_unchanged(history):
    eng = MasterSignalEngine()
    sig = eng.analyze(history)
    assert "bull_score" in sig  # from task 2
    # No filter wired → no Path D blocking introduced
    assert not sig.get("ai_blocked", False)
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_engine_filter_integration.py -v
```
Expected: `KeyError: 'path_d_filter_path'` and ctor TypeError.

- [ ] **Step 3: Extend `xau_pro_bot/config.py` `load_ai_config()`**

Append before the closing brace of the return dict:
```python
        "feature_set": os.getenv("AI_FEATURE_SET", "internal"),
        "local_path": os.getenv("AI_MODEL_LOCAL_PATH", ""),
        "path_d_filter_path": os.getenv("AI_PATH_D_FILTER_PATH", ""),
        "path_d_directional_path": os.getenv("AI_PATH_D_DIRECTIONAL_PATH", ""),
        "hybrid_mode": os.getenv("AI_HYBRID_MODE", "off"),
        "filter_threshold_weak":   _env_float("AI_FILTER_THRESHOLD_WEAK", 0.70),
        "filter_threshold_normal": _env_float("AI_FILTER_THRESHOLD_NORMAL", 0.55),
        "filter_threshold_strong_block": _env_float("AI_FILTER_THRESHOLD_STRONG_BLOCK", 0.80),
        "filter_unresolved_policy": os.getenv("FILTER_UNRESOLVED_POLICY", "bad"),
        "label_tp_target": os.getenv("LABEL_TP_TARGET", "tp1"),
    }
```

- [ ] **Step 4: Extend `xau_pro_bot/signals/engine.py`**

Add imports near existing imports:
```python
from xau_pro_bot.signals.hybrid_policy import (
    HybridDecision, HybridThresholds, decide as hybrid_decide,
)
```

Update `__init__` signature (around line 30):
```python
    def __init__(self, ai_enabled: bool | None = None, ai_model=None,
                 filter_model=None,
                 hybrid_thresholds: HybridThresholds | None = None):
        ...
        self.filter_model = filter_model
        self.hybrid_thresholds = hybrid_thresholds or HybridThresholds()
```

At the **end** of `analyze()`, just before each `return` (both branches) — wrap the post-tier signal through the filter+hybrid. Simplest: compute `decision` after `tier` is known, before building return dicts:

```python
        # Path D filter/hybrid gate (opt-in)
        filter_pred = None
        if self.filter_model is not None and tier != "NO_SIGNAL":
            try:
                feats_29, _ = build_ai_features(data)   # already computed elsewhere; reuse if cheap
            except Exception:
                feats_29 = pd.DataFrame([{}])
            filter_pred = self.filter_model.predict(feats_29)
            d = hybrid_decide(tier=tier, baseline_dir=direction,
                              ai_directional=None,
                              ai_filter=filter_pred,
                              thresholds=self.hybrid_thresholds)
            if d == HybridDecision.BLOCK:
                tier = "NO_SIGNAL"
                ai_fields = dict(ai_fields)
                ai_fields["ai_blocked"] = True
                ai_fields["ai_reason"] = (ai_fields.get("ai_reason") or
                                           f"path_d_filter:{filter_pred.get('decision')}")
```

(Place this block right after the existing `tier = ...` line, before the `if tier == "NO_SIGNAL"` branch.)

Add `build_ai_features` import to engine.py if not already imported.

- [ ] **Step 5: Update `.env.example`**

Append:
```
AI_PATH_D_FILTER_PATH=
AI_PATH_D_DIRECTIONAL_PATH=
AI_HYBRID_MODE=off
AI_FILTER_THRESHOLD_WEAK=0.70
AI_FILTER_THRESHOLD_NORMAL=0.55
AI_FILTER_THRESHOLD_STRONG_BLOCK=0.80
FILTER_UNRESOLVED_POLICY=bad
LABEL_TP_TARGET=tp1
```

- [ ] **Step 6: Run integration + full test suite for regression**

```bash
.venv/bin/pytest tests/test_engine_filter_integration.py -v
.venv/bin/pytest -q
```
Expected: integration tests pass, full suite shows **no new failures** vs pre-Task-7 baseline (the 2 pre-existing test_ai_model_filename interferences remain unchanged).

- [ ] **Step 7: Commit**

```bash
git add xau_pro_bot/config.py xau_pro_bot/signals/engine.py .env.example tests/test_engine_filter_integration.py
git commit -m "feat(engine): opt-in Path D filter+hybrid seam (backwards-compatible)"
```

---

## Task 8: Backtest wiring (filter + hybrid)

**Files:**
- Modify: `xau_pro_bot/backtest.py:104-145, 186-225`
- Test: `tests/test_backtest_filter_wiring.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/test_backtest_filter_wiring.py`:
```python
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.backtest import run_backtest
from xau_pro_bot.models.trade_filter_model import TradeFilterModel


class _StubFilter:
    classes_ = [0, 1]
    def predict_proba(self, X): return [[0.9, 0.1]]  # always BAD


@pytest.fixture
def history():
    rng = np.random.default_rng(5)
    n = 600
    base = 2000.0 + np.cumsum(rng.normal(0, 1.0, n))
    m15 = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2,
        "Close": base + rng.normal(0, 0.3, n),
        "Volume": rng.integers(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def test_run_backtest_accepts_filter_model(tmp_path, history):
    fp = tmp_path / "f.joblib"
    joblib.dump({"model": _StubFilter(), "feature_cols": []}, fp)
    filt = TradeFilterModel(local_path=str(fp), threshold=0.55)
    base = run_backtest(history, timeout_bars=24, step=4)
    blocked = run_backtest(history, timeout_bars=24, step=4,
                           filter_model=filt)
    # always-BAD filter must block at least as many signals as baseline emits
    assert blocked.signals_generated <= base.signals_generated
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_backtest_filter_wiring.py -v
```
Expected: `TypeError: run_backtest() got an unexpected keyword argument 'filter_model'`.

- [ ] **Step 3: Modify `xau_pro_bot/backtest.py`**

Extend `_build_analyzer` signature and body:
```python
def _build_analyzer(stream: str, use_ai: bool = False,
                    ai_model: Any | None = None,
                    ai_model_id: str = "",
                    ai_model_type: str = "sklearn",
                    ai_model_revision: str = "",
                    filter_model: Any | None = None,
                    hybrid_thresholds=None):
    router = StreamRouter()
    if stream not in router.analyzers:
        raise ValueError(f"Unknown stream: {stream}")
    if stream != "intraday" or (not use_ai and filter_model is None):
        return router.analyzers[stream]
    model = ai_model
    if model is None and ai_model_id:
        model = HFTradingModel(
            model_id=ai_model_id, model_type=ai_model_type,
            revision=ai_model_revision,
        )
    return MasterSignalEngine(ai_enabled=use_ai, ai_model=model,
                              filter_model=filter_model,
                              hybrid_thresholds=hybrid_thresholds)
```

Extend `run_backtest` and `compare_backtests` to accept and forward `filter_model` and `hybrid_thresholds`:
```python
def run_backtest(history, timeout_bars=48, step=4, stream="intraday",
                 use_ai=False, ai_model=None, ai_model_id="",
                 ai_model_type="sklearn", ai_model_revision="",
                 filter_model=None, hybrid_thresholds=None):
    analyzer = _build_analyzer(stream=stream, use_ai=use_ai, ai_model=ai_model,
                               ai_model_id=ai_model_id, ai_model_type=ai_model_type,
                               ai_model_revision=ai_model_revision,
                               filter_model=filter_model,
                               hybrid_thresholds=hybrid_thresholds)
    ...
```

Add the same two params to `compare_backtests` and forward into the AI-side `run_backtest` call.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_backtest_filter_wiring.py -v
.venv/bin/pytest -q tests/test_backtest*.py
```
Expected: new test passes; existing backtest tests still pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/backtest.py tests/test_backtest_filter_wiring.py
git commit -m "feat(backtest): pipe filter_model and hybrid_thresholds through harness"
```

---

## Task 9: Training CLI

**Files:**
- Create: `scripts/train_path_d_model.py`

- [ ] **Step 1: Create `scripts/train_path_d_model.py`**

```python
"""Train Path D models from a long M15 CSV.

Run all three artifacts in one go:

    PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \\
        --csv ./data_long_m15.csv --out-dir ./models_cache

Outputs:
    models_cache/path_d_directional_a1_lgb.joblib
    models_cache/path_d_directional_a2_lgb.joblib
    models_cache/path_d_trade_outcome_lgb.joblib
    models_cache/path_d_dataset.parquet         (harvested rows)
    models_cache/path_d_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from xau_pro_bot.models.path_d_harvest import HarvestConfig, harvest_path_d_samples
from xau_pro_bot.models.train_path_d import (
    save_model, train_directional, train_filter,
)


def _load_history(csv: Path) -> dict[str, pd.DataFrame]:
    m15 = pd.read_csv(csv)
    m15["datetime"] = pd.to_datetime(m15["datetime"], utc=True)
    m15 = m15.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step-h1", type=int, default=4)
    ap.add_argument("--timeout-m15", type=int, default=192)
    ap.add_argument("--synth-stride", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history = _load_history(Path(args.csv))
    print(f"Loaded M15: {len(history['M15'])} bars")

    cfg = HarvestConfig(step_h1=args.step_h1, timeout_m15=args.timeout_m15,
                        include_synthetic=True, synth_stride=args.synth_stride)
    df = harvest_path_d_samples(history, cfg)
    print(f"Dataset: rows={len(df)}, baseline={int(df['baseline_sample'].sum())}, "
          f"synthetic={int(df['is_synthetic'].sum())}")
    if len(df) < 200:
        print("Not enough samples — aborting.")
        return 1
    df.to_parquet(out_dir / "path_d_dataset.parquet")

    outcome_dist = df["outcome_class"].value_counts(normalize=True).to_dict()
    print("Outcome distribution:", outcome_dist)

    print("Training Directional A1...")
    m_a1, met_a1 = train_directional(df, variant="A1")
    save_model(m_a1, met_a1["feature_cols"], out_dir / "path_d_directional_a1_lgb.joblib")

    print("Training Directional A2...")
    m_a2, met_a2 = train_directional(df, variant="A2")
    save_model(m_a2, met_a2["feature_cols"], out_dir / "path_d_directional_a2_lgb.joblib")

    print("Training Filter...")
    m_f, met_f = train_filter(df)
    save_model(m_f, met_f["feature_cols"], out_dir / "path_d_trade_outcome_lgb.joblib")

    metrics = {
        "outcome_distribution": outcome_dist,
        "directional_a1": {k: v for k, v in met_a1.items() if k not in ("report", "feature_cols")},
        "directional_a2": {k: v for k, v in met_a2.items() if k not in ("report", "feature_cols")},
        "filter":         {k: v for k, v in met_f.items()  if k not in ("report", "feature_cols")},
        "reports": {
            "directional_a1": met_a1["report"],
            "directional_a2": met_a2["report"],
            "filter":         met_f["report"],
        },
    }
    (out_dir / "path_d_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Done. Artifacts:", sorted(p.name for p in out_dir.glob("path_d_*")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run training end-to-end**

```bash
PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
    --csv ./data_long_m15.csv \
    --out-dir ./models_cache
```
Expected: prints dataset size, outcome distribution, three model artifacts written, `path_d_metrics.json` populated.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_path_d_model.py
git commit -m "chore(scripts): Path D training CLI (A1 + A2 + Filter)"
```

---

## Task 10: Evaluator — 6 AI modes + 4 non-AI baselines + threshold sweep

**Files:**
- Create: `scripts/eval_path_d.py`
- Test: `tests/test_path_d_eval.py`

- [ ] **Step 1: Write the failing test (small unit, real run is manual)**

`tests/test_path_d_eval.py`:
```python
from __future__ import annotations

from xau_pro_bot.backtest import BacktestResult
from scripts.eval_path_d import (
    pick_best_threshold,
    tier_filter_result,
)


def _mk(sig=100, wins=40, losses=50, blocked=0, rr_values=None):
    r = BacktestResult()
    r.signals_generated = sig
    r.wins = wins
    r.losses = losses
    r.blocked_signals = blocked
    r.rr_values = list(rr_values or [1.5] * sig)
    return r


def test_pick_best_threshold_prefers_higher_pf_then_more_trades():
    # threshold -> (PF, kept_trades)
    sweep = {
        0.50: {"pf": 0.9, "kept": 200, "expectancy": -0.05, "wr": 0.30},
        0.60: {"pf": 1.1, "kept": 140, "expectancy": +0.04, "wr": 0.38},
        0.65: {"pf": 1.1, "kept": 120, "expectancy": +0.04, "wr": 0.39},
        0.70: {"pf": 1.2, "kept":  40, "expectancy": +0.05, "wr": 0.40},
    }
    # min_kept_floor = 60: 0.70 excluded → best = 0.60 (PF tie → more kept)
    assert pick_best_threshold(sweep, min_kept=60) == 0.60


def test_tier_filter_result_drops_below_tier(monkeypatch):
    # synthetic per_tier dict
    r = _mk(sig=10, wins=4, losses=6)
    r.per_tier = {
        "WEAK":   {"n": 5, "w": 1, "l": 4},
        "NORMAL": {"n": 3, "w": 2, "l": 1},
        "STRONG": {"n": 2, "w": 1, "l": 1},
    }
    out = tier_filter_result(r, keep={"NORMAL", "STRONG"})
    assert out.signals_generated == 5
    assert out.wins == 3
    assert out.losses == 2
```

- [ ] **Step 2: Run test, expect fail**

```bash
.venv/bin/pytest tests/test_path_d_eval.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `scripts/eval_path_d.py`**

```python
"""Path D evaluator.

Modes (all run on the same history, same step / timeout):

  AI modes:
    A baseline only (AI off)
    B Path C directional (existing)
    C Path D directional A1
    D Path D directional A2
    E Path D filter only
    F Hybrid (Path D filter + optional directional)

  Non-AI baselines (prove AI > simple tier rules):
    G baseline all              (= A)
    H baseline without WEAK
    I baseline STRONG only
    J baseline STRONG + NORMAL only

For the filter, sweeps thresholds {0.50..0.75} on **validation slice** and
records the best by PF (tiebreak: kept trades), enforcing min-kept floor
(25% of baseline test trades). Test slice evaluated once with chosen threshold.

Writes `docs/reports/path_d_trade_outcome_results.md`.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from xau_pro_bot.backtest import BacktestResult, run_backtest
from xau_pro_bot.models.hf_model import HFTradingModel
from xau_pro_bot.models.trade_filter_model import TradeFilterModel
from xau_pro_bot.signals.hybrid_policy import HybridThresholds


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def _load(csv: Path) -> dict[str, pd.DataFrame]:
    m15 = pd.read_csv(csv)
    m15["datetime"] = pd.to_datetime(m15["datetime"], utc=True)
    m15 = m15.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return {
        "M15": m15,
        "H1":  m15.resample("1h").agg(agg).dropna(),
        "H4":  m15.resample("4h").agg(agg).dropna(),
        "D1":  m15.resample("1D").agg(agg).dropna(),
        "W1":  m15.resample("1W").agg(agg).dropna(),
    }


def _slice_history(history, start, end):
    return {tf: df.loc[start:end] for tf, df in history.items()}


def _result_summary(r: BacktestResult) -> dict:
    return {
        "trades": r.signals_generated,
        "blocked": r.blocked_signals,
        "wins": r.wins, "losses": r.losses,
        "wr": round(r.win_rate, 4),
        "expectancy": round(r.expectancy, 4),
        "pf": round(r.profit_factor, 4),
        "avg_rr": round(r.average_rr, 4),
        "max_dd": round(r.max_drawdown, 4),
    }


def tier_filter_result(r: BacktestResult, keep: set[str]) -> BacktestResult:
    """Synthesize a 'baseline without tier X' result from per_tier counters."""
    out = BacktestResult()
    out.per_tier = {t: {"n": 0, "w": 0, "l": 0} for t in keep}
    for tier, cnt in r.per_tier.items():
        if tier in keep:
            out.signals_generated += cnt["n"]
            out.wins += cnt["w"]
            out.losses += cnt["l"]
            out.per_tier[tier] = dict(cnt)
    # rr_values not separable per tier without richer tracking
    out.rr_values = []
    return out


def pick_best_threshold(sweep: dict[float, dict], min_kept: int) -> float | None:
    eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
    if not eligible:
        return None
    return sorted(eligible.items(),
                  key=lambda kv: (kv[1]["pf"], kv[1]["kept"]),
                  reverse=True)[0][0]


def _threshold_sweep(history_val, filter_path: str, min_kept: int,
                     base_kwargs: dict) -> tuple[dict, float | None]:
    sweep: dict[float, dict] = {}
    for t in THRESHOLDS:
        flt = TradeFilterModel(local_path=filter_path, threshold=float(t))
        r = run_backtest(history_val, filter_model=flt, **base_kwargs)
        sweep[t] = {
            "pf": float(r.profit_factor),
            "expectancy": float(r.expectancy),
            "wr": float(r.win_rate),
            "kept": int(r.signals_generated),
            "blocked": int(r.blocked_signals),
        }
    chosen = pick_best_threshold(sweep, min_kept=min_kept)
    return sweep, chosen


def run_all_modes(history, *, path_c_local: str | None,
                  path_d_filter: str | None,
                  path_d_dir_a1: str | None,
                  path_d_dir_a2: str | None,
                  val_split=(0.70, 0.85)) -> dict:
    n = len(history["H1"])
    t_tr = history["H1"].index[int(n * val_split[0])]
    t_te = history["H1"].index[int(n * val_split[1])]
    val_hist = _slice_history(history, t_tr, t_te)
    test_hist = _slice_history(history, t_te, history["H1"].index[-1])

    base_kwargs = dict(timeout_bars=48, step=4, stream="intraday")

    results: dict = {}

    # A baseline only (test slice)
    a = run_backtest(test_hist, **base_kwargs)
    results["A_baseline"] = _result_summary(a)

    # G–J non-AI tier filters (derived from A.per_tier)
    results["H_no_weak"]            = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))
    results["I_strong_only"]        = _result_summary(tier_filter_result(a, {"STRONG"}))
    results["J_strong_normal_only"] = _result_summary(tier_filter_result(a, {"NORMAL", "STRONG"}))

    # B Path C
    if path_c_local:
        ai = HFTradingModel(model_id="", model_type="sklearn", local_path=path_c_local)
        b = run_backtest(test_hist, ai_model=ai, use_ai=True, **base_kwargs)
        results["B_path_c"] = _result_summary(b)

    # E Path D filter — pick threshold on val, apply on test once
    chosen_threshold = None
    sweep = {}
    if path_d_filter:
        min_kept = max(1, int(a.signals_generated * 0.25))
        sweep, chosen_threshold = _threshold_sweep(val_hist, path_d_filter,
                                                    min_kept=min_kept,
                                                    base_kwargs=base_kwargs)
        if chosen_threshold is not None:
            flt = TradeFilterModel(local_path=path_d_filter,
                                    threshold=float(chosen_threshold))
            e = run_backtest(test_hist, filter_model=flt, **base_kwargs)
            results["E_path_d_filter"] = _result_summary(e)

    # F Hybrid — same threshold + thresholds for all tiers
    if path_d_filter and chosen_threshold is not None:
        flt = TradeFilterModel(local_path=path_d_filter,
                                threshold=float(chosen_threshold))
        thr = HybridThresholds(weak=0.70, normal=float(chosen_threshold),
                                strong_block=0.80)
        f = run_backtest(test_hist, filter_model=flt,
                          hybrid_thresholds=thr, **base_kwargs)
        results["F_hybrid"] = _result_summary(f)

    return {
        "results": results,
        "threshold_sweep": sweep,
        "chosen_threshold": chosen_threshold,
        "test_window": (str(test_hist["H1"].index.min()),
                         str(test_hist["H1"].index.max())),
        "val_window":  (str(val_hist["H1"].index.min()),
                         str(val_hist["H1"].index.max())),
    }


def _md_table(summary: dict[str, dict]) -> str:
    cols = ["trades", "blocked", "wins", "losses", "wr",
            "expectancy", "pf", "avg_rr", "max_dd"]
    header = "| mode | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for k, v in summary.items():
        rows.append("| " + k + " | " + " | ".join(str(v.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def write_report(payload: dict, out_path: Path,
                  metrics_json_path: Path | None = None) -> None:
    res = payload["results"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Path D — Trade Outcome Results",
        "",
        f"**Test window:** {payload['test_window'][0]} → {payload['test_window'][1]}",
        f"**Validation window:** {payload['val_window'][0]} → {payload['val_window'][1]}",
        f"**Chosen filter threshold (from validation):** {payload['chosen_threshold']}",
        "",
        "## Modes",
        "",
        _md_table(res),
        "",
        "## Filter Threshold Sweep (validation)",
        "",
    ]
    sweep = payload["threshold_sweep"]
    if sweep:
        lines.append("| threshold | kept | blocked | wr | expectancy | pf |")
        lines.append("|---|---|---|---|---|---|")
        for t, m in sorted(sweep.items()):
            lines.append(f"| {t:.2f} | {m['kept']} | {m['blocked']} | "
                         f"{m['wr']:.3f} | {m['expectancy']:.3f} | {m['pf']:.3f} |")
    else:
        lines.append("_(no sweep — filter not provided)_")
    lines += [
        "",
        "## Acceptance check",
        "",
        f"- Min trade floor (25% of baseline test trades): "
        f"{int(res['A_baseline']['trades'] * 0.25)} ",
        "",
        "## Notes",
        "",
        "- Path C: forward-return labels (legacy).",
        "- Path D: TP/SL outcomes on M15, time-split 70/15/15, threshold picked on validation only.",
        "- Acceptance: PF > Path C **and** kept_trades >= 25% baseline, else 'do not deploy'.",
    ]
    if metrics_json_path is not None and metrics_json_path.exists():
        try:
            j = json.loads(metrics_json_path.read_text())
            lines += [
                "",
                "## Training metrics",
                "",
                "```json",
                json.dumps({k: v for k, v in j.items() if k != "reports"}, indent=2),
                "```",
            ]
        except Exception:
            pass
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--path-c", default="./models_cache/path_c_lgb.joblib")
    ap.add_argument("--path-d-filter",
                     default="./models_cache/path_d_trade_outcome_lgb.joblib")
    ap.add_argument("--path-d-dir-a1",
                     default="./models_cache/path_d_directional_a1_lgb.joblib")
    ap.add_argument("--path-d-dir-a2",
                     default="./models_cache/path_d_directional_a2_lgb.joblib")
    ap.add_argument("--report", default="docs/reports/path_d_trade_outcome_results.md")
    ap.add_argument("--metrics-json", default="models_cache/path_d_metrics.json")
    args = ap.parse_args()

    history = _load(Path(args.csv))
    payload = run_all_modes(
        history,
        path_c_local=args.path_c if Path(args.path_c).exists() else None,
        path_d_filter=args.path_d_filter if Path(args.path_d_filter).exists() else None,
        path_d_dir_a1=args.path_d_dir_a1 if Path(args.path_d_dir_a1).exists() else None,
        path_d_dir_a2=args.path_d_dir_a2 if Path(args.path_d_dir_a2).exists() else None,
    )
    write_report(payload, Path(args.report), Path(args.metrics_json))
    print(json.dumps(payload["results"], indent=2))
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests**

```bash
.venv/bin/pytest tests/test_path_d_eval.py -v
```
Expected: pass.

- [ ] **Step 5: Run full evaluation**

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_path_d.py \
    --csv ./data_long_m15.csv
```
Expected: console prints results dict, `docs/reports/path_d_trade_outcome_results.md` written with all modes + sweep + acceptance line.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_path_d.py tests/test_path_d_eval.py docs/reports/path_d_trade_outcome_results.md
git commit -m "feat(scripts): Path D evaluator with 6 AI modes + 4 non-AI baselines + threshold sweep"
```

---

## Task 11: HANDOFF update + full test sweep

- [ ] **Step 1: Run full test suite for regression check**

```bash
.venv/bin/pytest -q
```
Expected: no NEW failures vs the pre-Path-D baseline (the 2 pre-existing `test_ai_model_filename` interference failures remain unchanged — they are documented in earlier HANDOFF).

- [ ] **Step 2: Append Path D summary to `docs/HANDOFF.md`**

Copy the table from `docs/reports/path_d_trade_outcome_results.md` plus a 3-sentence verdict referencing acceptance criteria (PF vs Path C, kept_trades ≥ 25%, WEAK behaviour).

- [ ] **Step 3: Commit**

```bash
git add docs/HANDOFF.md
git commit -m "docs(handoff): Path D trade-outcome results"
```

---

## Self-Review

**Spec coverage:**
- Path C left intact: §3 (no Path C files modified) ✓
- TP/SL outcome resolver, conservative same-candle rule, MFE/MAE/R: Task 1 ✓
- Sample harvest, baseline-only + synthetic, `is_synthetic`, outcome bookkeeping: Task 3 ✓
- Modes A1/A2/Filter: Task 4 ✓
- TradeFilterModel adapter contract: Task 5 ✓
- HybridPolicy (mode E) pure function: Task 6 ✓
- Engine seam, additive return fields, optional ctor args: Tasks 2 + 7 ✓
- Backtest wiring: Task 8 ✓
- CLI training: Task 9 ✓
- Time-split 70/15/15, threshold sweep on validation only, test once: Task 4 + Task 10 ✓
- 6 AI modes + 4 non-AI baselines (incl. baseline-no-WEAK, STRONG-only, STRONG+NORMAL): Task 10 ✓
- Outcome-class counts + same_candle_conflicts persisted: Task 9 (`outcome_distribution`), Task 10 (report) ✓
- min-kept floor 25%: Task 10 (`pick_best_threshold` + acceptance line in report) ✓
- `tp_used` per row, `LABEL_TP_TARGET` configurable: Task 3 ✓
- `FILTER_UNRESOLVED_POLICY=bad` v1 with separate UNRESOLVED counts: Task 3 (`_filter_label`) ✓
- Tests for every component, no random splits, threshold not picked on test: Tasks 1, 3–6, 8, 10 ✓

**Placeholder scan:** none — every code step contains the actual content; commands have expected outputs.

**Type consistency:** `FilterDecision` enum used in Tasks 5–7 consistently; `HybridDecision`/`HybridThresholds` referenced identically in Tasks 6–8; `Outcome.outcome_class` returns `OutcomeClass` enum (stored as `.value` string in DataFrame, matching test expectations in Task 3); `train_directional` / `train_filter` return tuple `(model, metrics_dict_with_'feature_cols')` consumed by `save_model(model, feature_cols, path)` — consistent.

**Known gap:** Task 11 expects test suite has the same 2 pre-existing failures (`test_ai_model_filename.py` order-dependent), not introduced by this plan. If those get fixed independently, count drops to zero.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-13-path-d-trade-outcome.md`.
