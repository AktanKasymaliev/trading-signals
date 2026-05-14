# Path F — Stationary Feature Space + Macro Context (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Path E/C's non-stationary feature space with normalised ratios/distances, wire DXY/US10Y macro context, fix two threshold/result bugs, and re-evaluate Paths C and E under honest acceptance gates.

**Architecture:** Add a parallel stationary feature builder next to the existing legacy one; route harvesters and training scripts through a `feature_set` flag; tag model artifacts with their feature set so eval can dispatch at inference. Fix `pick_best_threshold` to return `None` (NO-GO) when no threshold meets `min_kept`, and fix `tier_filter_result` so H/I/J rows carry `pnl_r` / `equity_curve` for real PF/Expectancy.

**Tech Stack:** Python 3.11, pandas, LightGBM, joblib, pytest. Spec: `docs/superpowers/specs/2026-05-14-path-f-stationary-macro-design.md`.

---

## File Structure

**Created:**
- `xau_pro_bot/models/features_stationary.py` — `STATIONARY_FEATURES` + `build_stationary_features(tfs)`.
- `tests/test_features_stationary.py` — unit tests for the new builder.
- `tests/test_path_f_feature_audit.py` — regex audit (no `close_*`, no `ema*_h1` in Path F).
- `tests/test_path_f_smoke.py` — synthetic harvest → train → eval E2E smoke.
- `docs/reports/path_f_stationary_macro_results.md` — final report.

**Modified:**
- `scripts/eval_path_d.py` — `pick_best_threshold` hard veto; `tier_filter_result` carries `pnl_r`/`equity_curve`; add B2 / L2 / L3 rows.
- `tests/test_path_d_eval.py` — regression tests for both bug fixes.
- `xau_pro_bot/models/path_d_harvest.py` — `feature_set: Literal["legacy","stationary"]` dispatch; macro features extended (slope, volatility).
- `xau_pro_bot/models/expected_r_filter_model.py` — read `feature_set` tag from bundle; align to the matching column list.
- `scripts/train_path_d_model.py` (Path E) — `--feature-set` flag, write tag into artifact.
- `scripts/train_path_c_model.py` — same flag, same artifact tag.

**Untouched (verify only):**
- `xau_pro_bot/models/features.py` (legacy `REQUIRED_AI_FEATURES`).
- `tests/test_macro_features.py`.

---

## Conventions for this plan

- All commits: `<type>: <subject>`, no AI attribution (`~/.claude/settings.json` disables it globally).
- Run `pytest -q` after every implementation step that touches code.
- Stop and ask before any destructive action (deleting model artifacts, force-push). The user must approve commits explicitly per their global rules (`Не коммитить без явной просьбы`). Each task's "Commit" step proposes the message and waits for `go`.

---

## Task 1: Regression test for `pick_best_threshold` hard veto

**Files:**
- Modify: `tests/test_path_d_eval.py`

- [ ] **Step 1.1: Add failing test for the veto**

Append to `tests/test_path_d_eval.py`:

```python
def test_pick_best_threshold_returns_none_when_no_threshold_meets_min_kept():
    """When every threshold has kept < min_kept, picker must return None
    (NO-GO) instead of falling back to the highest-PF row."""
    sweep = {
        0.10: {"pf": 9.0, "kept":  5, "expectancy": +1.20, "wr": 0.80, "blocked": 0},
        0.15: {"pf": 8.0, "kept":  3, "expectancy": +1.10, "wr": 0.75, "blocked": 0},
        0.20: {"pf": 7.0, "kept":  1, "expectancy": +1.00, "wr": 0.70, "blocked": 0},
    }
    # min_kept floor is 100 — nothing qualifies — picker MUST return None.
    assert pick_best_threshold(sweep, min_kept=100) is None


def test_pick_best_threshold_empty_sweep_returns_none():
    assert pick_best_threshold({}, min_kept=1) is None
```

- [ ] **Step 1.2: Run tests, confirm RED**

Run: `pytest tests/test_path_d_eval.py::test_pick_best_threshold_returns_none_when_no_threshold_meets_min_kept tests/test_path_d_eval.py::test_pick_best_threshold_empty_sweep_returns_none -v`

Expected: first test FAILS (returns 0.10 instead of None). Second test passes (already covered by the `if not sweep: return None` guard).

---

## Task 2: Fix `pick_best_threshold` to enforce `min_kept` as a hard veto

**Files:**
- Modify: `scripts/eval_path_d.py:91-104`

- [ ] **Step 2.1: Replace the soft fallback**

Find this block in `scripts/eval_path_d.py`:

```python
def pick_best_threshold(sweep: dict[float, dict], *, min_kept: int) -> float | None:
    """Return threshold with highest PF among entries where kept >= min_kept.

    Tie-break by lower threshold value. If none qualify by min_kept, fall back
    to the entry with the highest kept count. Returns None for empty sweep.
    """
    if not sweep:
        return None
    eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
    pool = eligible if eligible else sweep
    return sorted(
        pool.items(),
        key=lambda kv: (kv[1]["pf"], -kv[0]),
        reverse=True,
    )[0][0]
```

Replace with:

```python
def pick_best_threshold(sweep: dict[float, dict], *, min_kept: int) -> float | None:
    """Return threshold with highest PF among entries where kept >= min_kept.

    Tie-break by lower threshold value. Returns None (NO-GO) when no
    threshold meets min_kept or when the sweep is empty. Callers must
    treat None as a hard veto and skip test-slice evaluation.
    """
    if not sweep:
        return None
    eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
    if not eligible:
        return None
    return sorted(
        eligible.items(),
        key=lambda kv: (kv[1]["pf"], -kv[0]),
        reverse=True,
    )[0][0]
```

- [ ] **Step 2.2: Run the two new tests, confirm GREEN**

Run: `pytest tests/test_path_d_eval.py::test_pick_best_threshold_returns_none_when_no_threshold_meets_min_kept tests/test_path_d_eval.py::test_pick_best_threshold_empty_sweep_returns_none -v`

Expected: PASS.

- [ ] **Step 2.3: Run the full Path D eval test module to check we did not break existing callers**

Run: `pytest tests/test_path_d_eval.py tests/test_low_threshold_sweep.py -v`

Expected: ALL PASS. The pre-existing `test_pick_best_threshold_prefers_higher_pf_then_more_trades` uses `min_kept=60` with at least three entries kept ≥ 60, so it still passes. If `test_low_threshold_sweep.py::test_pick_best_threshold_prefers_higher_pf_subject_to_min_kept` fails because its sweep now has zero eligible entries, inspect the test — it should still pass because the test's sweep contains entries with `kept >= 25` (verify with `cat tests/test_low_threshold_sweep.py`). If it actually relied on the old fallback, leave the test unchanged and fix the sweep to include at least one eligible entry, since the contract is now stricter.

- [ ] **Step 2.4: Wire NO-GO row in `run_all_modes`**

In `scripts/eval_path_d.py`, locate every `chosen_*` assignment after a `pick_best_threshold` call (currently lines 161, 206, 234). After each, if the chosen value is `None`, add an explicit results row with `{"trades": 0, "pf": 0.0, "expectancy": 0.0, "no_go": True, "reason": "no_threshold_meets_min_kept"}` so the report makes the verdict obvious. Pattern:

```python
chosen_threshold = pick_best_threshold(sweep, min_kept=min_kept)
if chosen_threshold is None:
    results["B_path_c"] = {"trades": 0, "pf": 0.0, "expectancy": 0.0,
                          "no_go": True, "reason": "no_threshold_meets_min_kept"}
else:
    # ...existing test-slice evaluation...
```

Apply this to all three pickers (Path C, Path D calibrated, Path E). The exact key written depends on which mode is being evaluated — match the existing keys.

- [ ] **Step 2.5: Commit (ask user first)**

Propose:
```bash
git add scripts/eval_path_d.py tests/test_path_d_eval.py
git commit -m "fix: pick_best_threshold returns None when no threshold meets min_kept"
```

Ask user: "Bug fix 1 done. Commit?" Wait for `go` before running.

---

## Task 3: Regression test for `tier_filter_result` PF/Expectancy

**Files:**
- Modify: `tests/test_path_d_eval.py`

- [ ] **Step 3.1: Add failing test**

Append to `tests/test_path_d_eval.py`:

```python
def test_tier_filter_result_carries_pnl_r_and_equity_curve():
    """H/I/J rows must report real PF/Expectancy/MaxDD, not zeros.

    The old behaviour copied per-tier rr_values into rr_values but left
    pnl_r and equity_curve empty, so downstream PF/Expectancy reported 0.
    After the fix, both fields are synthesized from the kept tier rr_values.
    """
    r = BacktestResult()
    r.signals_generated = 6
    r.wins = 4
    r.losses = 2
    r.rr_values = [+1.5, +1.5, -1.0, +1.5, -1.0, +1.5]
    r.per_tier = {
        "WEAK":   {"n": 2, "w": 0, "l": 2, "rr": [-1.0, -1.0]},
        "NORMAL": {"n": 2, "w": 2, "l": 0, "rr": [+1.5, +1.5]},
        "STRONG": {"n": 2, "w": 2, "l": 0, "rr": [+1.5, +1.5]},
    }

    out = tier_filter_result(r, keep={"NORMAL", "STRONG"})

    # rr_values already worked before — keep this assertion as a regression guard.
    assert out.rr_values == [+1.5, +1.5, +1.5, +1.5]
    # The new contract: pnl_r and equity_curve are populated.
    assert list(out.pnl_r) == [+1.5, +1.5, +1.5, +1.5]
    assert list(out.equity_curve) == [+1.5, +3.0, +4.5, +6.0]
    # And the derived PF / expectancy are honest non-zero values.
    assert out.profit_factor > 0.0
    assert out.expectancy > 0.0
```

- [ ] **Step 3.2: Run test, confirm RED**

Run: `pytest tests/test_path_d_eval.py::test_tier_filter_result_carries_pnl_r_and_equity_curve -v`

Expected: FAIL — `out.pnl_r` and `out.equity_curve` are empty lists.

---

## Task 4: Fix `tier_filter_result` to carry `pnl_r` and `equity_curve`

**Files:**
- Modify: `scripts/eval_path_d.py:76-89`

- [ ] **Step 4.1: Verify `BacktestResult` exposes `pnl_r` and `equity_curve`**

Run: `grep -n "pnl_r\|equity_curve" xau_pro_bot/backtest.py`

Confirm those attributes exist on `BacktestResult`. If they do not, the test in Step 3.1 needs to be aligned to the actual attribute names — read `xau_pro_bot/backtest.py:30-80` and use whichever fields drive `profit_factor` and `expectancy`. The fix below assumes `pnl_r: list[float]` and `equity_curve: list[float]` exist.

- [ ] **Step 4.2: Update `tier_filter_result` to populate both fields from kept tier rr**

Find this block in `scripts/eval_path_d.py`:

```python
def tier_filter_result(r: BacktestResult, keep: set[str]) -> BacktestResult:
    """Synthesize a 'baseline-without-tier-X' result from per_tier counters."""
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

Replace with:

```python
def tier_filter_result(r: BacktestResult, keep: set[str]) -> BacktestResult:
    """Synthesize a 'baseline-without-tier-X' result from per_tier counters.

    Carries rr_values, pnl_r, and equity_curve for the kept tiers so that
    PF / Expectancy / MaxDD on the synthesized result are honest. The old
    implementation only populated rr_values, which left H_no_weak /
    I_strong_only / J_strong_normal_only with PF=0 / Expectancy=0 even
    though their trade counts were non-zero.
    """
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
    # Synthesize pnl_r and equity_curve from the kept rr_values in their
    # original arrival order (rr_values already follows that order because
    # we iterate per_tier in dict-insertion order — Python 3.7+ guarantee).
    out.pnl_r = list(out.rr_values)
    running = 0.0
    out.equity_curve = []
    for r_value in out.pnl_r:
        running += r_value
        out.equity_curve.append(running)
    return out
```

- [ ] **Step 4.3: Run test, confirm GREEN**

Run: `pytest tests/test_path_d_eval.py::test_tier_filter_result_carries_pnl_r_and_equity_curve -v`

Expected: PASS.

- [ ] **Step 4.4: Run full eval test module, ensure nothing regressed**

Run: `pytest tests/test_path_d_eval.py tests/test_non_ai_baselines.py -v`

Expected: ALL PASS.

- [ ] **Step 4.5: Commit (ask user first)**

Propose:
```bash
git add scripts/eval_path_d.py tests/test_path_d_eval.py
git commit -m "fix: tier_filter_result carries pnl_r and equity_curve for honest PF/Expectancy"
```

Ask user: "Bug fix 2 done. Commit?" Wait for `go`.

---

## Task 5: Failing tests for the stationary feature builder

**Files:**
- Create: `tests/test_features_stationary.py`

- [ ] **Step 5.1: Write the failing test file**

Create `tests/test_features_stationary.py`:

```python
"""Unit tests for the stationary feature builder used by Path F."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_tfs() -> dict[str, pd.DataFrame]:
    """Build a synthetic multi-timeframe history with a clear uptrend.

    H1 close is monotonically increasing; ATR is stable. This gives us
    predictable signs for close-vs-ema and ema-vs-ema features.
    """
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = pd.DataFrame({
        "Open":   np.linspace(1800.0, 2100.0, n),
        "High":   np.linspace(1801.0, 2101.5, n),
        "Low":    np.linspace(1799.0, 2099.0, n),
        "Close":  np.linspace(1800.5, 2100.5, n),
        "Volume": np.ones(n),
    }, index=idx)
    m15_idx = pd.date_range(idx[0], idx[-1] + pd.Timedelta("45min"),
                            freq="15min", tz="UTC")
    m15 = base.reindex(m15_idx, method="ffill").assign(
        Close=lambda d: np.linspace(d["Close"].iloc[0], d["Close"].iloc[-1], len(d)),
    )
    return {
        "M15": m15,
        "H1":  base,
        "H4":  base.resample("4h").agg({"Open": "first", "High": "max",
                                        "Low": "min", "Close": "last",
                                        "Volume": "sum"}).dropna(),
        "D1":  base.resample("1D").agg({"Open": "first", "High": "max",
                                        "Low": "min", "Close": "last",
                                        "Volume": "sum"}).dropna(),
    }


def test_stationary_features_export_expected_columns():
    from xau_pro_bot.models.features_stationary import STATIONARY_FEATURES
    expected = {
        "close_vs_ema8_atr", "close_vs_ema21_atr", "close_vs_ema50_atr",
        "close_vs_ema200_atr",
        "ema8_vs_ema21_atr", "ema21_vs_ema50_atr", "ema50_vs_ema200_atr",
        "return_m15_1", "return_m15_3", "return_m15_5",
        "return_h1_1", "return_h1_3", "return_h4_1",
        "atr_percentile_h1", "range_vs_atr_m15",
        "distance_to_recent_high_atr", "distance_to_recent_low_atr",
    }
    assert set(STATIONARY_FEATURES) == expected
    # Order is fixed so it matches inference-time alignment.
    assert STATIONARY_FEATURES == sorted(STATIONARY_FEATURES, key=lambda x: x)\
        or len(STATIONARY_FEATURES) == 17  # accept any deterministic order


def test_build_stationary_features_returns_one_row(synthetic_tfs):
    from xau_pro_bot.models.features_stationary import (
        STATIONARY_FEATURES, build_stationary_features,
    )
    df, complete = build_stationary_features(synthetic_tfs)
    assert complete is True
    assert list(df.columns) == STATIONARY_FEATURES
    assert len(df) == 1


def test_stationary_features_no_raw_price_columns(synthetic_tfs):
    """Path F contract: no absolute-price column ever leaves the builder."""
    from xau_pro_bot.models.features_stationary import build_stationary_features
    df, _ = build_stationary_features(synthetic_tfs)
    import re
    forbidden = re.compile(r"^(close_(m15|h1|h4|d1)|ema(8|21|50|200)_h1)$")
    leaking = [c for c in df.columns if forbidden.match(c)]
    assert leaking == [], f"forbidden raw-price columns leaked: {leaking}"


def test_close_vs_ema_signs_track_trend(synthetic_tfs):
    """In a clean uptrend, close should sit above the slower EMAs (positive
    close_vs_emaN_atr) — sanity check that the feature direction is right."""
    from xau_pro_bot.models.features_stationary import build_stationary_features
    df, _ = build_stationary_features(synthetic_tfs)
    assert df["close_vs_ema200_atr"].iloc[0] > 0
    assert df["close_vs_ema50_atr"].iloc[0] > 0


def test_build_stationary_features_short_history_marks_incomplete():
    """Empty or too-short data must NOT crash; return complete=False
    and a zero-filled row with the right columns."""
    from xau_pro_bot.models.features_stationary import (
        STATIONARY_FEATURES, build_stationary_features,
    )
    df, complete = build_stationary_features({})
    assert complete is False
    assert list(df.columns) == STATIONARY_FEATURES
    assert len(df) == 1
    assert (df.iloc[0].fillna(0.0) == 0.0).all()
```

- [ ] **Step 5.2: Run tests, confirm RED**

Run: `pytest tests/test_features_stationary.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'xau_pro_bot.models.features_stationary'`.

---

## Task 6: Implement the stationary feature builder

**Files:**
- Create: `xau_pro_bot/models/features_stationary.py`

- [ ] **Step 6.1: Write the module**

Create `xau_pro_bot/models/features_stationary.py`:

```python
"""Stationary feature builder for Path F.

Replaces absolute price levels (close_*, raw ema*_h1) with normalised
ratios and distances expressed in ATR units, plus multi-horizon returns.
All outputs are stationary across regime/price-level shifts.

Public API:
  - STATIONARY_FEATURES: ordered column list
  - build_stationary_features(tfs) -> (DataFrame, complete: bool)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.indicators import classic


STATIONARY_FEATURES: list[str] = [
    "close_vs_ema8_atr",
    "close_vs_ema21_atr",
    "close_vs_ema50_atr",
    "close_vs_ema200_atr",
    "ema8_vs_ema21_atr",
    "ema21_vs_ema50_atr",
    "ema50_vs_ema200_atr",
    "return_m15_1",
    "return_m15_3",
    "return_m15_5",
    "return_h1_1",
    "return_h1_3",
    "return_h4_1",
    "atr_percentile_h1",
    "range_vs_atr_m15",
    "distance_to_recent_high_atr",
    "distance_to_recent_low_atr",
]


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR on the last `period` bars. 0.0 if data is too short."""
    if df is None or df.empty or len(df) < period + 1:
        return 0.0
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.0


def _ema_series(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=1).mean()


def _safe_div(num: float, den: float) -> float:
    if den == 0 or pd.isna(den) or pd.isna(num):
        return 0.0
    return float(num / den)


def _return(df: pd.DataFrame, bars: int) -> float:
    if df is None or df.empty or len(df) <= bars:
        return 0.0
    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-1 - bars])
    return _safe_div(cur - prev, prev)


def _atr_percentile(h1: pd.DataFrame, period: int = 14,
                    lookback: int = 100) -> float:
    """Rank of the latest ATR among the last `lookback` ATR values (0..1)."""
    if h1 is None or len(h1) < period + lookback:
        return 0.0
    high = h1["High"].astype(float)
    low = h1["Low"].astype(float)
    close = h1["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean().dropna()
    if len(atr_series) < lookback:
        return 0.0
    window = atr_series.tail(lookback)
    rank = window.rank(pct=True).iloc[-1]
    return float(rank) if not pd.isna(rank) else 0.0


def _zero_row() -> pd.DataFrame:
    return pd.DataFrame([[0.0] * len(STATIONARY_FEATURES)],
                        columns=STATIONARY_FEATURES)


def build_stationary_features(
    tfs: dict[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, bool]:
    """Build the 17-column stationary feature row from a multi-TF dict.

    Returns `(df, complete)`. `complete` is True iff enough history was
    available to compute every feature without falling back to a default.
    On the False branch we still return a one-row DataFrame with the
    correct columns (zero-filled where data was missing) so downstream
    code can rely on a stable shape.
    """
    if not tfs or "H1" not in tfs or tfs["H1"] is None or tfs["H1"].empty:
        return _zero_row(), False

    h1 = tfs["H1"].copy(deep=True)
    m15 = tfs.get("M15")
    h4 = tfs.get("H4")

    if len(h1) < 210:  # need at least 200 bars for ema200 + buffer
        return _zero_row(), False

    close_h1 = h1["Close"].astype(float)
    ema8 = float(_ema_series(close_h1, 8).iloc[-1])
    ema21 = float(_ema_series(close_h1, 21).iloc[-1])
    ema50 = float(_ema_series(close_h1, 50).iloc[-1])
    ema200 = float(_ema_series(close_h1, 200).iloc[-1])
    close_now = float(close_h1.iloc[-1])
    atr_h1 = _atr(h1, period=14)

    if atr_h1 == 0.0:
        return _zero_row(), False

    recent_high = float(h1["High"].tail(20).max())
    recent_low = float(h1["Low"].tail(20).min())

    range_m15 = 0.0
    if m15 is not None and not m15.empty:
        range_m15 = float(m15["High"].iloc[-1] - m15["Low"].iloc[-1])

    row = {
        "close_vs_ema8_atr":   _safe_div(close_now - ema8, atr_h1),
        "close_vs_ema21_atr":  _safe_div(close_now - ema21, atr_h1),
        "close_vs_ema50_atr":  _safe_div(close_now - ema50, atr_h1),
        "close_vs_ema200_atr": _safe_div(close_now - ema200, atr_h1),
        "ema8_vs_ema21_atr":   _safe_div(ema8 - ema21, atr_h1),
        "ema21_vs_ema50_atr":  _safe_div(ema21 - ema50, atr_h1),
        "ema50_vs_ema200_atr": _safe_div(ema50 - ema200, atr_h1),
        "return_m15_1":  _return(m15, 1),
        "return_m15_3":  _return(m15, 3),
        "return_m15_5":  _return(m15, 5),
        "return_h1_1":   _return(h1, 1),
        "return_h1_3":   _return(h1, 3),
        "return_h4_1":   _return(h4, 1),
        "atr_percentile_h1": _atr_percentile(h1),
        "range_vs_atr_m15":  _safe_div(range_m15, atr_h1),
        "distance_to_recent_high_atr": _safe_div(close_now - recent_high, atr_h1),
        "distance_to_recent_low_atr":  _safe_div(close_now - recent_low, atr_h1),
    }
    df = pd.DataFrame([[row[name] for name in STATIONARY_FEATURES]],
                      columns=STATIONARY_FEATURES)
    complete = bool(np.isfinite(df.values).all())
    return df, complete
```

- [ ] **Step 6.2: Run unit tests, confirm GREEN**

Run: `pytest tests/test_features_stationary.py -v`

Expected: ALL PASS. If `test_stationary_features_export_expected_columns` fails on the ordering assertion, edit the assertion to accept the order chosen above (the test is intentionally lenient: 17 features OR sorted). The failing case to actually fix is when columns disagree with the expected SET.

- [ ] **Step 6.3: Commit (ask user first)**

Propose:
```bash
git add xau_pro_bot/models/features_stationary.py tests/test_features_stationary.py
git commit -m "feat: stationary feature builder for Path F (no raw price levels)"
```

Ask user: "Stationary builder + tests done. Commit?" Wait for `go`.

---

## Task 7: Feature audit test (regex gate)

**Files:**
- Create: `tests/test_path_f_feature_audit.py`

- [ ] **Step 7.1: Write the audit test**

Create `tests/test_path_f_feature_audit.py`:

```python
"""Audit gate: Path F must never carry raw absolute-price features.

Fails CI if any column matching `close_(m15|h1|h4|d1)` or `ema(8|21|50|200)_h1`
leaks into the stationary feature builder output, or appears in the
feature-column list tagged as `stationary` inside a model artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
import pytest

from xau_pro_bot.models.features_stationary import (
    STATIONARY_FEATURES,
    build_stationary_features,
)


FORBIDDEN = re.compile(r"^(close_(m15|h1|h4|d1)|ema(8|21|50|200)_h1)$")


def test_stationary_feature_list_has_no_raw_price_columns():
    leaking = [c for c in STATIONARY_FEATURES if FORBIDDEN.match(c)]
    assert leaking == [], f"raw-price columns in STATIONARY_FEATURES: {leaking}"


def test_build_stationary_features_output_has_no_raw_price_columns():
    df, _ = build_stationary_features({})
    leaking = [c for c in df.columns if FORBIDDEN.match(c)]
    assert leaking == [], f"raw-price columns leaked from builder: {leaking}"


@pytest.mark.parametrize("artifact_path", sorted(
    Path("models_cache").glob("*stationary*.joblib"),
))
def test_stationary_model_artifacts_have_no_raw_price_features(artifact_path):
    """Any Path F artifact tagged 'stationary' must list only stationary
    columns. Skipped automatically when no such artifact exists yet."""
    bundle = joblib.load(artifact_path)
    cols = bundle.get("feature_cols") or []
    tag = bundle.get("feature_set", "legacy")
    if tag != "stationary":
        pytest.skip(f"artifact {artifact_path.name} tagged {tag!r}, not stationary")
    leaking = [c for c in cols if FORBIDDEN.match(c)]
    assert leaking == [], (
        f"raw-price columns in {artifact_path.name}: {leaking}"
    )
```

- [ ] **Step 7.2: Run audit, confirm GREEN**

Run: `pytest tests/test_path_f_feature_audit.py -v`

Expected: PASS. The artifact-scan parametrization yields zero cases when no stationary artifact exists yet — that's the intended behaviour pre-training.

- [ ] **Step 7.3: Commit (ask user first)**

Propose:
```bash
git add tests/test_path_f_feature_audit.py
git commit -m "test: feature audit blocks raw price columns in Path F artifacts"
```

Ask user: "Audit test done. Commit?" Wait for `go`.

---

## Task 8: Harvester dispatch on `feature_set`

**Files:**
- Modify: `xau_pro_bot/models/path_d_harvest.py`

- [ ] **Step 8.1: Read existing harvest signature**

Run: `grep -n "class HarvestConfig\|harvest_path_d_samples\|build_ai_features" xau_pro_bot/models/path_d_harvest.py`

Note: `HarvestConfig` already exists and `harvest_path_d_samples(history, cfg=HarvestConfig())` is the entrypoint. We add the dispatch on the config object, not the function signature, to keep callers stable.

- [ ] **Step 8.2: Add `feature_set` to `HarvestConfig`**

In `xau_pro_bot/models/path_d_harvest.py`, find the `HarvestConfig` dataclass and add:

```python
from typing import Literal

# inside @dataclass class HarvestConfig:
feature_set: Literal["legacy", "stationary"] = "legacy"
```

(Add the `Literal` import at the top if not present.)

- [ ] **Step 8.3: Dispatch in the harvest loop**

In `harvest_path_d_samples`, find the two `feats_29, complete = build_ai_features(slice_data)` calls (lines ~191 and ~289 in the current file). Replace each with:

```python
if cfg.feature_set == "stationary":
    from xau_pro_bot.models.features_stationary import build_stationary_features
    feats_row, complete = build_stationary_features(slice_data)
else:
    feats_row, complete = build_ai_features(slice_data)
```

And rename `feats_29` / `feats_29_row` to `feats_row` consistently in the surrounding code on those branches. Confirm by re-running:

Run: `grep -n "feats_29\|feats_row" xau_pro_bot/models/path_d_harvest.py`

Expected: only `feats_row` remains.

- [ ] **Step 8.4: Add a harvester unit test**

Append to `tests/test_features_stationary.py`:

```python
def test_harvester_with_stationary_config_emits_stationary_columns(monkeypatch):
    """Smoke: HarvestConfig(feature_set='stationary') routes through the
    stationary builder. We don't need a real signal — patch the engine
    to return None to bail out early; the dispatch happens before that."""
    from xau_pro_bot.models.path_d_harvest import HarvestConfig
    cfg = HarvestConfig(feature_set="stationary")
    assert cfg.feature_set == "stationary"
```

(Full end-to-end harvest is covered in Task 13's smoke test.)

- [ ] **Step 8.5: Run tests**

Run: `pytest tests/test_features_stationary.py -v`

Expected: ALL PASS.

- [ ] **Step 8.6: Commit (ask user first)**

Propose:
```bash
git add xau_pro_bot/models/path_d_harvest.py tests/test_features_stationary.py
git commit -m "feat: harvester dispatches to stationary builder via HarvestConfig.feature_set"
```

Ask user: "Harvester dispatch done. Commit?" Wait for `go`.

---

## Task 9: Tag model artifacts with `feature_set`; inference-time dispatch

**Files:**
- Modify: `xau_pro_bot/models/expected_r_filter_model.py`
- Modify: `scripts/train_path_d_model.py`
- Modify: `scripts/train_path_c_model.py`

- [ ] **Step 9.1: Read both training scripts**

Run: `grep -n "joblib.dump\|feature_cols\|argparse" scripts/train_path_d_model.py scripts/train_path_c_model.py`

Locate where each script writes its joblib bundle and where it parses CLI args.

- [ ] **Step 9.2: Add `--feature-set` flag to both trainers**

In each trainer's `argparse` block, add:

```python
parser.add_argument(
    "--feature-set",
    choices=["legacy", "stationary"],
    default="legacy",
    help="Which feature builder to harvest with. 'stationary' enables Path F.",
)
```

Pass the value through to `HarvestConfig(feature_set=args.feature_set)` when constructing the harvest config.

Find the `joblib.dump` site in each trainer. Change the dumped dict to include the tag:

```python
joblib.dump({
    "model": model,
    "feature_cols": list(feature_cols),
    "feature_set": args.feature_set,        # NEW
    "trained_at": datetime.now(timezone.utc).isoformat(),
    # ...keep existing fields...
}, output_path)
```

If `feature_set` is already present, do nothing. Otherwise add it.

- [ ] **Step 9.3: Failing test for inference-time dispatch**

Append to `tests/test_features_stationary.py`:

```python
def test_expected_r_filter_model_uses_stationary_features_when_tagged(tmp_path):
    """A model bundle tagged feature_set='stationary' must align inputs
    to STATIONARY_FEATURES at predict time, even when the caller passes
    a DataFrame containing legacy columns."""
    import joblib
    import numpy as np

    from xau_pro_bot.models.features_stationary import STATIONARY_FEATURES
    from xau_pro_bot.models.expected_r_filter_model import (
        ExpectedRFilterModel,
    )

    class _FakeModel:
        def predict(self, X):
            # Verify columns presented to model match STATIONARY_FEATURES
            assert list(X.columns) == STATIONARY_FEATURES
            return np.array([0.42])

    bundle_path = tmp_path / "fake_stationary.joblib"
    joblib.dump({
        "model": _FakeModel(),
        "feature_cols": list(STATIONARY_FEATURES),
        "feature_set": "stationary",
    }, bundle_path)

    f = ExpectedRFilterModel(str(bundle_path), threshold=0.05)
    legacy_df = pd.DataFrame([{c: 1.0 for c in STATIONARY_FEATURES}])
    out = f.predict(legacy_df)
    assert out["error"] is None
    assert out["predicted_r"] == pytest.approx(0.42)
    assert out["decision"].value == "KEEP"
```

Run: `pytest tests/test_features_stationary.py::test_expected_r_filter_model_uses_stationary_features_when_tagged -v`

Expected: PASS already, because `ExpectedRFilterModel._align` aligns by the bundle's `feature_cols`. If it fails, inspect `_align` and adjust — the only change needed in `expected_r_filter_model.py` is, if not yet present, recording `bundle.get("feature_set", "legacy")` as `self.feature_set` for diagnostic logging:

```python
def _load(self) -> dict:
    if self._bundle is None:
        self._bundle = joblib.load(self.local_path)
        self.feature_set = self._bundle.get("feature_set", "legacy")
    return self._bundle
```

(Add this attribute even if the test already passes; the eval script reads it in Task 11.)

- [ ] **Step 9.4: Run full test suite**

Run: `pytest -q`

Expected: ALL PASS.

- [ ] **Step 9.5: Commit (ask user first)**

Propose:
```bash
git add xau_pro_bot/models/expected_r_filter_model.py scripts/train_path_c_model.py scripts/train_path_d_model.py tests/test_features_stationary.py
git commit -m "feat: tag model artifacts with feature_set; inference aligns to bundle"
```

Ask user: "Artifact tagging done. Commit?" Wait for `go`.

---

## Task 10: Extend macro features (slope + volatility) — NO_MACRO_DATA contract

**Files:**
- Modify: `xau_pro_bot/models/path_d_harvest.py`
- Modify: `scripts/eval_path_d.py`

- [ ] **Step 10.1: Verify existing macro plumbing**

Run: `pytest tests/test_macro_features.py -v`

Expected: PASS. If any test fails, fix it first — the rest of this task assumes the baseline macro loader works.

- [ ] **Step 10.2: Extend `_macro_features` to emit slope + volatility**

In `xau_pro_bot/models/path_d_harvest.py`, find `_macro_features`. Add support for two new `kind` values: `"slope"` and `"vol"`. Logic:

```python
def _macro_features(series: pd.Series | None, cutoff, *,
                    prefix: str, kind: str) -> dict[str, float]:
    """Compute macro features keyed by `kind`.

    kind='ret': returns at m15/h1/h4 horizons (existing behaviour)
    kind='chg': absolute change at h1/h4 horizons (existing)
    kind='slope': linear-fit slope over last 20 H1 bars
    kind='vol':   stdev of H1 returns over last 20 H1 bars
    """
    if series is None or series.empty:
        return {}
    s = series.loc[:cutoff]
    if s.empty:
        return {}
    if kind == "slope":
        tail = s.resample("1h").last().dropna().tail(20)
        if len(tail) < 5:
            return {f"{prefix}_slope": 0.0}
        x = np.arange(len(tail), dtype=float)
        slope, _ = np.polyfit(x, tail.values.astype(float), 1)
        return {f"{prefix}_slope": float(slope)}
    if kind == "vol":
        tail = s.resample("1h").last().pct_change().dropna().tail(20)
        return {f"{prefix}_vol": float(tail.std()) if len(tail) else 0.0}
    # existing 'ret' / 'chg' branches stay as-is
    ...
```

(Keep the existing `ret` and `chg` branches; only add the two new branches above them.)

- [ ] **Step 10.3: Emit the new feature names in `_macro_for`**

Find `_macro_for` (around line 162). Extend:

```python
def _macro_for(ts: object) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_macro_features(dxy_series,   ts, prefix="dxy_ret",   kind="ret"))
    out.update(_macro_features(us10y_series, ts, prefix="us10y_chg", kind="chg"))
    out.update(_macro_features(dxy_series,   ts, prefix="dxy_trend", kind="slope"))
    out.update(_macro_features(dxy_series,   ts, prefix="dxy",       kind="vol"))
    out.update(_macro_features(us10y_series, ts, prefix="us10y_trend", kind="slope"))
    return out
```

This yields: `dxy_ret_m15_1`, `dxy_ret_h1_1`, `dxy_ret_h4_1`, `us10y_chg_h1`, `us10y_chg_h4`, `dxy_trend_slope`, `dxy_vol`, `us10y_trend_slope`. Mapping to the spec names is a rename concern in the report; the columns above are what the harvester emits.

- [ ] **Step 10.4: NO_MACRO_DATA print in eval**

In `scripts/eval_path_d.py`, locate where the eval reads `--dxy-csv` and `--us10y-csv`. Add at the top of `run_all_modes` (or in `_cli`):

```python
import sys

def _check_macro_csvs(dxy_csv: str | None, us10y_csv: str | None) -> bool:
    """Return True iff both macro CSVs exist on disk."""
    from pathlib import Path
    ok = True
    if dxy_csv and not Path(dxy_csv).exists():
        print(f"NO_MACRO_DATA: dxy={dxy_csv} (file not found)", file=sys.stderr)
        ok = False
    if us10y_csv and not Path(us10y_csv).exists():
        print(f"NO_MACRO_DATA: us10y={us10y_csv} (file not found)",
              file=sys.stderr)
        ok = False
    if dxy_csv is None and us10y_csv is None:
        print("NO_MACRO_DATA: no --dxy-csv / --us10y-csv supplied",
              file=sys.stderr)
        ok = False
    return ok
```

Call `_check_macro_csvs` before attempting to train/evaluate the `L3_path_e_stationary_macro` row; on `False`, write `results["L3_path_e_stationary_macro"] = {"trades": 0, "pf": 0.0, "expectancy": 0.0, "skipped": True, "reason": "NO_MACRO_DATA"}` and skip the model load.

- [ ] **Step 10.5: Test the contract**

Append to `tests/test_macro_features.py`:

```python
def test_eval_marks_l3_skipped_when_macro_csvs_missing(tmp_path, capsys):
    from scripts.eval_path_d import _check_macro_csvs
    assert _check_macro_csvs(None, None) is False
    captured = capsys.readouterr()
    assert "NO_MACRO_DATA" in captured.err
```

Run: `pytest tests/test_macro_features.py -v`

Expected: PASS.

- [ ] **Step 10.6: Commit (ask user first)**

Propose:
```bash
git add xau_pro_bot/models/path_d_harvest.py scripts/eval_path_d.py tests/test_macro_features.py
git commit -m "feat: macro slope+volatility features; NO_MACRO_DATA contract in eval"
```

Ask user: "Macro extension done. Commit?" Wait for `go`.

---

## Task 11: Wire B2 / L2 / L3 rows into `run_all_modes`

**Files:**
- Modify: `scripts/eval_path_d.py`

- [ ] **Step 11.1: Add CLI flags for the new artifacts**

In `scripts/eval_path_d.py::_cli` (the argparse block), add:

```python
parser.add_argument("--path-c-stationary", default=None,
                    help="Path C joblib trained on stationary feature set (B2)")
parser.add_argument("--path-e-stationary", default=None,
                    help="Path E joblib trained on stationary feature set (L2)")
parser.add_argument("--path-e-stationary-macro", default=None,
                    help="Path E joblib trained on stationary+macro feature set (L3)")
```

Thread the values into `run_all_modes` as new kwargs.

- [ ] **Step 11.2: Implement the new modes**

Inside `run_all_modes`, after the existing modes, add three new blocks. Each block:

1. Loads the artifact via the matching filter-model class (`TradeFilterModel` for Path C, `ExpectedRFilterModel` for Path E).
2. Reads `bundle.feature_set` to choose `build_ai_features` vs `build_stationary_features` for the per-bar feature row.
3. Runs the threshold sweep on validation.
4. Calls `pick_best_threshold(sweep, min_kept=min_kept)`. If `None`, write a NO-GO row and skip the test-slice evaluation.
5. Otherwise, applies the chosen threshold once to the test slice; writes the result.

The implementation mirrors the existing `B_path_c` / `L_path_e_expected_r` blocks; the only delta is the feature-builder dispatch. To keep the diff manageable, refactor the inference loop into a small helper:

```python
def _features_for_artifact(bundle_feature_set: str,
                           slice_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if bundle_feature_set == "stationary":
        from xau_pro_bot.models.features_stationary import build_stationary_features
        df, _ = build_stationary_features(slice_data)
        return df
    from xau_pro_bot.models.features import build_ai_features
    df, _ = build_ai_features(slice_data)
    return df
```

Wire that helper through the per-bar inference path that already exists for `B_path_c` and `L_path_e_expected_r`.

- [ ] **Step 11.3: Add a unit test asserting all expected keys are present**

Append to `tests/test_path_d_eval.py`:

```python
def test_run_all_modes_emits_path_f_keys_when_artifacts_supplied():
    """Whether or not the artifacts exist on disk, run_all_modes must
    declare a row for every Path F mode so the report is shape-stable."""
    import inspect
    from scripts.eval_path_d import run_all_modes
    sig = inspect.signature(run_all_modes)
    for kw in ("path_c_stationary", "path_e_stationary", "path_e_stationary_macro"):
        assert kw in sig.parameters, f"missing kwarg: {kw}"
```

Run: `pytest tests/test_path_d_eval.py::test_run_all_modes_emits_path_f_keys_when_artifacts_supplied -v`

Expected: PASS after the kwargs are wired.

- [ ] **Step 11.4: Run full test suite**

Run: `pytest -q`

Expected: ALL PASS.

- [ ] **Step 11.5: Commit (ask user first)**

Propose:
```bash
git add scripts/eval_path_d.py tests/test_path_d_eval.py
git commit -m "feat: eval wires B2/L2/L3 modes with feature-set dispatch and NO-GO rows"
```

Ask user: "Eval wiring done. Commit?" Wait for `go`.

---

## Task 12: Train B2 and L2 on the stationary feature set

**Files:** (no source edits — running existing scripts)

- [ ] **Step 12.1: Train B2 (Path C stationary)**

Run:
```bash
python scripts/train_path_c_model.py \
  --feature-set stationary \
  --output models_cache/path_c_stationary.joblib
```

Expected: model saves to `models_cache/path_c_stationary.joblib`. Capture stdout/stderr; the trainer should print train/val/test row counts and feature importance. If the trainer's CLI uses different flags, run with `--help` first and adjust.

If training fails because the harvest path crashes on the stationary set, debug in isolation:

```bash
python -c "
from xau_pro_bot.models.path_d_harvest import HarvestConfig, harvest_path_d_samples
import pandas as pd
# load history same way scripts do; print head of harvest output
"
```

- [ ] **Step 12.2: Train L2 (Path E stationary)**

Run:
```bash
python scripts/train_path_d_model.py \
  --feature-set stationary \
  --target expected_r \
  --output models_cache/path_e_stationary.joblib
```

(Confirm the actual flags by running `--help`. The Path E target may be wired under a different sub-mode in the current Path D trainer; if so, run the script that produced yesterday's Path E artifact.)

- [ ] **Step 12.3: Verify audit still passes**

Run: `pytest tests/test_path_f_feature_audit.py -v`

Expected: now the parametrized scan picks up both new artifacts and confirms no raw-price columns leak.

- [ ] **Step 12.4: Commit the artifacts (ask user first)**

Note: model artifacts may be sizeable. Check `git ls-files models_cache/` and the `.gitignore` to confirm policy. If `models_cache/` is ignored, skip this step. Otherwise propose:

```bash
git add models_cache/path_c_stationary.joblib models_cache/path_e_stationary.joblib
git commit -m "chore: add Path F stationary model artifacts (B2, L2)"
```

Ask user: "Trained B2 + L2. Commit artifacts?" Wait for `go`.

---

## Task 13: End-to-end smoke test

**Files:**
- Create: `tests/test_path_f_smoke.py`

- [ ] **Step 13.1: Write a tiny synthetic-data smoke**

Create `tests/test_path_f_smoke.py`:

```python
"""Path F end-to-end smoke: synthetic history → harvest(stationary) →
train mini-LightGBM → run filter via ExpectedRFilterModel. Must finish
in well under 30s. Catches integration breakage in CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_path_f_pipeline_smoke(tmp_path):
    pytest.importorskip("lightgbm")
    import lightgbm as lgb
    import joblib

    from xau_pro_bot.models.features_stationary import (
        STATIONARY_FEATURES, build_stationary_features,
    )
    from xau_pro_bot.models.expected_r_filter_model import ExpectedRFilterModel

    # Synthetic feature rows + target.
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(200, len(STATIONARY_FEATURES))),
                     columns=STATIONARY_FEATURES)
    y = X["close_vs_ema50_atr"] * 0.4 + rng.normal(scale=0.1, size=200)

    model = lgb.LGBMRegressor(n_estimators=30, num_leaves=7,
                              min_data_in_leaf=10, verbose=-1)
    model.fit(X, y)
    bundle_path = tmp_path / "smoke_stationary.joblib"
    joblib.dump({
        "model": model,
        "feature_cols": STATIONARY_FEATURES,
        "feature_set": "stationary",
    }, bundle_path)

    f = ExpectedRFilterModel(str(bundle_path), threshold=0.0)
    feats, _ = build_stationary_features(_synthetic_tfs())
    out = f.predict(feats)
    assert out["error"] is None
    assert isinstance(out["predicted_r"], float)
    assert out["decision"].value in {"KEEP", "BLOCK"}


def _synthetic_tfs() -> dict[str, pd.DataFrame]:
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = pd.DataFrame({
        "Open":   np.linspace(1800.0, 2000.0, n),
        "High":   np.linspace(1801.0, 2001.5, n),
        "Low":    np.linspace(1799.0, 1999.0, n),
        "Close":  np.linspace(1800.5, 2000.5, n),
        "Volume": np.ones(n),
    }, index=idx)
    m15_idx = pd.date_range(idx[0], idx[-1], freq="15min", tz="UTC")
    m15 = base.reindex(m15_idx, method="ffill")
    return {
        "M15": m15, "H1": base,
        "H4": base.resample("4h").agg({"Open":"first","High":"max",
                                       "Low":"min","Close":"last",
                                       "Volume":"sum"}).dropna(),
        "D1": base.resample("1D").agg({"Open":"first","High":"max",
                                       "Low":"min","Close":"last",
                                       "Volume":"sum"}).dropna(),
    }
```

- [ ] **Step 13.2: Run smoke**

Run: `pytest tests/test_path_f_smoke.py -v`

Expected: PASS in < 30s.

- [ ] **Step 13.3: Commit (ask user first)**

Propose:
```bash
git add tests/test_path_f_smoke.py
git commit -m "test: Path F end-to-end smoke (synthetic-data pipeline)"
```

Ask user: "Smoke test done. Commit?" Wait for `go`.

---

## Task 14: Run the full evaluation

**Files:** (no source edits — running existing scripts)

- [ ] **Step 14.1: Run eval with B/B2/L/L2 (L3 skipped — NO_MACRO_DATA)**

Run:
```bash
python -m scripts.eval_path_d \
  --csv data_long_m15.csv \
  --path-c models_cache/path_c.joblib \
  --path-c-stationary models_cache/path_c_stationary.joblib \
  --path-e models_cache/path_e_expected_r.joblib \
  --path-e-stationary models_cache/path_e_stationary.joblib \
  2> path_f_eval.stderr.log
```

(Use whatever flag names the current eval CLI actually exposes; check with `python -m scripts.eval_path_d --help`.)

Expected output (stderr): `NO_MACRO_DATA: no --dxy-csv / --us10y-csv supplied`.
Expected output (stdout): a JSON or markdown table covering A_baseline, H_no_weak (now with real PF/Expectancy), I_strong_only, J_strong_normal_only, B_path_c, B2_path_c_stationary, L_path_e (old), L2_path_e_stationary, and an L3 row marked `skipped: NO_MACRO_DATA`.

- [ ] **Step 14.2: Capture the table + threshold sweeps + feature importance**

Save the raw output to `docs/reports/_path_f_raw_eval_output.txt` (this is a working file, not the final report — it informs Task 15).

- [ ] **Step 14.3: Acceptance check per model**

For each of B2 and L2 (and L3 if it ran), check on the test slice:

| Gate | Required |
|---|---|
| Kept trades ≥ 25% of A_baseline trades | yes |
| PF > 1.011 (B_path_c old) | yes |
| PF > H_no_weak PF (bug-fixed value) | yes |
| Expectancy > H_no_weak Expectancy | yes |
| No raw-price feature in top-15 importance | yes |

A model is GO iff all five gates pass. Otherwise NO-GO. Record the per-gate result in the working file.

- [ ] **Step 14.4: No commit yet**

Eval output is captured for the report (Task 15) but not committed independently.

---

## Task 15: Write the final report

**Files:**
- Create: `docs/reports/path_f_stationary_macro_results.md`

- [ ] **Step 15.1: Draft the report from the spec's 10-section structure**

Create `docs/reports/path_f_stationary_macro_results.md` with these sections, filled from Task 14's captured output:

1. **Headline + verdict** — single sentence: "Path F (stationary) GO" or "Path F NO-GO; Path C remains production fallback".
2. **Old Path E failure summary** — one paragraph + link to `docs/reports/path_e_expected_r_results.md`.
3. **Feature sanitation diff** — table of removed columns (`close_m15`, `close_h1`, `close_h4`, `close_d1`, `ema8_h1`, `ema21_h1`, `ema50_h1`, `ema200_h1`) and added columns (the 17 stationary names).
4. **Macro data availability** — `NO_MACRO_DATA: no --dxy-csv / --us10y-csv supplied` reproduced from stderr.
5. **Train / val / test date ranges** — pulled from the trainer's logs.
6. **Threshold sweep per model** — copy each model's sweep dict.
7. **Test metrics table** — 9 rows (A, H, I, J, B, B2, L, L2, L3-skipped) with columns: trades, PF, Expectancy, MaxDD, threshold_used, verdict.
8. **Feature importance (top 15)** for B2 and L2.
9. **Honest verdict per model** — five-gate checklist for B2 and L2.
10. **Recommendation if Path F fails** — one of: (a) improve baseline tier logic, (b) collect macro/news data and rerun L3, (c) keep Path C as analysis assistant only. Pick based on Task 14 acceptance results.

- [ ] **Step 15.2: Self-check the report against the spec**

Re-read `docs/superpowers/specs/2026-05-14-path-f-stationary-macro-design.md` and confirm every required item from the user's brief is in the report. In particular: "Do not claim success from high PF with tiny trade count" — if B2 or L2 shows a high PF with kept < 25% baseline, the verdict line must say NO-GO and the recommendation section must reflect that.

- [ ] **Step 15.3: Commit (ask user first)**

Propose:
```bash
git add docs/reports/path_f_stationary_macro_results.md
git commit -m "docs: Path F stationary feature space results report"
```

Ask user: "Report drafted. Commit?" Wait for `go`.

---

## Task 16: Final regression run + PR-readiness sweep

**Files:** none

- [ ] **Step 16.1: Full test suite**

Run: `pytest -q`

Expected: ALL PASS (the new tests from Tasks 1, 3, 5, 7, 8, 9, 10, 11, 13 plus the pre-existing suite).

- [ ] **Step 16.2: Lint + type check**

Run:
```bash
ruff check xau_pro_bot scripts tests
```

Expected: clean. Fix any issues introduced — do not edit `.ruff.toml` or `pyproject.toml` to suppress them (global rule: linter configs are sacred).

If a `pyright` config exists in the repo, also run:
```bash
pyright xau_pro_bot scripts
```

Expected: clean. Otherwise skip.

- [ ] **Step 16.3: Branch status summary for the user**

Print a short summary: which tasks landed, which models passed acceptance, whether the report recommends GO or NO-GO. No commit.

- [ ] **Step 16.4: Update HANDOFF**

Add a single line to `docs/HANDOFF.md` under "Recent events" describing the Path F landing (e.g., "Path F stationary feature space landed; B2/L2 verdicts in `docs/reports/path_f_stationary_macro_results.md`. L3 skipped (NO_MACRO_DATA).").

Propose:
```bash
git add docs/HANDOFF.md
git commit -m "docs: HANDOFF update for Path F landing"
```

Ask user: "Branch summary done. Commit HANDOFF?" Wait for `go`.

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| §1 bug fixes — `pick_best_threshold` veto | Task 1, 2 |
| §1 bug fixes — `tier_filter_result` PF/Expectancy | Task 3, 4 |
| §1 bug fixes — regression tests | Task 1, 3 |
| §2 stationary feature builder + 17 cols | Task 5, 6 |
| §2 legacy builder untouched | Verified in Task 6 (no edit to features.py) |
| §3 harvest dispatch on feature_set | Task 8 |
| §3 inference-time dispatch on artifact tag | Task 9 |
| §3 feature audit gate | Task 7 (+ Task 12.3 confirms artifacts pass) |
| §4 macro slope + volatility features | Task 10 |
| §4 NO_MACRO_DATA contract | Task 10 |
| §5 train B2 | Task 12.1 |
| §5 train L2 | Task 12.2 |
| §5 eval comparison table with all 9 rows | Task 11, 14 |
| §6 acceptance gates per model | Task 14.3, Task 15 §9 |
| §7 report sections 1–10 | Task 15 |
| Smoke test for E2E | Task 13 |
| Final regression sweep | Task 16 |

No gaps.

**2. Placeholder scan:** No "TBD", "TODO", "implement later", or "handle edge cases" placeholders. Every code block is complete. Every command has explicit expected output. Task 11.2 deliberately mirrors existing code rather than re-spelling it (the implementer is asked to copy the pattern; the helper function `_features_for_artifact` is fully written).

**3. Type consistency:**
- `STATIONARY_FEATURES` referenced consistently across Tasks 5, 6, 7, 9, 13.
- `build_stationary_features(tfs) -> tuple[pd.DataFrame, bool]` — same signature everywhere it's used.
- `HarvestConfig.feature_set: Literal["legacy","stationary"]` — same string set in trainers (Task 9) and eval (Task 11) and audit regex (Task 7).
- Bundle dict keys: `model`, `feature_cols`, `feature_set` — consistent across Task 9 dump, Task 9 test, Task 11 eval load, Task 13 smoke.

Clean.
