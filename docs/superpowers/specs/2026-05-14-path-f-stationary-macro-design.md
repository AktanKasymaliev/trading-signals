# Path F — Stationary Feature Space + Macro Context (Design)

**Date:** 2026-05-14
**Branch:** `feature/hugging-face-ai-layer`
**Predecessor:** Path E (NO-GO, see `docs/reports/path_e_expected_r_results.md`)
**Status:** Approved for planning

## Context

Path E (Expected R regressor) collapsed on test: 0/391 trades kept after a validation threshold that admitted only 27/~98-floor trades. Two root causes surfaced:

1. **Process bugs masked the failure on validation.**
   - `pick_best_threshold` (`scripts/eval_path_d.py:91-104`) silently fell back to "highest PF in sweep" when no threshold met `min_kept`, instead of returning `None` and marking NO-GO.
   - `tier_filter_result` (`scripts/eval_path_d.py:76-89`) does not surface `pnl_r` / `equity_curve` for `H_no_weak` / `I_strong_only` / `J_strong_normal_only` rows, leaving PF=0 / Expectancy=0 in the comparison table.

2. **Feature space is non-stationary.** Path E feature importance is dominated by absolute price levels (`close_d1`, `close_m15`) and price-space EMAs (`ema8_h1`, `ema200_h1`, …). Across the time split, gold's price level shifts; the model memorised the train regime rather than learning regime-invariant structure.

Path F replaces the feature space with stationary ratios/distances/returns and wires DXY / US10Y macro context. Path C remains the production fallback until Path F beats PF 1.011 on test.

## Goals

1. Eliminate non-stationary absolute price columns from Path F model inputs.
2. Provide a normalised feature set: close-vs-EMA in ATR units, EMA-vs-EMA in ATR units, multi-horizon returns, ATR percentile, range-vs-ATR, distance-to-recent-high/low in ATR units.
3. Wire macro plumbing for DXY / US10Y; surface `NO_MACRO_DATA` clearly when CSVs are absent.
4. Make `pick_best_threshold` a hard veto on `min_kept`.
5. Make `tier_filter_result` carry `pnl_r` / `equity_curve` so the H/I/J baselines have honest PF / Expectancy.
6. Re-evaluate Path C and Path E on the cleaned feature set.
7. Keep Path C (PF 1.011) as fallback until Path F beats it on all acceptance gates.

## Non-goals (out of scope for this iteration)

- News / sentiment / order-flow data.
- Ensembles of Path C + Path E.
- Retraining the old `REQUIRED_AI_FEATURES` set; it stays alive only for the `L_path_e_old` / `B_path_c_old` baseline comparison.
- Fabricated or auto-fetched macro data. If `--dxy-csv` / `--us10y-csv` paths are missing, L3 is skipped with `NO_MACRO_DATA` and we move on.

## Architecture

### 1. Bug fixes (must land first)

**`pick_best_threshold` hard veto** (`scripts/eval_path_d.py:91-104`)

Replace the soft fallback:

```python
eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
pool = eligible if eligible else sweep
```

with:

```python
eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
if not eligible:
    return None
pool = eligible
```

Callers in `run_all_modes` (`scripts/eval_path_d.py:161, 206, 234`) already handle `None` (they skip the model). Add an explicit NO-GO row to the results table so the report makes the verdict obvious.

**`tier_filter_result` carries PF / Expectancy** (`scripts/eval_path_d.py:76-89`)

The function already copies `rr_values` per tier. The bug is downstream: `_result_summary` reads `pnl_r` / `equity_curve` (or recomputes from `rr_values` *of trades that were kept*). Fix by populating both fields from the kept tier `rr_values` so PF, Expectancy, MaxDD report real numbers. Path of least change: synthesize `pnl_r` and `equity_curve` from `rr_values` inside `tier_filter_result` itself.

**Regression tests** (`tests/test_path_d_eval.py`):
- `pick_best_threshold` returns `None` when no threshold meets `min_kept`.
- `tier_filter_result` output has non-empty `pnl_r` and `equity_curve` whose PF matches the PF computed from `rr_values` directly.

### 2. Stationary feature builder

**New module:** `xau_pro_bot/models/features_stationary.py`

Exports:
- `STATIONARY_FEATURES: list[str]` — ordered list of 16 features.
- `build_stationary_features(tfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, bool]` — same shape contract as `build_ai_features`.

**Feature list** (all stationary, no absolute price levels):

```
close_vs_ema8_atr
close_vs_ema21_atr
close_vs_ema50_atr
close_vs_ema200_atr
ema8_vs_ema21_atr
ema21_vs_ema50_atr
ema50_vs_ema200_atr
return_m15_1
return_m15_3
return_m15_5
return_h1_1
return_h1_3
return_h4_1
atr_percentile_h1
range_vs_atr_m15
distance_to_recent_high_atr   # to H1 high-20
distance_to_recent_low_atr    # to H1 low-20
```

Formulas:
- `close_vs_emaN_atr = (close_h1 - emaN_h1) / atr_h1` (signed)
- `emaA_vs_emaB_atr = (emaA_h1 - emaB_h1) / atr_h1` (signed)
- `return_*_K = (close_t - close_{t-K}) / close_{t-K}` (relative return)
- `atr_percentile_h1` — already implemented in `_baseline_context_features`; reuse.
- `range_vs_atr_m15` — already implemented; reuse.
- `distance_to_recent_*_atr = (close - rolling_max/min_h1_20) / atr_h1`

Defensive defaults: 0.0 when ATR is 0 or NaN, or when the lookback window is too short.

**Backward compatibility:**
- `xau_pro_bot/models/features.py` and `REQUIRED_AI_FEATURES` stay unchanged.
- `build_ai_features` still works for Path C old, Path E old.

### 3. Harvest dispatch

Add `feature_set: Literal["legacy", "stationary"] = "legacy"` to:
- `xau_pro_bot/models/path_d_harvest.py::harvest_path_d_samples`
- Any other entrypoints used by `scripts/train_path_d_model.py` / `scripts/train_path_c_model.py`.

Inside harvest, dispatch to `build_ai_features` vs `build_stationary_features`. The 29-column legacy header is preserved when `feature_set="legacy"`. New rows for `stationary` carry the 16 stationary cols (plus baseline context cols already added by `_baseline_context_features` — those are already stationary and stay).

### 4. Macro features (NO_MACRO_DATA path)

Plumbing already exists from prior work: `--dxy-csv` / `--us10y-csv` flags, `_macro_features` helper, `tests/test_macro_features.py`. Verify and extend.

**Required macro columns on the final harvest row** (added when CSVs supplied):

```
dxy_return_m15_1
dxy_return_h1_1
dxy_return_h4_1
dxy_trend_slope          # linear fit slope over 20 H1 bars
dxy_volatility           # stdev of dxy returns over 20 H1 bars
us10y_change_h1
us10y_change_h4
us10y_trend_slope        # linear fit slope over 20 H1 bars
```

**Missing-data behaviour:** If a CSV path is required by the eval mode (L3) but the file is missing or empty, `scripts/eval_path_d.py` prints `NO_MACRO_DATA: dxy=<path> us10y=<path>` on stderr, skips the L3 row, and continues. No silent fallback. No zeroes substituted.

**Alignment:** macro series are loaded on each CSV's native timestamp and forward-filled onto XAU M15 bars (already implemented).

### 5. Feature audit

**New test:** `tests/test_path_f_feature_audit.py`

- For any model loaded under a Path F flag (`stationary` feature set), its trained feature columns must NOT contain any column matching the regex `^close_(m15|h1|h4|d1)$` or `^ema(8|21|50|200)_h1$`.
- Feature importance dumps (top-15) for Path F models must not include any column matching those regexes.
- A unit-level check on `build_stationary_features` output: no column starts with `close_` or matches raw `ema*_h1`.

The audit is a regular pytest test (not a script) so CI catches a feature regression instantly.

### 6. Training scripts

- `scripts/train_path_c_model.py` — add `--feature-set {legacy,stationary}` flag, default `legacy`. When `stationary`, harvest with the new set and train Path C as B2.
- `scripts/train_path_d_model.py` (Path E lives here too) — add the same flag. When `stationary`, train L2.
- Both write the feature-set tag into the model artifact so `eval_path_d.py` can dispatch to the right feature builder at inference time.

### 7. Eval comparison table

`scripts/eval_path_d.py::run_all_modes` produces a single results table with these rows:

| key | description |
|---|---|
| `A_baseline`             | engine output, no filter |
| `H_no_weak`              | tier filter (NORMAL + STRONG), PF/Expectancy now real after bug fix |
| `I_strong_only`          | tier filter (STRONG), PF/Expectancy real |
| `J_strong_normal_only`   | duplicate of H_no_weak, kept for continuity |
| `B_path_c_old`           | Path C with legacy features (current PF 1.011 baseline) |
| `B2_path_c_stationary`   | Path C retrained on stationary set |
| `L_path_e_old`           | Path E with legacy features (the NO-GO from yesterday) |
| `L2_path_e_stationary`   | Path E with stationary set |
| `L3_path_e_stationary_macro` | L2 + macro features; skipped if NO_MACRO_DATA |

Threshold selection: validation only, via fixed `pick_best_threshold`. Test slice touched once per model. If the picker returns `None`, the row is logged as NO-GO and not evaluated on test.

### 8. Acceptance gates

A Path F model is **GO** iff, on the test slice:

- PF > 1.011 (Path C current baseline), **and**
- PF > `H_no_weak` PF (with the bug-fixed value), **and**
- Expectancy > `H_no_weak` Expectancy, **and**
- Kept trades ≥ 25% of `A_baseline` trades, **and**
- No raw price feature (`close_*`, raw `ema*_h1`) appears in top-15 importance.

Otherwise **NO-GO**. We do not interpret high PF on tiny trade counts.

### 9. Report

`docs/reports/path_f_stationary_macro_results.md` — sections:

1. Headline + verdict.
2. Old Path E failure summary (link to predecessor report).
3. Feature sanitation diff (removed cols, added cols).
4. Macro data availability (paths checked, NO_MACRO_DATA if missing).
5. Train / val / test date ranges.
6. Threshold sweep per model.
7. Test metrics table (all 8–9 rows).
8. Feature importance (top-15) for B2, L2, and L3 if run.
9. Honest verdict per model against the acceptance gates.
10. If Path F fails: recommend (a) improve baseline tier logic, (b) collect macro/news data, or (c) keep Path C as analysis assistant only.

## Data flow

```
raw H1/M15/H4/D1 CSVs ──► harvester (feature_set=stationary)
                              │
                              ├── build_stationary_features (16 cols)
                              ├── _baseline_context_features  (already stationary)
                              ├── _macro_features (if CSVs supplied)
                              └── label = final_R
                              │
                              ▼
                        train_path_c_model.py / train_path_d_model.py
                              │
                              ▼
                        model artifact (with feature_set tag)
                              │
                              ▼
                        eval_path_d.py
                              ├── reload data
                              ├── dispatch to matching feature builder
                              ├── pick_best_threshold (val only)
                              ├── apply on test (once)
                              └── results table + NO-GO rows
                              │
                              ▼
                        docs/reports/path_f_stationary_macro_results.md
```

## Error handling

- Missing macro CSV → `NO_MACRO_DATA` stderr, L3 row skipped, eval continues.
- `pick_best_threshold` returns `None` → caller logs NO-GO row, skips test evaluation.
- ATR=0 or insufficient history → stationary feature defaults to 0.0 (logged once per harvest run with a count).
- Audit test failure → CI red. Hard stop.

## Testing strategy

1. **Bug-fix regression tests** (TDD-first, in `tests/test_path_d_eval.py`):
   - `pick_best_threshold` veto.
   - `tier_filter_result` PF/Expectancy.

2. **Stationary feature unit tests** (`tests/test_features_stationary.py`):
   - Output column order matches `STATIONARY_FEATURES`.
   - No `close_*` or raw `ema*_h1` columns leak.
   - Defensive defaults for short / empty history.
   - Sample-driven sanity: with synthetic OHLCV where close>ema, `close_vs_emaN_atr > 0`.

3. **Feature audit** (`tests/test_path_f_feature_audit.py`):
   - Stationary feature set rejects any raw-price regex match.
   - Models tagged `stationary` reject any raw-price column at inference.

4. **Macro alignment** — verify existing `tests/test_macro_features.py` still passes; extend if new slope/volatility features need coverage.

5. **End-to-end smoke** — `pytest -v` green; one synthetic harvest → train → eval cycle in `tests/test_path_f_smoke.py` that runs in < 30s.

## Acceptance criteria (project-level, not just per-model)

- All bug-fix regression tests pass.
- Stationary feature audit passes.
- `pytest -v` green.
- Report `docs/reports/path_f_stationary_macro_results.md` exists with all 10 sections filled in honestly.
- If at least one Path F model passes the per-model acceptance gates: PR-ready commit "feat: Path F stationary feature space".
- If all Path F models fail: the report recommends next steps; Path C stays as the production fallback.

## Open questions / risks

- **Stationary set may still be insufficient.** If the lift on test is < Path C, the issue is likely deeper (label policy, harvest cadence, or genuinely low signal-to-noise on XAU M15). The report's recommendation section addresses this branch.
- **Path C retrain may regress.** B2 uses the same model class on a smaller feature space. Possible PF dip; explicitly tolerated as long as B2 stays as a comparison row, not a replacement.
- **Macro CSVs eventually need a sourcing decision.** Out of scope for Path F; the missing-data path documents what's needed.
