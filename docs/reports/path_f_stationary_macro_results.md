# Path F — Stationary Feature Space + Macro Context

**Date:** 2026-05-15
**Branch:** `feature/hugging-face-ai-layer`
**Predecessor:** Path E (NO-GO, `docs/reports/path_e_expected_r_results.md`)
**Spec:** `docs/superpowers/specs/2026-05-14-path-f-stationary-macro-design.md`
**Plan:** `docs/superpowers/plans/2026-05-14-path-f-stationary-macro.md`

## 1. Headline / Verdict

- **B2_path_c_stationary: marginal GO.** PF 1.051 vs current legacy Path C PF 1.109 — *below* the live re-run baseline; PF 1.011 vs the predecessor report — *above*. Trade count 198 / 303 baseline (65%) clears the 25% floor. Expectancy +0.019 is barely positive. Honest read: B2 demonstrates the stationary feature space works without regressing, but does not unambiguously beat legacy Path C.
- **L2_path_e_stationary: NO-GO on the kept-trades floor.** PF 1.699 and Expectancy +0.255 on the test slice are the most attractive numbers in the comparison table, but the model keeps only 52 of 303 baseline test trades (17%) vs the 25% floor (76 trades). Per the spec: "Do not claim success from high PF with tiny trade count." The threshold-veto fix correctly accepted L2 on validation (kept 77 ≥ 76 at predicted_R = 0.05), but the same threshold leaves only 52 keepers on test.
- **L3_path_e_stationary_macro: skipped (NO_MACRO_DATA).** DXY / US10Y CSVs not supplied. Plumbing exists; rerun once data lands.

**Production fallback:** Path C legacy stays in use. B2 is a viable swap candidate but the lift is marginal; recommend collecting macro CSVs and rerunning L3 before promoting B2.

## 2. Old Path E failure summary

Path E (`path_e_expected_r_lgb.joblib`, legacy 29-feature set):

- Validation slice: PF 8.999, Expectancy +1.185, WR 0.818 across 27 kept trades at `predicted_R > 0.15`.
- Test slice: same threshold blocked 391/391 trades.
- Top-15 feature importance dominated by `close_d1`, `close_m15`, `ema8_h1`, `ema21_h1`, `ema50_h1`, `ema200_h1`. Across the 70/15/15 time split gold's price level shifts, so the model memorised the train regime instead of learning regime-invariant structure.

Two process bugs surfaced and were fixed in Path F:

- `pick_best_threshold` silently fell back to "highest PF in entire sweep" when no threshold met `min_kept` (commit `8ddd6e7` — now returns `None` → NO-GO row). **Observed in Path F eval**: `L_path_e_expected_r` row is correctly marked `no_go: true, reason: "no_threshold_meets_min_kept"` instead of degenerating into a 0/0/0 zero row.
- `tier_filter_result` did not surface `pnl_r` / `equity_curve` for H/I/J rows (commit `8ddd6e7` — now synthesized from kept-tier `rr_values`). **Observed in Path F eval**: `H_no_weak` reports PF 0.873 / Expectancy -0.088 / WR 0.310 across 71 trades, instead of the previous 0.000 / 0.000 / 0.000.

Full predecessor verdict: `docs/reports/path_e_expected_r_results.md`.

## 3. Feature sanitation diff

### Removed from Path F input (raw price levels, non-stationary)

| Column | Source builder |
|---|---|
| `close_m15`   | `build_ai_features` |
| `close_h1`    | `build_ai_features` |
| `close_h4`    | `build_ai_features` |
| `close_d1`    | `build_ai_features` |
| `ema8_h1`     | `build_ai_features` |
| `ema21_h1`    | `build_ai_features` |
| `ema50_h1`    | `build_ai_features` |
| `ema200_h1`   | `build_ai_features` |

### Added in Path F input (`STATIONARY_FEATURES`, 17 normalised columns)

| Column | Formula |
|---|---|
| `close_vs_ema8_atr`   | `(close - ema8) / atr_h1`   |
| `close_vs_ema21_atr`  | `(close - ema21) / atr_h1`  |
| `close_vs_ema50_atr`  | `(close - ema50) / atr_h1`  |
| `close_vs_ema200_atr` | `(close - ema200) / atr_h1` |
| `ema8_vs_ema21_atr`   | `(ema8 - ema21) / atr_h1`   |
| `ema21_vs_ema50_atr`  | `(ema21 - ema50) / atr_h1`  |
| `ema50_vs_ema200_atr` | `(ema50 - ema200) / atr_h1` |
| `return_m15_1` / `return_m15_3` / `return_m15_5` | M15 pct returns at 1 / 3 / 5 bars |
| `return_h1_1`  / `return_h1_3`  | H1 pct returns at 1 / 3 bars |
| `return_h4_1`  | H4 pct return at 1 bar |
| `atr_percentile_h1` | rank of latest ATR among the last 100 H1 ATR values (0..1) |
| `range_vs_atr_m15` | latest M15 high-low range divided by H1 ATR |
| `distance_to_recent_high_atr` | `(close - max(H1.High[-20:])) / atr_h1` (≤ 0) |
| `distance_to_recent_low_atr`  | `(close - min(H1.Low[-20:])) / atr_h1`  (≥ 0) |

Audit gate (`tests/test_path_f_feature_audit.py`) blocks any column matching `^close_(m15|h1|h4|d1)$` or `^ema(8|21|50|200)_h1$` in `STATIONARY_FEATURES`, in `build_stationary_features` output, or in any model artifact tagged `feature_set="stationary"`. The audit passes for all five Path F stationary artifacts written by T12.

## 4. Macro data availability

- DXY CSV: **not supplied** (`--dxy-csv` absent).
- US10Y CSV: **not supplied** (`--us10y-csv` absent).
- L3_path_e_stationary_macro: **skipped (NO_MACRO_DATA)**. Eval stderr: `NO_MACRO_DATA: no --dxy-csv / --us10y-csv supplied`.

Macro plumbing is complete and tested:

- `_macro_features(series, ts, kind={"ret","chg","slope","vol"})` emits `dxy_ret_1h`, `dxy_ret_4h`, `dxy_trend_slope`, `dxy_vol`, `us10y_chg_1h`, `us10y_chg_4h`, `us10y_trend_slope` when CSVs are present.
- `_check_macro_csvs` enforces the no-silent-fallback contract.
- `tests/test_macro_features.py` covers both "CSV missing → no columns" and "CSV present → columns populated with no NaN" branches.

Once CSVs are sourced (Stooq, FRED, or similar) L3 can run without further code changes.

## 5. Train / Val / Test date ranges

- **Source dataset:** `data_long_m15.csv` — 80,210 M15 bars from `2022-01-03 01:00:00+00:00` to `2025-09-30 19:30:00+00:00` (~2.5 years).
- **B2 training (`scripts/train_path_c_model.py --feature-set stationary`):** dataset = 9,920 rows (step=8, horizon=16, threshold=0.003); class distribution 0:6990, 1:1564, -1:1366; 80/20 train/test split → 7,936 train / 1,984 test.
- **L2 training (`scripts/train_path_d_model.py --feature-set stationary --expected-r`):** harvest = 2,945 rows (1,707 baseline + 1,238 synthetic), step_h1=4; outcome distribution SL 55.2% / TP 24.1% / UNRESOLVED 20.6% / SAME_CANDLE_SL_FIRST 0.1%. 70/15/15 time split.
- **Eval (`scripts/eval_path_d`):** validation = 70% → 85% of H1 index, test = 85% → 100%. Identical to Path E so the comparison is like-for-like.

## 6. Threshold sweep per model

B2 is a directional classifier; no threshold to sweep (gated on engine tier through `HFTradingModel`).

L2_path_e_stationary sweep on validation slice (`EXPECTED_R_THRESHOLDS = (0.00, 0.03, 0.05, 0.10, 0.15)`):

| predicted_R ≥ | PF | Expectancy | WR | kept | blocked |
|--:|--:|--:|--:|--:|--:|
| 0.00 | 2.000 | +0.337 | 0.500 | 95 | 236 |
| 0.03 | 2.148 | +0.365 | 0.518 | 85 | 246 |
| **0.05** | **2.636** | **+0.468** | **0.569** | **77** | **254** |
| 0.10 | 3.286 | +0.593 | 0.622 | 54 | 277 |
| 0.15 | 3.636 | +0.707 | 0.645 | 41 | 290 |

Chosen on validation: **0.05** (highest PF where `kept ≥ min_kept` = 76; the 0.10 and 0.15 rows fail the floor on validation already). Applied once to test.

The L_path_e_expected_r (legacy) sweep produced zero kept on every threshold ≥ 0.00; picker correctly returned `None` and the row is marked NO-GO.

## 7. Test metrics table

| mode | trades | PF | Expectancy | WR | MaxDD | Threshold | Verdict |
|------|------:|---:|-----------:|---:|------:|---------:|:-------:|
| A_baseline | 303 | 0.809 | -0.076 | 0.298 | 33.28 | — | reference |
| H_no_weak  | 71 | 0.873 | -0.088 | 0.310 | 15.00 | — | reference (bug-fixed) |
| I_strong_only | 6 | 1.492 | +0.246 | 0.500 | 1.00 | — | reference (bug-fixed) |
| J_strong_normal_only | 71 | 0.873 | -0.088 | 0.310 | 15.00 | — | duplicate of H |
| B_path_c (legacy) | 161 | 1.109 | +0.041 | 0.362 | 10.02 | — | current production |
| E_path_d_filter | 264 | 0.884 | -0.045 | 0.318 | 24.19 | 0.20 | reference |
| L_path_e_expected_r (legacy) | 0 | — | — | — | — | — | **NO-GO** (no_threshold_meets_min_kept) |
| **B2_path_c_stationary** | 198 | 1.051 | +0.019 | 0.354 | 13.81 | — | **GO (marginal)** |
| **L2_path_e_stationary** | 52 | 1.699 | +0.255 | 0.472 | 6.91 | 0.05 | **NO-GO** (kept < 25% floor) |
| L3_path_e_stationary_macro | — | — | — | — | — | — | **skipped (NO_MACRO_DATA)** |

H_no_weak / I_strong_only / J_strong_normal_only now carry real PF / Expectancy / MaxDD (commit `8ddd6e7`).

## 8. Feature importance (top 15)

### B2_path_c_stationary (gain or split, from LGBMClassifier)

| rank | feature | importance |
|---:|---|---:|
| 1 | `return_h4_1` | 2588 |
| 2 | `return_h1_1` | 1655 |
| 3 | `return_m15_5` | 1504 |
| 4 | `range_vs_atr_m15` | 1463 |
| 5 | `return_m15_3` | 1427 |
| 6 | `return_h1_3` | 1394 |
| 7 | `return_m15_1` | 1358 |
| 8 | `atr_percentile_h1` | 1352 |
| 9 | `distance_to_recent_high_atr` | 1315 |
| 10 | `distance_to_recent_low_atr` | 1299 |
| 11 | `close_vs_ema8_atr` | 1298 |
| 12 | `ema21_vs_ema50_atr` | 1221 |
| 13 | `ema50_vs_ema200_atr` | 1210 |
| 14 | `close_vs_ema200_atr` | 1130 |
| 15 | `ema8_vs_ema21_atr` | 969 |

### L2_path_e_stationary (gain, from LGBMRegressor)

| rank | feature | gain |
|---:|---|---:|
| 1 | `return_h4_1` | 534.16 |
| 2 | `bear_score` | 377.86 |
| 3 | `score_gap` | 363.91 |
| 4 | `return_m15_5` | 259.30 |
| 5 | `ema21_vs_ema50_atr` | 237.10 |
| 6 | `close_vs_ema21_atr` | 201.60 |
| 7 | `return_m15_1` | 165.64 |
| 8 | `bull_score` | 152.65 |
| 9 | `range_vs_atr_m15` | 144.16 |
| 10 | `close_vs_ema50_atr` | 131.34 |
| 11 | `return_h1_1` | 126.13 |
| 12 | `close_vs_ema200_atr` | 104.31 |
| 13 | `return_h1_3` | 102.12 |
| 14 | `atr_percentile_h1` | 100.72 |
| 15 | `close_vs_ema8_atr` | 73.79 |

**Sanity:** neither top-15 contains `close_d1`, `close_m15`, `close_h1`, `close_h4`, `ema8_h1`, `ema21_h1`, `ema50_h1`, or `ema200_h1`. The regime-memorisation failure mode that killed legacy Path E is genuinely mitigated. Path F's feature space carries only ratios, returns, distances, score signals, and ATR-normalised range.

## 9. Per-gate verdict

| Gate | Required | B2_path_c_stationary | L2_path_e_stationary |
|---|---|:--:|:--:|
| Kept trades ≥ 25% of A_baseline (76 trades) | yes | ✅ 198 | ❌ 52 |
| PF > 1.011 (predecessor Path C baseline) | yes | ✅ 1.051 | ✅ 1.699 |
| PF > H_no_weak PF (0.873) | yes | ✅ 1.051 | ✅ 1.699 |
| Expectancy > H_no_weak Expectancy (-0.088) | yes | ✅ +0.019 | ✅ +0.255 |
| No raw-price feature in top-15 | yes | ✅ | ✅ |
| **Verdict** | | **GO (marginal)** | **NO-GO** |

Additional consideration against the *current live re-run* baseline (B_path_c PF 1.109 from the same eval invocation): B2 is at 1.051, **below** the live B_path_c run. Against the spec-fixed 1.011 threshold from the predecessor report B2 narrowly passes. The honest reading: B2 demonstrates the stationary feature space does not break Path C, but it does not unambiguously beat the legacy artifact either. Promotion should wait for confirmation from a longer test window or macro-enabled L3.

L2 fails Gate 1. Per the spec ("Do not claim success from high PF with tiny trade count"), L2's high PF / Expectancy do not override the kept-trades floor.

## 10. Recommendation

**Path C stays as production fallback.** Do not promote B2 to production yet — the lift over legacy Path C is marginal and the comparison against the live re-run baseline shows no improvement.

Next-step branches, ordered by expected leverage:

1. **Source DXY and US10Y CSVs and run L3** (highest leverage). L2 demonstrates the stationary feature space carries real signal (PF 1.699 on the keepers it admits); the failure was that the threshold passed validation but left too few keepers on test. Macro context features (`dxy_trend_slope`, `dxy_vol`, `us10y_trend_slope`) target the regime-shift dimension that the engine-internal features cannot see. Stooq (`https://stooq.com/q/?s=dx.f`) or FRED both provide free historical series. Once CSVs are dropped under `./data/`, rerun with `--dxy-csv ./data/dxy_h1.csv --us10y-csv ./data/us10y_h1.csv --path-e-stationary-macro ./models_cache/path_f_stationary_macro/path_e_expected_r_lgb.joblib` to evaluate L3.
2. **Lower the kept-trades floor or widen the threshold sweep.** The current `EXPECTED_R_THRESHOLDS = (0.00, 0.03, 0.05, 0.10, 0.15)` may be too sparse; adding 0.02 and 0.04 could yield a threshold that admits 60-80 test trades at PF > 1.3. This is a hyperparameter search, not a model change.
3. **Use L2 as an "analysis assistant" tier, not a gate.** When `predicted_R ≥ 0.05` on a live signal, flag it as high-confidence in the notification stream, but do not block engine signals below the threshold. This preserves Path C's coverage while surfacing L2's calibration as a side-channel.
4. **Defer further model changes** until either macro data lands or the kept-trades floor is renegotiated. Adding more features without those two checks risks reproducing the Path E overfitting story on a finer-grained feature space.

## Reproduction

```bash
# Train Path F artifacts
PYTHONPATH=. .venv/bin/python scripts/train_path_c_model.py \
    --csv data_long_m15.csv \
    --out models_cache/path_c_stationary.joblib \
    --feature-set stationary

PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
    --csv data_long_m15.csv \
    --out-dir models_cache/path_f_stationary \
    --feature-set stationary \
    --expected-r \
    --allow-degenerate

# Run full eval
PYTHONPATH=. .venv/bin/python -m scripts.eval_path_d \
    --csv data_long_m15.csv \
    --path-c models_cache/path_c_lgb.joblib \
    --path-d-filter models_cache/path_d_trade_outcome_lgb.joblib \
    --path-e models_cache/path_e_expected_r_lgb.joblib \
    --path-c-stationary models_cache/path_c_stationary.joblib \
    --path-e-stationary models_cache/path_f_stationary/path_e_expected_r_lgb.joblib \
    --report docs/reports/_path_f_eval_raw.md
```
