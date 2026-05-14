# Path E (Expected R Regressor) — Verdict

**Date:** 2026-05-14
**Branch:** `feature/hugging-face-ai-layer`
**Verdict:** **NO-GO**

## Headline

Path E regressor is **over-fitted to the validation window** and collapses on test:

- On validation (`pick_best_threshold` picked `predicted_R > 0.15`), the sweep looked excellent — PF 8.999, Expectancy +1.185, WR 0.818 across 27 kept trades.
- On the test slice that same threshold blocks **391/391 trades — zero trades kept.**
- The regressor's predicted-R distribution shifted between val and test. p90 of train predictions is +0.168; on test the model never crosses +0.15 for any signal that survives the engine's tier gate.

## Acceptance gates (test slice)

| Gate | Required | Path E (test) | Pass? |
|------|----------|---------------|:----:|
| Kept trades ≥ 25% of `A_baseline` (≥ 97) | ≥ 97 | **0** | ❌ |
| PF on test > `B_path_c` PF (1.0114) | > 1.0114 | **0.000** | ❌ |
| PF on test > `H_no_weak` PF | > value (report bug — see note) | **0.000** | ❌ |
| Expectancy on test > `H_no_weak` Expectancy | > value | **0.000** | ❌ |
| Threshold chosen on validation only | yes | yes (0.15) | ✅ |

> **H_no_weak / I_strong_only / J_strong_normal_only rows show PF=0 and Expectancy=0** despite having 100+ trades. This is the unfixed `tier_filter_result.rr_values` aggregation bug carried over from Path D iteration 2 (plan task 7). It does not affect the Path E verdict — Path E lost on the kept-trades gate before the comparison even started. The bug should be revisited if/when AI gating is replanned.

## Process issue surfaced

The user's "kept ≥ 25% baseline" floor was violated **on validation** (chosen threshold 0.15 kept only 27 trades vs. ~98 floor) but `pick_best_threshold` silently fell back to the "highest PF among all entries" branch. The current logic:

```python
eligible = {t: m for t, m in sweep.items() if m["kept"] >= min_kept}
pool = eligible if eligible else sweep
```

This fallback is appropriate for Path D's continuous probability threshold (where threshold 0.5 reliably keeps ~half the trades), but for Path E with a skewed predicted_R distribution it accepts any threshold including ones that already violate the floor — and then the test slice has even less mass above the threshold. Recommendation if Path E is ever revisited: enforce the floor as a hard veto and return `None` when nothing qualifies, surfacing NO-GO directly from the picker.

## Headline numbers (test slice)

| mode | trades | PF | Expectancy | MaxDD |
|------|------:|----:|-----------:|------:|
| A_baseline           | 391 |  0.924 | -0.029 | 26.72 |
| H_no_weak\*          | 105 |  0.000 |  0.000 |  0.00 |
| B_path_c             | 206 |  1.011 | +0.004 | 14.02 |
| E_path_d_filter      | 351 |  0.967 | -0.013 | 24.05 |
| F_hybrid_no_weak     | 189 |  0.716 | -0.116 | 24.25 |
| F_hybrid_strong_only |  34 |  0.648 | -0.104 |  5.00 |
| **L_path_e_expected_r** | **0** | **0.000** | **0.000** | **0.00** |

\* H_no_weak/I/J rows are zero due to the unrelated `rr_values` aggregation bug noted above. PF/Expectancy on those rows cannot be trusted.

## Feature importance (top 15 by gain)

| feature                 | gain |
|-------------------------|----:|
| close_d1                | 5422 |
| close_m15               | 2751 |
| return_h4_1             | 2175 |
| ema8_h1                 | 1777 |
| ema200_h1               | 1410 |
| bull_score              | 1263 |
| ema21_h1                | 1262 |
| score_gap               |  813 |
| return_h1_1             |  742 |
| atr_h1                  |  703 |
| bear_score              |  664 |
| rsi_m15                 |  635 |
| rsi_h1                  |  576 |
| ema50_h1                |  566 |
| close_h1                |  484 |

**Observation:** raw price levels (`close_d1`, `close_m15`) dominate importance. This is a strong signal the model memorised the train regime's price level rather than learning regime-invariant structure — which directly explains the validation→test collapse. Price levels (and to a lesser extent EMAs) are non-stationary and shift across the time-split. A future Path E should drop absolute price features in favour of normalised returns / distances / ratios.

## Macro features (DXY / US10Y)

- DXY series available locally: **no** (`./data/` directory absent).
- US10Y series available locally: **no**.
- Out-of-sample comparison with vs. without macro features: **not run — data unavailable**.
- Verdict on macro features: **inconclusive — investigation deferred.** The wiring (`--dxy-csv` / `--us10y-csv` flags, harvest-side feature loader with default-off behaviour) is in place and covered by `tests/test_macro_features.py`; only the CSVs are missing.

## Decision

**NO-GO.** Path E in its current form is not a viable trade filter. The regressor over-fits to validation and produces an empty test set under the spec'd 25%-kept floor.

**Recommended next step (one of):**

1. **Continue Path C as the production AI gate.** Path C is the only mode that beats baseline on the test slice (PF 1.011 vs A_baseline 0.924) and meets the 25% floor (206 trades = 53% of baseline). It is weakly profitable but real.
2. **Park AI gating; ship H_no_weak as production baseline** — *but only after fixing the `tier_filter_result.rr_values` aggregation bug* (Path D iter-2 task 7). Without that fix, H_no_weak's reported PF/Expectancy is not trustworthy.
3. **Redesign features and labels before another Path E iteration:**
   - Drop absolute price columns (`close_d1`, `close_m15`, `close_h1`, `close_h4`, EMAs in price space). Replace with normalised returns or price-distance-to-EMA ratios.
   - Try MFE/MAE-derived targets instead of `final_R` — `final_R` clusters strongly at ±1.0R/+2.0R and is dominated by SL/TP geometry rather than the predictive component.
   - Collect macro data (DXY, US10Y) at M15 frequency and re-run with macro features on.

## Reproduction

```bash
# Train
PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
  --csv ./data_long_m15.csv --out-dir ./models_cache \
  --step-h1 1 --expected-r

# Eval (writes this report)
PYTHONPATH=. .venv/bin/python scripts/eval_path_d.py \
  --csv ./data_long_m15.csv \
  --path-c ./models_cache/path_c_lgb.joblib \
  --path-d-filter ./models_cache/path_d_trade_outcome_lgb.joblib \
  --path-e ./models_cache/path_e_expected_r_lgb.joblib \
  --report docs/reports/path_e_expected_r_results.md
```

Artifact: `models_cache/path_e_expected_r_lgb.joblib` (121 KB).
Full training metrics: `models_cache/path_d_metrics.json` → `expected_r`.

---

# Path D — Trade Outcome Results (auto-generated below)

**Test window:** 2024-11-18 18:00:00+00:00 -> 2025-09-30 19:00:00+00:00
**Validation window:** 2024-05-17 12:00:00+00:00 -> 2024-11-18 18:00:00+00:00
**Chosen filter threshold (from validation):** 0.2

## Modes

| mode | trades | blocked | wins | losses | wr | expectancy | pf | avg_rr | max_dd |
|---|---|---|---|---|---|---|---|---|---|
| A_baseline | 391 | 0 | 72 | 149 | 0.3258 | -0.0291 | 0.9236 | 1.9964 | 26.7219 |
| G_baseline_all | 391 | 0 | 72 | 149 | 0.3258 | -0.0291 | 0.9236 | 1.9964 | 26.7219 |
| H_no_weak | 105 | 0 | 28 | 77 | 0.2667 | 0.0 | 0.0 | -0.2082 | 0.0 |
| I_strong_only | 14 | 0 | 4 | 10 | 0.2857 | 0.0 | 0.0 | -0.2517 | 0.0 |
| J_strong_normal_only | 105 | 0 | 28 | 77 | 0.2667 | 0.0 | 0.0 | -0.2082 | 0.0 |
| B_path_c | 206 | 375 | 41 | 80 | 0.3388 | 0.0044 | 1.0114 | 2.0146 | 14.0234 |
| E_path_d_filter | 351 | 40 | 68 | 134 | 0.3366 | -0.0125 | 0.9673 | 1.996 | 24.0471 |
| F_hybrid_default | 351 | 40 | 68 | 134 | 0.3366 | -0.0125 | 0.9673 | 1.996 | 24.0471 |
| F_hybrid_no_weak | 189 | 202 | 28 | 77 | 0.2667 | -0.1157 | 0.7161 | 2.0171 | 24.2474 |
| F_hybrid_strong_only | 34 | 357 | 4 | 10 | 0.2857 | -0.1037 | 0.6476 | 2.0147 | 5.0 |
| F_hybrid_normal_strong | 189 | 202 | 28 | 77 | 0.2667 | -0.1157 | 0.7161 | 2.0171 | 24.2474 |
| L_path_e_expected_r | 0 | 391 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Filter Threshold Sweep (validation)

| th | kept | blocked | PF | Expectancy | WR | MaxDD | AvgRR |
|---|---|---|---|---|---|---|---|
| 0.20 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.25 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.30 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.35 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.40 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.45 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.50 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.55 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |
| 0.60 | 414 | 14 | 0.898 | -0.043 | 0.310 | 48.878 | 2.023 |

## L Path E (expected_R) — Threshold Sweep (validation)

**Chosen threshold (predicted_R >):** 0.15

| th | kept | blocked | PF | Expectancy | WR | MaxDD | AvgRR |
|---|---|---|---|---|---|---|---|
| 0.00 | 88 | 340 | 3.172 | 0.592 | 0.613 | 7.000 | 2.001 |
| 0.03 | 65 | 363 | 4.409 | 0.787 | 0.688 | 7.000 | 2.002 |
| 0.05 | 53 | 375 | 5.466 | 0.927 | 0.725 | 7.000 | 2.040 |
| 0.10 | 34 | 394 | 7.687 | 1.180 | 0.786 | 5.000 | 2.063 |
| 0.15 | 27 | 401 | 8.999 | 1.185 | 0.818 | 3.000 | 2.000 |

## Acceptance check

- Min trade floor (25% of baseline test trades): 97

## Notes

- Path C: forward-return labels (legacy).
- Path D: TP/SL outcomes on M15, time-split 70/15/15, threshold picked on validation only.
- Acceptance: PF > Path C **and** kept_trades >= 25% baseline, else 'do not deploy'.

## Training metrics

```json
{
  "outcome_distribution": {
    "SL": 0.5245436695513054,
    "UNRESOLVED": 0.23780088720820303,
    "TP": 0.23569194967638718,
    "SAME_CANDLE_SL_FIRST": 0.0019634935641044287
  },
  "directional_a1": {
    "accuracy": 0.7471612414837244,
    "precision_macro": 0.559756459255151,
    "recall_macro": 0.7522368459281211,
    "f1_macro": 0.6177949709864604,
    "confusion_matrix": [
      [
        3,
        0,
        0
      ],
      [
        4,
        867,
        172
      ],
      [
        0,
        158,
        117
      ]
    ],
    "n_train": 6157,
    "n_val": 1319,
    "n_test": 1321,
    "variant": "A1"
  },
  "directional_a2": {
    "accuracy": 0.690406976744186,
    "precision_macro": 0.5401734665340473,
    "recall_macro": 0.7045693378099993,
    "f1_macro": 0.5765284744483429,
    "confusion_matrix": [
      [
        96,
        7,
        0
      ],
      [
        182,
        1150,
        225
      ],
      [
        0,
        225,
        179
      ]
    ],
    "n_train": 9625,
    "n_val": 2062,
    "n_test": 2064,
    "variant": "A2"
  },
  "filter": {
    "accuracy": 0.7517032551097653,
    "precision_macro": 0.6154660616373605,
    "recall_macro": 0.6053098077626106,
    "f1_macro": 0.6095913615754457,
    "confusion_matrix": [
      [
        895,
        148
      ],
      [
        180,
        98
      ]
    ],
    "n_train": 6157,
    "n_val": 1319,
    "n_test": 1321
  },
  "expected_r": {
    "n_train": 6157,
    "n_val": 1319,
    "n_test": 1321,
    "mean_pred": -0.1523330186389756,
    "p10_pred": -0.4748662115663134,
    "p50_pred": -0.1419339140106186,
    "p90_pred": 0.16775638328571685,
    "feature_importance": {
      "close_d1": 5422.193052768707,
      "close_m15": 2751.308006286621,
      "return_h4_1": 2174.509802341461,
      "ema8_h1": 1776.8291053771973,
      "ema200_h1": 1409.8419597148895,
      "bull_score": 1263.0337135791779,
      "ema21_h1": 1261.9978232383728,
      "score_gap": 812.7365274429321,
      "return_h1_1": 742.1126079559326,
      "atr_h1": 702.7405371665955,
      "bear_score": 663.9834973812103,
      "rsi_m15": 634.891660451889,
      "rsi_h1": 575.629599571228,
      "ema50_h1": 565.5925209522247,
      "close_h1": 484.4052724838257,
      "ema21_above_ema50_h1": 440.37720012664795,
      "close_h4": 431.8159193992615,
      "dir_BUY": 371.9591431617737,
      "wyckoff_bias_h4_encoded": 296.09617280960083,
      "atr_m15": 188.3483214378357,
      "price_above_ema50_h1": 166.9950008392334,
      "return_m15_5": 164.39603924751282,
      "day_of_week": 143.94017934799194,
      "pd_zone_h4_encoded": 124.74524974822998,
      "atr_percentile_h1": 121.92588925361633,
      "price_above_ema200_h1": 104.35799074172974,
      "return_m15_1": 103.99868822097778,
      "final_score": 98.84481000900269,
      "hour_utc": 95.77831077575684,
      "ema50_above_ema200_h1": 83.80069828033447,
      "return_h1_3": 82.12608981132507,
      "hour_ny": 74.14684009552002,
      "ema8_above_ema21_h1": 73.16830062866211,
      "h1_range_pct": 35.01211905479431,
      "dir_SELL": 28.81679916381836,
      "return_m15_3": 23.74289035797119,
      "range_vs_atr_m15": 21.11848020553589,
      "m15_range_pct": 10.770920038223267,
      "tier_WEAK": 0.0,
      "tier_NORMAL": 0.0,
      "tier_STRONG": 0.0,
      "tier_NO_SIGNAL": 0.0,
      "rr": 0.0,
      "is_weak": 0.0,
      "is_normal": 0.0,
      "is_strong": 0.0,
      "kz_Asian_KZ": 0.0,
      "kz_London_KZ": 0.0,
      "kz_NY_AM_KZ": 0.0,
      "kz_NY_PM_KZ": 0.0,
      "kz_OFF": 0.0,
      "is_synthetic": 0.0
    }
  }
}
```