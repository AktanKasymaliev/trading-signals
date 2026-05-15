# Path D — Trade Outcome Results

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
| K_path_d_filter_calibrated | 7 | 384 | 3 | 1 | 0.75 | 0.4966 | 4.4761 | 1.7829 | 0.0 |

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

## K Calibrated Filter — Threshold Sweep (validation)

**Chosen threshold:** 0.2

| th | kept | blocked | PF | Expectancy | WR | MaxDD | AvgRR |
|---|---|---|---|---|---|---|---|
| 0.20 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.25 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.30 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.35 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.40 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.45 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.50 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.55 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |
| 0.60 | 11 | 417 | 10.011 | 0.819 | 0.857 | 1.000 | 2.069 |

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
  "filter_calibrated": {
    "accuracy": 0.7880393641180924,
    "precision_macro": 0.3946171341925701,
    "recall_macro": 0.4990412272291467,
    "f1_macro": 0.44072819644369177,
    "confusion_matrix": [
      [
        1041,
        2
      ],
      [
        278,
        0
      ]
    ],
    "n_train": 6157,
    "n_val": 1319,
    "n_test": 1321,
    "good_prob_stats_test": {
      "min": 0.02528735632183908,
      "p10": 0.13371584927286886,
      "p25": 0.1417618262843631,
      "median": 0.15815275261021236,
      "p75": 0.19239115626398862,
      "p90": 0.23989455761167636,
      "max": 0.609006734006734
    },
    "predicts_only_bad": false
  }
}
```
---

## Iteration 2 — Dataset-Scale Re-evaluation (2026-05-14)

See full verdict report: [`path_d_iteration_2_dataset_scale.md`](./path_d_iteration_2_dataset_scale.md).

**Verdict: NO-GO.** Path D iteration 2 trains on a 24× larger dataset (357 → 8,797 baseline rows via step_h1=1) and adds an isotonic-calibrated filter wrapper. The classifier is no longer "predict-BAD-always", but the operational gates still fail:

- E_path_d_filter: PF=0.97, expectancy=−0.013 R on 351/391 baseline trades (kept 90%) — fails PF>1.05 and expectancy>0 gates.
- K_path_d_filter_calibrated: PF=4.48 but only 7 trades retained (kept_pct=0.15%) — fails the 25% min-kept floor (97 trades).
- F_hybrid_no_weak: PF=0.72, expectancy=−0.12 — strictly worse than A_baseline.
- B_path_c: PF=1.011, expectancy=+0.004 R on 206 trades — still the only mode at/above break-even.

**Recommendation:** continue Path C, activate Path E (Expected R regressor stub at `xau_pro_bot/models/expected_r.py`), and explore macro context features (DXY/US10Y hooks now in `HarvestConfig`).
