# Path D — Trade Outcome Results

**Test window:** 2024-11-18 18:00:00+00:00 -> 2025-09-30 19:00:00+00:00
**Validation window:** 2024-05-17 12:00:00+00:00 -> 2024-11-18 18:00:00+00:00
**Chosen filter threshold (from validation):** None

## Modes

| mode | trades | blocked | wins | losses | wr | expectancy | pf | avg_rr | max_dd |
|---|---|---|---|---|---|---|---|---|---|
| A_baseline | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| H_no_weak | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| I_strong_only | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| J_strong_normal_only | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B_path_c | 0 | 343 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Filter Threshold Sweep (validation)

| threshold | kept | blocked | wr | expectancy | pf |
|---|---|---|---|---|---|
| 0.50 | 0 | 1 | 0.000 | 0.000 | 0.000 |
| 0.55 | 0 | 1 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0 | 1 | 0.000 | 0.000 | 0.000 |
| 0.65 | 0 | 1 | 0.000 | 0.000 | 0.000 |
| 0.70 | 0 | 1 | 0.000 | 0.000 | 0.000 |
| 0.75 | 0 | 1 | 0.000 | 0.000 | 0.000 |

## Acceptance check

- Min trade floor (25% of baseline test trades): 0

## Notes

- Path C: forward-return labels (legacy).
- Path D: TP/SL outcomes on M15, time-split 70/15/15, threshold picked on validation only.
- Acceptance: PF > Path C **and** kept_trades >= 25% baseline, else 'do not deploy'.

## Training metrics

```json
{
  "outcome_distribution": {
    "SL": 0.6163009404388715,
    "TP": 0.284012539184953,
    "UNRESOLVED": 0.09843260188087774,
    "SAME_CANDLE_SL_FIRST": 0.0012539184952978057
  },
  "directional_a1": {
    "accuracy": 0.0,
    "precision_macro": 0.0,
    "recall_macro": 0.0,
    "f1_macro": 0.0,
    "n_train": 249,
    "n_val": 53,
    "n_test": 55,
    "variant": "A1"
  },
  "directional_a2": {
    "accuracy": 0.5208333333333334,
    "precision_macro": 0.5249085361444912,
    "recall_macro": 0.7022555875497053,
    "f1_macro": 0.5135762483130905,
    "n_train": 1116,
    "n_val": 239,
    "n_test": 240,
    "variant": "A2"
  },
  "filter": {
    "accuracy": 0.7818181818181819,
    "precision_macro": 0.39090909090909093,
    "recall_macro": 0.5,
    "f1_macro": 0.4387755102040816,
    "n_train": 249,
    "n_val": 53,
    "n_test": 55
  }
}
```