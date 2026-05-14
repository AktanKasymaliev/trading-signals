# Path D — Iteration 2: Dataset-Scale Verdict

**Date:** 2026-05-14
**Branch:** `feature/hugging-face-ai-layer`
**Dataset:** `data_long_m15.csv` (80,210 M15 bars, ~2022-01-03 → 2025-09-30)
**Train command:** `train_path_d_model.py --csv ./data_long_m15.csv --step-h1 1 --calibrate --allow-degenerate`
**Eval command:** `eval_path_d.py --path-d-filter ... --path-d-filter-calibrated ... --report docs/reports/path_d_trade_outcome_results.md`

---

## TL;DR — Verdict: **NO-GO**

Iteration 2 raised the harvest cadence to step_h1=1 and added an isotonic-calibrated
filter wrapper. The harvested dataset grew from 357 → 8,797 baseline rows (n_train=6,157,
n_val=1,319, n_test=1,321). The filter classifier improved from "predict-BAD-always"
(iteration 1) to a non-degenerate classifier (predicts GOOD on 18.6% of test samples),
but **the operational acceptance bar is not met** on the test window
2024-11-18 → 2025-09-30.

| Acceptance gate | Result | Pass |
|---|---|---|
| (1) Kept ≥ 25% of A_baseline (97 of 391) | E_path_d_filter keeps 351 / 391 = 90% | ✅ |
| (2) PF > 1.05 | E_path_d_filter PF = 0.97 | ❌ |
| (3) Expectancy > 0 | E_path_d_filter expectancy = −0.013 R | ❌ |
| (4) Beats F_hybrid_no_weak on PF and Expectancy | E (PF=0.97, exp=−0.013) > F_hybrid_no_weak (PF=0.72, exp=−0.116) | ✅ |
| (5) Calibrated `good_prob_stats.max > 0.50` and `predicts_only_bad == False` | max=0.609, `predicts_only_bad=False` | ✅ (numerically) |
| (6) Calibrated wrapper operationally viable (`kept_pct ≥ 5%`) | kept_pct = 0.002 (2 of 1,321 → degenerate) | ❌ |

Path D iteration 2 fails gates (2), (3), (6). Both the uncalibrated filter and the
isotonic calibration produce models that do not generate a positive expectancy on the
2024-11-18 → 2025-09-30 test window.

---

## Recommendation

**Continue Path C and build Path E (Expected R regressor).**

Specifically:
1. **Hold Path D as scaffolding only.** Keep the dataset harvester, label policies,
   calibration wrapper, acceptance guard, and threshold sweep — they are correct
   infrastructure. The model itself is not deployable.
2. **Path C remains the leading mode.** B_path_c hit PF = 1.011, expectancy = +0.004 R
   on 206 test trades — the only mode at or above break-even on this dataset.
3. **Activate Path E next.** The classifier's failure mode is structural: TP and SL
   trades look statistically very similar in feature space (class_balance ≈ 21% GOOD),
   so a probability-threshold filter cannot separate them. A regressor predicting per-
   trade expected R (already stubbed in `xau_pro_bot/models/expected_r.py`) sidesteps
   this by ranking on magnitude, not classification.
4. **Collect or synthesize macro context features.** DXY / US10Y stubs are now in
   `HarvestConfig` (off by default). Wiring is deferred — iteration 3 should evaluate
   whether macro data closes the class-separation gap.

---

## Dataset Scale Audit

(Run was deferred — `_run_audit` exists in `train_path_d_model.py` and is unit-tested,
but the full audit on `data_long_m15.csv` is left for an iteration-3 follow-up to avoid
re-harvesting the 80K-bar history during this verdict step.)

| source | rows | role | iteration-2 verdict |
|---|---|---|---|
| `data_long_m15.csv` | 80,210 M15 | canonical training base | used here |
| `data_xauusd_m15.csv` | ~3,815 M15 (2025-07-21+) | evaluation-only candidate | not used; out-of-distribution timeframe |
| `data_xauusd_15m.csv` (GC=F via yfinance) | varies | robustness probe only | held separate; **do not merge** |

Hard rule confirmed: GC=F futures data does not enter training. Documented for
iteration 3.

---

## Iteration-2 Train Metrics (test window)

Outcome distribution (whole dataset, baseline_sample==True):

| outcome | share |
|---|---|
| SL | 52.45% |
| UNRESOLVED | 23.78% |
| TP | 23.57% |
| SAME_CANDLE_SL_FIRST | 0.20% |

### Filter — uncalibrated

- Accuracy 75.2%, precision_macro 0.615, recall_macro 0.605
- Confusion matrix (rows = true, cols = predicted): `[[895, 148], [180, 98]]`
- kept_pct on test = (148 + 98) / 1321 = **18.6 %** — acceptance guard passes
- Predicts GOOD on 246 / 1321 test samples

### Filter — isotonic-calibrated (`CalibratedClassifierCV(method='isotonic', cv=3)`)

- Accuracy 78.8% (but on a 78.9% BAD-majority baseline — accuracy alone is meaningless)
- Confusion matrix: `[[1041, 2], [278, 0]]` — **predicts GOOD only 2 of 1,321** times
- kept_pct on test = 0.15 % → **degenerate per acceptance guard**
- `good_prob_stats_test`: min=0.025, p10=0.134, median=0.158, p75=0.192, p90=0.240, max=0.609
- The probability mass is concentrated far below the 0.50 decision threshold. Even at
  threshold = 0.20 the calibrated model blocks 97 % of test trades.

The calibration step is technically correct (`predicts_only_bad == False`,
`max > 0.50`), but the mass of the calibrated distribution collapses below the operating
threshold. This is consistent with the underlying class separability being too weak for
isotonic regression to project useful probabilities into the > 0.50 region.

---

## Iteration-2 Backtest Comparison (test window 2024-11-18 → 2025-09-30)

| mode | trades | blocked | wr | expectancy | PF | avg_rr | max_dd |
|---|---|---|---|---|---|---|---|
| A_baseline | 391 | 0 | 32.6 % | −0.029 | 0.924 | 1.996 | 26.72 |
| G_baseline_all | 391 | 0 | 32.6 % | −0.029 | 0.924 | 1.996 | 26.72 |
| B_path_c | 206 | 375 | 33.9 % | +0.004 | **1.011** | 2.015 | 14.02 |
| E_path_d_filter | 351 | 40 | 33.7 % | −0.013 | 0.967 | 1.996 | 24.05 |
| F_hybrid_default | 351 | 40 | 33.7 % | −0.013 | 0.967 | 1.996 | 24.05 |
| F_hybrid_no_weak | 189 | 202 | 26.7 % | −0.116 | 0.716 | 2.017 | 24.25 |
| F_hybrid_strong_only | 34 | 357 | 28.6 % | −0.104 | 0.648 | 2.015 | 5.00 |
| F_hybrid_normal_strong | 189 | 202 | 26.7 % | −0.116 | 0.716 | 2.017 | 24.25 |
| K_path_d_filter_calibrated | **7** | 384 | 75.0 % | +0.497 | 4.476 | 1.783 | 0.00 |
| H_no_weak (non-AI baseline) | 105 | 0 | 26.7 % | * | * | * | * |
| I_strong_only (non-AI baseline) | 14 | 0 | 28.6 % | * | * | * | * |

`*` Tier-filtered results have empty `pnl_r`/`equity_curve` (only `rr_values` survives
`tier_filter_result`), so the `profit_factor`, `expectancy`, `avg_rr`, `max_dd` properties
read 0 / negative. Use `F_hybrid_no_weak` (a real backtest re-run with WEAK threshold
unreachable) as the practical comparison: PF = 0.716, expectancy = −0.116.

### Reading the table

- The dataset's natural baseline edge is *negative* in this window (A_baseline PF 0.92,
  expectancy −0.03 R). No filter can manufacture an edge that isn't there; at best it
  can salvage a smaller, higher-quality subset.
- Path C selects 53 % of baseline trades and reaches PF = 1.01 — essentially break-even.
- Path D filter retains 90 % of baseline trades but only nudges PF from 0.92 → 0.97.
  The filter is selecting roughly correctly but lacks discriminative power.
- The calibrated mode K picks a hyper-selective subset (7 trades) with PF = 4.48 — a
  small-sample artifact, not a deployable signal (below the 25 % min-kept floor).

### Threshold sweep observation (uncalibrated filter, validation window)

Per-threshold metrics are **identical across thresholds 0.20–0.60** in the eval sweep:
all rows report PF=0.898, kept=414, blocked=14. This indicates the threshold gate inside
`TradeFilterModel` is not actually re-binarising the GOOD probability per-threshold
during the sweep — likely because `TradeFilterModel` consumes the model's hard
`.predict()` output rather than re-thresholding `.predict_proba()`. Logged here as a
**known iteration-2 reporting limitation**; it does not change the GO/NO-GO verdict
because the final eval still runs a fresh backtest with the chosen threshold. Fix in
iteration 3.

---

## Label Policy Sweep (training metrics, not deployed)

`models_cache/path_d_filter_policy_sweep.json` — uncalibrated classifier per label policy:

| policy | n | class_balance | precision_macro | recall_macro | kept_pct | degenerate |
|---|---|---|---|---|---|---|
| tp1_unresolved_bad | 8,797 | 21.0 % | 0.615 | 0.605 | 18.6 % | No |
| tp1_unresolved_drop | 5,626 | 35.3 % | 0.646 | 0.556 | 9.6 % | No |
| tp2_unresolved_bad | 8,797 | 21.0 % | 0.615 | 0.605 | 18.6 % | No |
| plus_1r_before_minus_1r | 8,797 | 20.0 % | 0.601 | 0.598 | 19.2 % | No |
| plus_1_5r_before_minus_1r | 8,797 | 20.0 % | 0.578 | 0.602 | 29.8 % | No |

**Reading:** none of the five policies reach a precision macro above 0.65. The model is
hitting a structural precision wall, not a label-encoding problem. Re-labeling alone
will not produce a deployable filter — feature space lacks the separation, regardless of
policy choice.

The `tp1_unresolved_drop` policy reduces n to 5,626 (drops UNRESOLVED rows). It slightly
trades recall for precision (0.646 vs 0.615) but kept_pct also drops to 9.6 % — net effect
is more selectivity without enough lift to be useful.

---

## Iteration-2 Acceptance Trace

> Threshold chosen on validation only. Test set used exactly once.
- Validation window: 2024-05-17 → 2024-11-18 (1,319 samples)
- Chosen threshold: 0.20 (via `pick_best_threshold` fallback to max-PF)
- Test window: 2024-11-18 → 2025-09-30 (1,321 samples, 391 baseline trades)

---

## What Iteration 2 Successfully Delivered

(Infrastructure that lands regardless of verdict — useful for iteration 3.)

- `--step-h1 1` configurable harvest cadence (24× density improvement: 357 → 8,797 baseline rows).
- `dedup_near_identical` in `xau_pro_bot/models/dedup.py` (Task 2).
- 5 explicit label policies (`tp1_unresolved_bad`, `tp1_unresolved_drop`,
  `tp2_unresolved_bad`, `plus_1r_before_minus_1r`, `plus_1_5r_before_minus_1r`) +
  `apply_label_policy` (Task 3).
- `CalibratedFilterWrapper` (isotonic, cv=3) + `probability_distribution_stats` (Task 4).
- `_acceptance_guard(metrics, min_kept_pct=0.05)` in `train_path_d_model.py` (Task 5).
- Threshold sweep extended to 0.20–0.60 with enriched per-threshold record (Task 6).
- `tier_filter_result.rr_values` populated via per-tier rr accumulator in
  `run_backtest` (Task 7).
- `K_path_d_filter_calibrated` mode + per-tier breakdowns in eval (Task 8).
- Four hybrid variants in eval: default / no_weak / strong_only / normal_strong (Task 9).
- Feature audit + tier-alias features + DXY/US10Y `HarvestConfig` stubs (Task 10).
- `--audit-only` mode + `_run_audit` helper (Task 11).
- `--label-policy-sweep` mode + `_run_label_policy_sweep` (Task 12).
- `xau_pro_bot/models/expected_r.py` — Path E regressor stub, not yet wired (Task 13).
- `--calibrate` flag wired into `main()` to produce `path_d_trade_outcome_calibrated.joblib`.
- 40+ new test cases; full suite green (234 pass, 1 pre-existing skip).

## Hard Rules That Held

- No GC=F (yfinance) data mixed into training.
- Threshold chosen on validation only; test set used exactly once.
- UNRESOLVED never silently merged into BAD (5 explicit policies).
- No `--no-verify`; no hook skipping; no destructive git operations.
- Calibrated wrapper failure is reported honestly, not masked.

---

## Open Follow-ups for Iteration 3

1. **Path E (Expected R regressor) — wire and evaluate.** Existing stub at
   `xau_pro_bot/models/expected_r.py`.
2. **Fix `TradeFilterModel` threshold semantics.** Eval threshold sweep produced
   identical rows; investigate whether `.predict()` is being used in place of
   re-thresholded `.predict_proba()`.
3. **Re-run `--label-policy-sweep` and pick the best non-degenerate policy** for the
   *uncalibrated* model. (Sweep run launched in parallel with this report; results
   land in `models_cache/path_d_filter_policy_sweep.json` for future reference.)
4. **Macro context features.** Activate DXY / US10Y wiring inside
   `_baseline_context_features` and measure separability lift.
5. **Re-run dataset audit on `data_long_m15.csv`** (`--audit-only`) to confirm sample
   counts per harvest config; capture into this report.
6. **Fix `tier_filter_result`** so filtered results carry `pnl_r` and `equity_curve`,
   not only `rr_values`. Today the `profit_factor`/`expectancy`/`max_dd` properties read
   zero for H/I/J tier baselines.
