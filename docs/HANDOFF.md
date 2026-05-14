# XAU Pro Bot — Execution Handoff Log

> **Purpose:** This file is the source of truth for execution progress. If the active agent (Claude Opus 4.7) runs out of context, Codex (or another agent) picks up from here.
>
> **Plan path:** `docs/superpowers/plans/2026-05-11-xau-pro-bot.md`
> **Spec path:** `docs/superpowers/specs/2026-05-11-xau-pro-bot-design.md` (Revision 3)
> **AI plan path:** `docs/superpowers/plans/2026-05-12-hugging-face-ai-layer.md`
> **Working dir:** `/Users/aktan.kasymalievicloud.com/Projects/self-projects/signals/`
> **Current branch:** `feature/hugging-face-ai-layer`

## How to use this file

1. Find the latest `[in_progress]` or first `[pending]` task in the status table below.
2. Read its "Notes" section if present.
3. Open the plan file and execute the matching Task N from its first incomplete step.
4. After completing a task: update the status here from `pending` → `done`, add a one-liner under "Recent events".
5. If you change architecture mid-flight, append a "Deviations" entry below.

## Conventions

- Each task corresponds to a section in the plan (Task 0–24).
- Tasks 0–16 build the **intraday baseline** (single-stream bot).
- Tasks 17–24 add the **multi-stream architecture** (swing/scalp/SR zones).
- Tests must pass at the end of every task (run `pytest -v` if in doubt).
- All commits use conventional-commit prefixes (feat/fix/chore/test/docs).

## Status

| # | Task                                     | Status      | Commit / Notes |
|---|------------------------------------------|-------------|----------------|
| 0 | Scaffolding + git init                   | done        | 257c6e4 |
| 1 | Config module                            | done        | |
| 2 | Test fixtures (conftest.py)              | done        | |
| 3 | State module (SQLite)                    | done        | |
| 4 | Data module (Twelve Data + cache)        | done        | |
| 5 | Classic indicators                       | done        | |
| 6 | ICT indicators                           | done        | |
| 7 | SMC indicators                           | done        | |
| 8 | Wyckoff (soft bias)                      | done        | |
| 9 | SR levels helpers                        | done        | |
| 10| Signal engine (5-layer)                  | done        | np.isnan→pd.isna fix |
| 11| Filters (dedup/ratelimit)                | done        | |
| 12| Formatter                                | done        | |
| 13| Bot entrypoint                           | done        | |
| 14| Backtest                                 | done        | |
| 15| README                                   | done        | |
| 16| Acceptance smoke                         | done        | 64/64 tests pass |
| 17| Pip constant + state migration           | done        | R3 |
| 18| SR zones module                          | done        | R3 |
| 19| SR zones into intraday engine            | done        | R3 |
| 20| Swing stream                             | done        | R3 |
| 21| Scalp stream                             | done        | R3 |
| 22| StreamRouter + per-stream filters        | done        | 212d7e6 |
| 23| Wire router + formatter labels           | done        | Codex takeover |
| 24| Per-stream backtest                      | done        | Codex takeover |

## Recent events

(append latest at the top)
- **AI layer execution paused during Task 5 review.** Tasks AI-1 through AI-4 are implemented and review-approved. Task AI-5 is implemented and spec-approved, but code-quality review requested fixes before proceeding to formatter/backtest/docs.
- **Codex takeover completed Tasks 23–24.** Finished router wiring in `bot.py`, persisted stream labels, added per-stream backtest support, and verified `python -c "from xau_pro_bot import bot; print('ok')"` plus `pytest -v` (88/88 passed).
- **Tasks 0–16 complete.** Intraday baseline green: 64/64 tests pass. Module smoke OK. Ready for multi-stream phase (Tasks 17–24).
- T10 fix: `np.isnan(value)` raised TypeError when pandas_ta_classic returns `None` for short series. Replaced with `pd.isna()` in `engine.py` + `classic_signals.py`.
- Task 0 in progress: scaffolded files, venv created, deps installed via `pandas-ta-classic` (see deviation).

## AI Layer Status

Plan: `docs/superpowers/plans/2026-05-12-hugging-face-ai-layer.md`

Goal: add an optional Hugging Face AI confirmation/filter layer for intraday deterministic signals. AI must remain disabled by default, tests must not require internet/model downloads, and the deterministic engine must stay intact.

| # | Task | Status | Commit / Notes |
|---|------|--------|----------------|
| AI-1 | AI config/env/dependencies | done | `593c7af`, `616fafe`, `131b295` |
| AI-2 | Deterministic AI feature builder | done | `977805a` |
| AI-3 | AI calibration rules | done | `db874b4`, `e1d8890` |
| AI-4 | Hugging Face model adapter | done | `f40a599`, `edd53d7`, `9f6b57d`, `bc5391a` |
| AI-5 | Integrate AI into `MasterSignalEngine` | in_progress | `c7b42b5`, `43f4e5f`; spec-approved, quality review has open fixes |
| AI-6 | Formatter compact AI line | pending | Not started |
| AI-7 | Backtest baseline vs AI comparison | pending | Not started |
| AI-8 | README AI documentation | pending | Not started |
| AI-9 | Final verification/regression fixes | pending | Not started |

### AI Tasks Completed

AI-1 added optional AI configuration:

- `AI_ENABLED`, `AI_MODEL_ID`, `AI_MODEL_TYPE`, confidence thresholds, score bonuses, conflict penalty, and cache dir.
- `load_ai_config()` reads live env values so tests and runtime overrides work after import.
- Import-time numeric parsing was fixed so malformed optional AI env vars do not crash `import xau_pro_bot.config` while AI is disabled.
- `.env.example` includes AI tuning variables.
- Dependencies added: `huggingface_hub`, `joblib`, `scikit-learn`. `torch` and `transformers` were intentionally not added to base requirements.

AI-2 added `xau_pro_bot/models/features.py`:

- `build_ai_features(tfs)` returns exactly one one-row DataFrame.
- Feature order is fixed via `REQUIRED_AI_FEATURES`.
- Uses copied DataFrames, so it does not mutate input.
- Handles short/missing data defensively.
- Encodes H4 premium/discount and Wyckoff bias as bull=1, neutral=0, bear=-1.

AI-3 added `xau_pro_bot/models/calibration.py`:

- `ai_prediction_to_adjustment(prediction, deterministic_direction)` maps AI output to buy/sell score deltas or signal blocking.
- Uses `config.load_ai_config()` at call time so env overrides affect calibration.
- Covers agreement bonus, strong agreement bonus, conflict penalty, `NO_TRADE` block, and low-confidence no-op behavior.

AI-4 added `xau_pro_bot/models/hf_model.py`:

- `HFTradingModel` lazy-loads only on `predict()`, not import or `__init__`.
- Supports sklearn/joblib artifacts through Hugging Face Hub.
- Supports `predict_proba`, string classes, numeric classes `1/-1/0`, and predict-only models.
- Transformer support is lazy and returns a clear neutral error if optional deps are missing.
- Custom mode returns a neutral error until an injected adapter exists.
- Security hardening: sklearn/joblib loading requires a pinned immutable 40-character Hugging Face commit SHA revision before any download/load. Missing, branch, tag, or short revisions return neutral and do not call `hf_hub_download` or `joblib.load`.

AI-5 currently implemented:

- `MasterSignalEngine` accepts `ai_enabled: bool | None = None` and `ai_model` injection.
- AI disabled/default path returns `ai_*` fields with disabled/null/zero values.
- Disabled path does not instantiate `HFTradingModel`.
- Injected AI model path can boost, penalize, or block deterministic intraday signals.
- Spec review approved after adding tests for default-disabled/no-download behavior.

### AI-5 Open Review Fixes

Code-quality review requested these fixes before AI-5 can be marked done:

1. `MasterSignalEngine` constructs `HFTradingModel` without a pinned revision. Since `HFTradingModel` now requires a 40-character SHA for sklearn/joblib, config needs an AI model revision value and engine must pass it through.
   - Add config/env support such as `AI_MODEL_REVISION`, default `""`.
   - Add `.env.example` entry.
   - Pass `revision=str(ai_cfg["model_revision"])` to `HFTradingModel`.
   - Add an engine/config test proving enabled sklearn config passes the revision through.

2. `_run_ai_adjustment()` should isolate AI failures. An injected model can raise, malformed predictions can break calibration, or feature building can fail. The deterministic engine should survive and return an enabled-but-neutral AI result.
   - Wrap feature building, `predict()`, and calibration conversion in `try/except`.
   - Return neutral AI fields with `ai_enabled=True`, no score deltas, `ai_blocked=False`, and a human-readable `ai_reason` containing the error.
   - Add a test with an injected model whose `predict()` raises.

3. AI score/tier consistency needs correction. Current code can keep pre-AI deterministic `direction` while using `final_score = max(bull_score, bear_score)` after AI deltas. That can produce a score/tier from the opposite side while levels remain for the original direction.
   - Preferred fix for this feature: AI is a confirmation layer, so keep deterministic direction and compute final score/tier from that selected direction only after applying AI deltas to that side.
   - Example approach: after preliminary `direction`, apply deltas, then set `final_score = bull_score if direction == "BUY" else bear_score`.
   - Add a regression test where conflict penalty lowers the selected side below the opposite side and assert result direction/score/tier remain internally consistent.

After fixing AI-5, rerun:

```bash
.venv/bin/pytest tests/test_engine_ai.py -v
.venv/bin/pytest tests/test_engine.py tests/test_router.py tests/test_pipeline.py -v
.venv/bin/pytest tests/test_ai_features.py tests/test_ai_calibration.py tests/test_hf_model.py tests/test_engine_ai.py -v
```

Then rerun Task AI-5 spec and code-quality review gates before moving to AI-6.

## Deviations from plan

- **Task 0 / requirements:** `pandas-ta==0.3.14b` is no longer installable on Python 3.11 (pulled from PyPI; latest 0.4.x requires Python 3.12+). Switched to maintained fork **`pandas-ta-classic>=0.3.15,<0.5`** with `pandas>=2.2,<3.0`. The numpy<2.0 pin was dropped because pandas-ta-classic uses `numpy.nan` correctly. The `numpy.NaN` monkey-patch in `xau_pro_bot/indicators/__init__.py` is kept defensively but is a no-op with the new fork. All call sites import via `from xau_pro_bot.indicators import pandas_ta as ta` — the package re-exports `pandas_ta_classic` under the `pandas_ta` name to keep code identical to the plan.

## Known issues / TODO for follow-up agent

- Continue from AI-5 code-quality review fixes above.
- AI-6, AI-7, AI-8, and AI-9 are not started.
- Remember that Task AI-4 added a stricter security requirement than the original AI plan: sklearn/joblib Hugging Face artifacts require a pinned 40-character commit SHA revision.

---

## PoC: SMC v2 (2026-05-13)

**Plan:** `docs/superpowers/plans/2026-05-12-poc-smc-v2-integration.md`
**Model:** `JonusNattapong/xauusd-trading-ai-smc-v2` @ `d1ee87d058bf714af1b6f4b3979646dd0024b726`, file `trading_model_15m.pkl` (XGBClassifier, binary BUY/NOT-BUY, 21 features).

**Infrastructure: ✅ WORKS end-to-end.**
- Model downloads with pinned SHA.
- Custom filename loaded via `AI_MODEL_FILENAME`.
- `smc_v2` feature builder (21 features in correct order, LabelEncoder `bearish=0/bullish=1/none=2`) verified against `model.feature_names_in_`.
- XGBoost `predict_proba` returns valid `{BUY, NO_TRADE}` distribution.
- Calibration → engine → backtest pipeline functional. Smoke test produced a SELL/BUY conflict with `score_delta_sell=-10` as expected.

**Comparison run (2025-07-21 → 2025-09-18, 956 H1 bars):**
- Baseline: **0 signals** (all NO_SIGNAL, scores -20..+12, median -6, all below WEAK threshold 40).
- AI run: **0 signals + 84 AI blocks** (NO_TRADE confidence ≥ 0.60 on 84 candidate setups).

**Findings:**
1. **Baseline calibration issue (separate from AI).** Macro penalties (D1 trend against = -20) + scoring weights produce scores far below tier thresholds for this XAU/USD window. Recalibrating WEAK/NORMAL/STRONG cutoffs (or relaxing macro penalty) needed before any AI evaluation is meaningful.
2. **AI is BUY-only and trigger-happy on NO_TRADE.** 84 blocks in 2 months ≈ 1.5/day. Suggests either (a) the model genuinely says "don't trade" most of the time, or (b) our feature-builder approximations (FVG/OB heuristics, raw prices vs trained on different scale) push the model toward defensive `NO_TRADE`. Trees ARE scale-robust (verified: scaled vs raw predictions <2.5% delta on training data), so likely (a).
3. **README of the HF model was wrong:** claimed 23 features incl. Volume + Recovery_Type; actual contract is 21 features without them. Always trust `feature_names_in_` over docs.

**Conclusion for Path C decision:**
Infrastructure validation complete — AI layer cleanly plugs in. The ceiling on a comparison-based assessment is now limited by baseline calibration, not by the AI seam. Pivot recommendations:
- Before training our own LightGBM (Path C), fix baseline calibration so we have a non-zero comparison baseline.
- When training, use OUR feature builder (29 deterministic features) + walk-forward labeling from existing backtester to keep methodology in-domain.

**Files added by PoC:**
- `xau_pro_bot/models/smc_v2_features.py`
- `tests/test_smc_v2_features.py`
- `tests/test_ai_model_filename.py`
- `tests/test_ai_feature_set.py`
- `scripts/poc_smc_v2_smoke.py`
- `scripts/poc_smc_v2_compare.py`
- `data_xauusd_15m.csv`, `data_xauusd_h1.csv`, `data_xauusd_m15.csv` (gitignored — model training data, ~1MB total)

**Commits:** `44a16fa` → `da9977c` → `1aefcfb` → `d93774e` → `29a62e7` → `ff3c3bd`.

---

## Path C: LightGBM (2026-05-13)

Trained an in-house LightGBM 3-class classifier (BUY=1 / NO_TRADE=0 / SELL=-1) on our own 29 deterministic features over 80K M15 bars (2022-01 → 2025-09, XAU/USD), labeled by forward-return at h=16 bars (4h), threshold ±0.3%.

**Training:** 9,925 samples → 7,940 train / 1,985 test (time-based 80/20). LightGBM with class_weight=balanced, early stopping at iter 184.

**Holdout metrics:**
- accuracy 65.2%, macro F1 0.53
- Class -1: P 0.53 / R 0.34 / F1 0.41
- Class  0: P 0.72 / R 0.81 / F1 0.76
- Class  1: P 0.45 / R 0.41 / F1 0.43

**Backtest comparison (full M15 history, step=12, timeout=48):**
- BASE: sig=717 W/L=129/339 WR=27.6% E=-0.12R PF=0.74
- AI  : sig=445 blocked=707 W/L=86/207 WR=29.4% E=-0.09R PF=0.81
- DELTA: trades=-272 WR=+1.8% E=+0.03R PF=+0.07

**Read:** AI lifts every metric modestly by filtering out 38% of baseline signals. Still unprofitable (PF<1) — baseline-calibration ceiling persists. Local-path infrastructure (`AI_MODEL_LOCAL_PATH`) wired in cleanly.

**Files:** `xau_pro_bot/models/train_lightgbm.py`, `scripts/train_path_c_model.py`, `scripts/poc_path_c_compare.py`, `tests/test_train_lightgbm.py`, `tests/test_local_model_path.py`, `models_cache/path_c_lgb.joblib`.

---

## Path D: Trade Outcome LightGBM (2026-05-13)

Built TP/SL outcome labeler, baseline-only sample harvester (with optional synthetic), 3-class Directional (A1/A2) + 2-class Filter trainers, `TradeFilterModel` adapter, `HybridPolicy`, opt-in engine seam, backtest `walk_from`/`walk_to` window controls, evaluator with 6 AI modes + 4 non-AI tier baselines + validation threshold sweep.

**Training (full 80K M15 history, step_h1=4, timeout 192 M15):**
- Harvested 1,595 samples (357 baseline + 1,238 synthetic).
- Outcome distribution: SL 61.6%, TP 28.4%, UNRESOLVED 9.8%, SAME_CANDLE_SL_FIRST 0.1%.
- Directional A1 (baseline-only, 249 train): **degenerate** — predicts single class, acc=0 on a test split with no `-1` support.
- Directional A2 (full 1,116 train): acc 52%, F1_macro 0.51 — recall-heavy on rare classes (P low).
- Filter (249 baseline train): acc 78% **by predicting BAD for everything** — class-1 (GOOD) precision/recall = 0.

**Backtest comparison (test window 2024-11-18 → 2025-09-30):**

| mode | trades | wr | pf | expectancy |
|---|---|---|---|---|
| A_baseline | 50 | 31.8% | 0.83 | −0.05R |
| B_path_c | 34 | 33.3% | ~1.00 | ~0R |
| H_no_weak | 5 | 60% | * | * |
| I_strong_only | 0 | — | — | — |
| E_path_d_filter | — | — | — | (blocked all) |
| F_hybrid | — | — | — | (blocked all) |

(* H/I/J non-AI tier filters drop rr_values — pf/expectancy not populated; minor reporting gap.)

**Verdict: do not deploy Path D filter.**
- Filter learned the majority-class shortcut (predict BAD always). At every sweep threshold 0.50–0.75, kept_trades=0 → fails the 25% baseline floor by construction.
- Directional A1 degenerate; A2 over-predicts rare directional classes (P low, R high).
- Path C still narrowly leads on out-of-sample PF (~1.0 vs baseline 0.83). Path D adds infrastructure but no edge yet.

**Root cause of filter degeneracy:** dataset is small (357 baseline rows), heavily imbalanced (72% BAD), and the LGBM `class_weight=balanced` with `predict_proba` threshold sweep should help — but at all tested thresholds the GOOD probability never exceeds 0.50, so the filter blocks every signal. Need calibration (e.g., isotonic) or lower-threshold sweep (0.20–0.45) or richer features.

**Recommended next iterations:**
1. Lower threshold sweep to {0.20, 0.25, 0.30, 0.35, 0.40, 0.45} — the model's `good_prob` distribution is shifted left because of class imbalance.
2. Drop step_h1 to 1 on M15 (not H1) to multiply baseline samples ~16x.
3. Add probability calibration (`CalibratedClassifierCV`) to make threshold semantics meaningful.
4. Fix `tier_filter_result` to preserve rr_values (or recompute from outcome class) so H/I/J get real PF/expectancy.
5. Consider focal loss or up-sampling the minority TP class.

**Files added:** `xau_pro_bot/models/{trade_outcome,path_d_harvest,train_path_d,trade_filter_model}.py`, `xau_pro_bot/signals/hybrid_policy.py`, `scripts/{train_path_d_model,eval_path_d}.py`, `docs/superpowers/specs/2026-05-13-path-d-trade-outcome-design.md`, `docs/superpowers/plans/2026-05-13-path-d-trade-outcome.md`, `docs/reports/path_d_trade_outcome_results.md`, 7 new test files.

**Commits:** `9deada9` → `7e5ae86` → `08d8842` → `e91d538` → `5dee0b6` → `53acb01` → `c1540ff` → `9fca8b7` → `f014982` → `27df0f6` → `37ea3c9` → `b797e4a` → `4ba35a4`.

---

## Iteration 2 — Dataset-scale re-evaluation (2026-05-14)

**Branch:** `feature/hugging-face-ai-layer`
**Plan:** `docs/superpowers/plans/2026-05-13-path-d-iteration-2.md`
**Verdict report:** `docs/reports/path_d_iteration_2_dataset_scale.md`
**Eval report:** `docs/reports/path_d_trade_outcome_results.md` (iteration-2 section appended)

**Verdict: NO-GO on Path D filter.** Continue Path C; activate Path E (Expected R regressor).

**What changed this iteration:**
- Harvest cadence: step_h1=4 → step_h1=1 (357 → 8,797 baseline rows; 24× density).
- Isotonic-calibrated filter wrapper (`CalibratedFilterWrapper`) + `--calibrate` flag.
- 5 explicit label policies; UNRESOLVED never silently merged into BAD.
- Acceptance guard rejects all-BAD or kept_pct<5% models.
- Threshold sweep extended to 0.20–0.60.
- Hybrid variants: default / no_weak / strong_only / normal_strong.
- Per-tier rr accumulator in `run_backtest` → `tier_filter_result` carries `rr_values`.
- Feature audit + tier-alias features (`is_weak`, `is_normal`, `is_strong`); DXY/US10Y stubs (off by default).
- `--audit-only` and `--label-policy-sweep` modes in trainer.
- Path E regressor stub at `xau_pro_bot/models/expected_r.py` (not wired into eval).
- 40+ new tests; full suite 234 passed, 1 pre-existing skip.

**Acceptance gates (test window 2024-11-18 → 2025-09-30):**

| gate | result | pass |
|---|---|---|
| Kept ≥ 25% of A_baseline (97 trades) | E: 351; K_calibrated: 7 | E ✅ / K ❌ |
| PF > 1.05 | E: 0.97; K: 4.48 (n=7) | ❌ |
| Expectancy > 0 | E: −0.013; K: +0.50 (n=7) | ❌ |
| Beats F_hybrid_no_weak | E (0.97 > 0.72) | ✅ |
| Calibrated `good_prob.max>0.50` and not all-BAD | max=0.609, predicts_only_bad=False | ✅ (numerically) |
| Calibrated kept_pct ≥ 5% | kept_pct=0.002 → degenerate | ❌ |

**Why NO-GO:** baseline dataset edge is negative in this window (A_baseline PF=0.92,
exp=−0.029). The classifier filter retains 90% of trades but barely lifts PF (0.92 → 0.97).
The isotonic-calibrated probability mass collapses below the 0.50 decision threshold —
calibration is technically correct but operationally degenerate. The issue is structural:
TP and SL trades look similar in feature space; a probability filter cannot separate them.

**Iteration 3 follow-ups (priority-ordered after post-iter code review):**
1. **[CRITICAL]** Fix `train_filter_calibrated` — `va` split is computed and ignored;
   `cv=3` fits without early stopping. Switch to `cv="prefit"` with a pre-fit base +
   early stopping on `va`. Plausibly explains the calibrated model's probability-mass
   collapse.
2. **[HIGH]** Guard `--calibrate` against silent artifact clobber (`--force` or
   timestamp suffix).
3. **[MEDIUM]** `TP1_UNRESOLVED_BAD` and `TP2_UNRESOLVED_BAD` produce identical
   `label_filter` — differentiate or remove the duplicate.
4. **[MEDIUM]** `_run_label_policy_sweep` writes `good_prob_stats: {}` stub —
   populate or drop the key.
5. Wire Path E (Expected R regressor) into eval.
6. Fix `tier_filter_result` to carry `pnl_r`/`equity_curve` (non-AI baselines
   currently report 0 for PF/expectancy/max_dd).
7. Wire DXY/US10Y macro features.

> Note: the prior hypothesis "identical sweep rows = threshold not plumbed" was
> incorrect. Code review confirmed `TradeFilterModel` re-thresholds `predict_proba`
> per call; the real cause is the good-prob distribution being concentrated below
> all tested thresholds.

---

## Path E (Expected R Regressor) — 2026-05-14 — **NO-GO**

Full report: `docs/reports/path_e_expected_r_results.md`. Plan: `docs/superpowers/plans/2026-05-14-path-e-expected-r.md`. All Path E commits between `f743c72` and HEAD on `feature/hugging-face-ai-layer`.

**What changed this iteration:**
- Promoted `xau_pro_bot/models/expected_r.py` stub → production trainer with gain-importance metrics.
- `ExpectedRFilterModel` runtime adapter (`predicted_R ≥ threshold → KEEP`), drop-in for `filter_model=` in `run_backtest`.
- Engine routing branch in `xau_pro_bot/signals/engine.py`: filter results carrying `predicted_r` are routed past `hybrid_decide` and consume `decision` directly. Path D shape (`good_prob`/`bad_prob`) unchanged.
- `--expected-r`, `--dxy-csv`, `--us10y-csv` trainer flags.
- DXY/US10Y feature wiring in `path_d_harvest.py` — off by default; opt-in via `HarvestConfig.dxy_csv` / `us10y_csv`.
- New eval mode `L_path_e_expected_r` with predicted_R sweep `(0.00, 0.03, 0.05, 0.10, 0.15)` reusing the existing `pick_best_threshold(min_kept=...)` 25% floor.
- ~10 new tests; full suite 247 passed, 1 skipped.

**Acceptance gates (test window 2024-11-18 → 2025-09-30):**

| gate | result | pass |
|---|---|---|
| Kept ≥ 25% of A_baseline (97 trades) | **L_path_e: 0 trades** | ❌ |
| PF on test > B_path_c (1.011) | L_path_e: 0.000 | ❌ |
| PF on test > H_no_weak | 0.000 vs report-broken | ❌ |
| Expectancy on test > H_no_weak | 0.000 | ❌ |
| Threshold chosen on validation only | yes (0.15) | ✅ |

**Why NO-GO:** validation/test distribution shift. The chosen threshold `predicted_R > 0.15` looked superb on validation (PF=9.0, n=27) but blocked 391/391 trades on the test slice. Feature importance shows raw price levels (`close_d1` gain 5422, `close_m15` 2751) dominating — the regressor memorised the train-regime price level rather than learning regime-invariant structure.

**Process bug surfaced:** the 25%-kept floor was violated already on validation (chosen threshold kept 27 ≪ 97). `pick_best_threshold` silently fell back to "highest PF" because no entry met the floor. Acceptable for Path D's continuous classifier but masks failure for Path E's skewed predictions.

**Recommendation:**
- **Best practical next step:** stick with Path C as the production AI gate (test PF 1.011, n=206, 53% kept). Path C is the only mode that beats baseline.
- **If Path E is revisited:** drop absolute price columns from the feature space (use normalised returns / EMA distance ratios); try MFE/MAE-derived targets; collect DXY/US10Y at M15 frequency.
- **Carry-over fixes still pending from iter-2:** `tier_filter_result.rr_values` aggregation bug (H/I/J modes report PF=0 incorrectly); `train_filter_calibrated` cv=3 + ignored `va` split.
