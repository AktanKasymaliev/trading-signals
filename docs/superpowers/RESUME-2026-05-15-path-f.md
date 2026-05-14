# Resume Note — 2026-05-15 — Path F landed

**Branch:** `feature/hugging-face-ai-layer` (5 commits ahead of `origin/feature/hugging-face-ai-layer`)
**Working tree:** clean
**Status:** Path F implementation COMPLETE. No work in flight.

## What just happened (one paragraph)

Path E was NO-GO due to absolute-price features memorising the train regime. Path F replaced the feature space with 17 stationary ratios/returns/distances, fixed two threshold/result bugs in the eval, added DXY/US10Y macro plumbing with a NO_MACRO_DATA contract, and re-evaluated. Result: **B2_path_c_stationary** is marginal GO (PF 1.051, +0.019 expectancy, 198 trades — clears 25% kept floor but barely beats legacy Path C). **L2_path_e_stationary** is NO-GO on the kept-trades floor (PF 1.699, +0.255 expectancy, but only 52 of 303 baseline test trades — below the 76 floor). L3 was skipped (NO_MACRO_DATA). Path C legacy stays as production fallback.

## Commits (newest first)

```
26e32f5 docs(path-f): Path F final report + plan/spec + HANDOFF + audit subdir scan
22bd78b feat(path-f): eval wires B2/L2/L3 modes; engine dispatches on artifact tag
f2535d5 feat(path-f): macro trend slope + volatility features + NO_MACRO_DATA contract
5a809c6 feat(path-f): stationary feature space + dispatch + artifact tagging
8ddd6e7 fix(path-f): pick_best_threshold hard veto + tier_filter_result PF
```

## Key artifacts on disk

- `models_cache/path_c_stationary.joblib` — B2 (Path C trained on stationary set, `feature_set="stationary"` tag).
- `models_cache/path_f_stationary/path_e_expected_r_lgb.joblib` — L2 (Path E expected_R regressor, stationary set).
- `models_cache/path_f_stationary/path_d_{directional_a1,directional_a2,trade_outcome}_lgb.joblib` — produced as side artifacts during L2 training (we only use the expected_R one).
- Legacy `models_cache/path_c_lgb.joblib`, `models_cache/path_d_trade_outcome_lgb.joblib`, `models_cache/path_e_expected_r_lgb.joblib` are untouched.

## Key code locations

- `xau_pro_bot/models/features_stationary.py` — `STATIONARY_FEATURES` list + `build_stationary_features(tfs)`. No raw price columns.
- `xau_pro_bot/models/path_d_harvest.py` — `HarvestConfig.feature_set: Literal["legacy","stationary"]`; dispatch at the two `build_*_features` call sites; macro features extended with slope/vol.
- `xau_pro_bot/models/train_lightgbm.py` — `build_training_dataset(feature_set=...)` and `save_model(model, path, feature_cols=, feature_set=)`. Bundle format is dict when feature_cols supplied, raw model otherwise (back compat).
- `xau_pro_bot/models/train_path_d.py` — `save_model(..., feature_set=...)` writes dict bundles with the tag.
- `xau_pro_bot/models/hf_model.py` — `HFTradingModel._load_sklearn` unwraps dict bundles, surfaces `self.feature_set` + `self.feature_cols`.
- `xau_pro_bot/models/expected_r_filter_model.py` — reads `feature_set` from bundle.
- `xau_pro_bot/signals/engine.py` — `_prime_feature_set(model)` helper at module level; both feature-build sites (ai_model line ~138 and filter_model line ~287) dispatch on the artifact-tagged feature_set with `build_stationary_features` for `"stationary"`, falling back to legacy.
- `scripts/eval_path_d.py` — `pick_best_threshold` returns `None` on no qualifying threshold; `tier_filter_result` synthesizes `pnl_r` and `equity_curve` from kept-tier `rr_values`; new kwargs `path_c_stationary`, `path_e_stationary`, `path_e_stationary_macro`, `dxy_csv`, `us10y_csv`; `_check_macro_csvs` enforces NO_MACRO_DATA contract.
- `scripts/train_path_c_model.py` and `scripts/train_path_d_model.py` — `--feature-set {legacy,stationary}` flag.

## Tests

All Path F tests pass (37+ tests across these files):
- `tests/test_features_stationary.py` (builder + dispatch + bundle unwrap)
- `tests/test_path_f_feature_audit.py` (regex audit gate, scans all stationary-tagged artifacts via `rglob` + `feature_set` filter)
- `tests/test_path_f_smoke.py` (synthetic-data E2E)
- `tests/test_path_d_eval.py` (picker veto + tier_filter PF/Expectancy regressions)
- `tests/test_macro_features.py` (slope/vol + NO_MACRO_DATA contract)

Full regression: `265 passed, 2 skipped` (the 2 skips are pre-existing; not introduced by Path F).

## Final verdict numbers (from eval `b557mjt2d`)

| mode | trades | PF | Expectancy | WR | MaxDD | verdict |
|------|------:|---:|-----------:|---:|------:|:---:|
| A_baseline | 303 | 0.809 | -0.076 | 0.298 | 33.28 | reference |
| H_no_weak (bug-fixed) | 71 | 0.873 | -0.088 | 0.310 | 15.00 | reference |
| I_strong_only (bug-fixed) | 6 | 1.492 | +0.246 | 0.500 | 1.00 | reference |
| B_path_c (legacy) | 161 | 1.109 | +0.041 | 0.362 | 10.02 | production |
| L_path_e_expected_r (legacy) | 0 | — | — | — | — | NO-GO (picker veto) |
| **B2_path_c_stationary** | 198 | 1.051 | +0.019 | 0.354 | 13.81 | **GO (marginal)** |
| **L2_path_e_stationary** | 52 | 1.699 | +0.255 | 0.472 | 6.91 | **NO-GO (kept floor)** |
| L3_path_e_stationary_macro | — | — | — | — | — | skipped (NO_MACRO_DATA) |

Min kept floor (25% of A_baseline=303) = 76 trades.

## What to do next (ordered by leverage)

1. **Source DXY / US10Y CSVs and run L3.** This is the highest-leverage move. L2 demonstrates the stationary feature space carries real signal on its keepers; macro context targets the regime-shift dimension. Free sources: Stooq (`https://stooq.com/q/?s=dx.f` for DXY) or FRED for US10Y. Drop CSVs with columns `timestamp,close` under `./data/` and rerun:
   ```bash
   # Retrain L3 with macro features
   PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
     --csv data_long_m15.csv \
     --out-dir models_cache/path_f_stationary_macro \
     --feature-set stationary \
     --dxy-csv ./data/dxy_h1.csv \
     --us10y-csv ./data/us10y_h1.csv \
     --expected-r --allow-degenerate
   # Then eval with --path-e-stationary-macro + --dxy-csv + --us10y-csv
   ```
2. **Widen the threshold sweep.** Current `EXPECTED_R_THRESHOLDS = (0.00, 0.03, 0.05, 0.10, 0.15)` is sparse; add 0.02 and 0.04 to see if L2 admits 60-80 test trades at PF > 1.3. One-line change in `scripts/eval_path_d.py`.
3. **Use L2 as analysis assistant, not a gate.** Notify on `predicted_R ≥ 0.05` instead of blocking signals below it. Preserves Path C coverage while surfacing L2 calibration as a side-channel.
4. **Promote B2 if a longer test window confirms PF > legacy Path C.** Current run shows B2 1.051 vs legacy 1.109 on this re-run — B2 is below in the live comparison, only above the predecessor-report 1.011 reference. Not enough lift to promote.

## Reproduction (full)

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

# Run full eval (B + B2 + L_old + L2; L3 will skip with NO_MACRO_DATA)
PYTHONPATH=. .venv/bin/python -m scripts.eval_path_d \
    --csv data_long_m15.csv \
    --path-c models_cache/path_c_lgb.joblib \
    --path-d-filter models_cache/path_d_trade_outcome_lgb.joblib \
    --path-e models_cache/path_e_expected_r_lgb.joblib \
    --path-c-stationary models_cache/path_c_stationary.joblib \
    --path-e-stationary models_cache/path_f_stationary/path_e_expected_r_lgb.joblib \
    --report docs/reports/_path_f_eval_raw.md \
    --metrics-json /tmp/path_f_eval_metrics.json

# Regression sweep
.venv/bin/pytest tests/test_path_f_smoke.py tests/test_path_f_feature_audit.py \
    tests/test_features_stationary.py tests/test_path_d_eval.py \
    tests/test_macro_features.py tests/test_non_ai_baselines.py \
    tests/test_low_threshold_sweep.py -q
```

## Open issues / forward-compat notes

- `TradeFilterModel` (used for Path D filter mode) does not expose `feature_set` attribute. If anyone later trains a Path D filter with `--feature-set stationary` and points the eval at it, `_prime_feature_set` falls through to "legacy" and the engine builds the wrong features. Path F doesn't use this path (we only train Path C and Path E with stationary), but it should mirror `ExpectedRFilterModel`'s tag-read for completeness. Trivial fix.
- `data_long_m15.csv` covers 2022-01-03 → 2025-09-30. Window may now extend past September 2025 — refresh dataset before any final promotion decision.
- Pre-existing slow test `tests/test_path_d_harvest_step1.py` (~33s) is sometimes flaky under full-suite timing pressure but passes in isolation. Not introduced by Path F.
- Pickled Path F bundles use joblib's default protocol. Verified loadable by `HFTradingModel._load_sklearn` (dict branch) and `ExpectedRFilterModel._load`.

## Where everything lives

- Spec: `docs/superpowers/specs/2026-05-14-path-f-stationary-macro-design.md`
- Plan: `docs/superpowers/plans/2026-05-14-path-f-stationary-macro.md`
- Final report: `docs/reports/path_f_stationary_macro_results.md`
- Raw eval output: `docs/reports/_path_f_eval_raw.md`
- Predecessor report: `docs/reports/path_e_expected_r_results.md`
- HANDOFF.md updated with one-line "Recent events" entry pointing here.

## How to resume

1. `git status` → working tree should be clean on `feature/hugging-face-ai-layer`.
2. `cat docs/HANDOFF.md` for the broader project context.
3. Read this file and `docs/reports/path_f_stationary_macro_results.md`.
4. Pick a next-step from the "What to do next" section above.
