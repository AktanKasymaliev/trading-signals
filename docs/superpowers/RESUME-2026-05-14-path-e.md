# Resume Note — Path E (2026-05-14)

> Read this first when resuming after `/clear` or a session break.

## Status

Tasks 1–8 of `docs/superpowers/plans/2026-05-14-path-e-expected-r.md` are **done and committed** on `feature/hugging-face-ai-layer`. Latest commit: `d0b6e8b` (Path E report section).

Task 9 (in-band training on `data_long_m15.csv`) was kicked off and is running in the background as bash task `b15kot3f1`:

```
PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
  --csv ./data_long_m15.csv --out-dir ./models_cache \
  --step-h1 1 --expected-r
```

Log: `/tmp/path_e_train.log`. Expected runtime ~30–50 min total.

## When you resume

1. `tail -50 /tmp/path_e_train.log` — check whether training finished.
2. `ls -la models_cache/path_e_expected_r_lgb.joblib` — exists ⇒ training succeeded.
3. `cat models_cache/path_d_metrics.json | jq '.expected_r'` — see regressor metrics + `feature_importance` (gain).

If the background bash died (terminal closed during my absence), re-run the same command (Bash, foreground or background). Harvest is deterministic given the CSV, so re-running is safe.

## After training succeeds — run eval (Task 9 step 2)

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_path_d.py \
  --csv ./data_long_m15.csv \
  --path-c ./models_cache/path_c_lgb.joblib \
  --path-d-filter ./models_cache/path_d_trade_outcome_lgb.joblib \
  --path-e ./models_cache/path_e_expected_r_lgb.joblib \
  --report docs/reports/path_e_expected_r_results.md
```

The printed JSON contains all modes (A_baseline, H_no_weak, B_path_c, E_path_d_filter, L_path_e_expected_r) plus `expected_r_sweep` and `chosen_expected_r_threshold`. Save those numbers for Task 10.

DXY/US10Y CSVs are unavailable locally (no `data/` directory). The macro-feature investigation gets a "data-unavailable, deferred" line in the verdict.

## Task 10 — verdict gates (must apply ALL)

Path E earns **GO** only if on the test slice:
1. Kept trades ≥ 25% of `A_baseline` test trades.
2. PF on test > PF of `B_path_c` **and** > PF of `H_no_weak`.
3. Expectancy on test > Expectancy of `H_no_weak`.
4. Threshold chosen on validation only — already enforced by `pick_best_threshold(min_kept=...)`.

Verdict template lives in the plan at Task 10 step 1. Write to `docs/reports/path_e_expected_r_results.md` (prepend above auto-generated tables). Append a "Path E (2026-05-14)" paragraph to `docs/HANDOFF.md`.

## Hard rules

- Do not push.
- Do not commit `.joblib` binaries unless the user explicitly asks — confirm first.
- Do not use `--no-verify`.
- Do not claim success on PF alone — gate table requires kept-floor + Expectancy + beating both B_path_c and H_no_weak.

## TaskList state

#1–8 completed; #9 in_progress; #10 pending.
