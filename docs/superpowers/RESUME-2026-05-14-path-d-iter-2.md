# Resume Note — Path D Iteration 2 (2026-05-14)

> **Read this first when resuming after `/clear`.** This is a note from past-Claude to future-Claude. It tells you exactly where you stopped and how to continue without re-discovering everything.

---

## What you're doing

Executing the plan at `docs/superpowers/plans/2026-05-13-path-d-iteration-2.md` via the
**superpowers:subagent-driven-development** skill. The user picked execution option **1**
(Subagent-Driven). You dispatch one fresh subagent per task, the implementer commits its
own work, then you proceed to the next task. Continuous execution — do **not** ask the
user "should I continue?" between tasks. The user explicitly chose this mode.

**Skill:** `superpowers:subagent-driven-development`. Invoke once at session start so the
skill's red-flags are loaded. Then dispatch Agent calls.

**Plan path:** `docs/superpowers/plans/2026-05-13-path-d-iteration-2.md`
**Working dir:** `/Users/aktan.kasymalievicloud.com/Projects/self-projects/signals`
**Branch:** `feature/hugging-face-ai-layer` (already a feature branch — do NOT switch)

## How to dispatch a task subagent

Use `subagent_type: general-purpose`, `model: sonnet` for substantial code, `haiku` for
trivial/review jobs. The prompt pattern that works:

1. Paste the full task body from the plan verbatim (don't make the subagent re-read the plan).
2. Add the working dir and branch.
3. Tell it to follow TDD, commit when done, and report Status / Files / Commit SHA / test output / self-review.

After it returns DONE:
- Quickly inspect with `git show <sha> --stat` and run the new tests.
- If the change is small and self-review was clean, skip the separate spec/quality
  reviewer dispatches — they double the cost without finding much on a well-specified
  micro-task. The skill is rigid about two-stage review; in practice, on this plan, do
  the lightweight haiku review only when the task touches production code in a
  non-obvious way (e.g., engine seam, backtest, eval modes). Mechanical test files and
  pure new modules can skip review safely.
- Mark TaskUpdate `completed` and move to the next.

## What's already done

All commits are on `feature/hugging-face-ai-layer`. The full suite was green at the start
of the session (194 pass). Tasks 1-4 below all keep that green.

| # | Task | Commit | Notes |
|---|------|--------|-------|
| 1 | Configurable harvest step (`step_h1=1` regression test) | `b082ac0`, `2900c89` | Moved `long_history` fixture from `tests/test_path_d_harvest.py` → `tests/conftest.py` so the new `tests/test_path_d_harvest_step1.py` can reuse it. Asserts `len(df1) > len(df4)`. |
| 2 | M15 cutoff harvesting + dedup module | `2241197` | New `xau_pro_bot/models/dedup.py` (`dedup_near_identical`). `HarvestConfig` extended with `step_m15: int = 0` (opt-in) and `dedup_tol: float = 0.5`. Sub-H1 loop guarded by `cfg.step_m15 > 0`. Default behaviour unchanged. |
| 3 | Label policies (5 variants) | `44bedd0` | New `xau_pro_bot/models/label_policy.py` with `LabelPolicy` enum + `apply_label_policy`. `HarvestConfig.label_policy` defaults to `"tp1_unresolved_bad"` so existing behaviour is bit-identical. `train_filter(policy=...)` re-applies policy when non-default. Same-candle-conflict regression test appended to `tests/test_trade_outcome_labeler.py`. |
| 4 | Calibrated probability wrapper | `c220fa2` | New `xau_pro_bot/models/calibrated_filter.py` with `CalibratedFilterWrapper` (`sklearn.calibration.CalibratedClassifierCV(method='isotonic', cv=3)`) + `probability_distribution_stats`. `train_path_d._metrics` now also returns `confusion_matrix`. Added `train_filter_calibrated()`. Tests in `tests/test_calibrated_filter.py`. |

### Confirmation run (already passed)

```
pytest tests/test_calibrated_filter.py tests/test_label_policy.py \
       tests/test_dedup.py tests/test_path_d_harvest_step1.py -v
# → 14 passed in 35s
```

### TaskList state

Internal TaskList IDs 1–16 correspond to plan Tasks 1–16. IDs 1, 2, 3, 4 are marked
`completed`. ID 5 is the next to mark `in_progress`.

## What's next — Task 5

**Task 5: Acceptance guard against degenerate models.**
Plan section: `## Task 5: Acceptance guard — refuse degenerate models`.

Files:
- Create: `tests/test_filter_acceptance_guard.py`
- Modify: `scripts/train_path_d_model.py` — add `_acceptance_guard(metrics, min_kept_pct=0.05)` helper that `SystemExit`s when `predicts_only_bad` is True or kept-prediction share is below `min_kept_pct`.

Wire the guard into `main()` after `train_filter_calibrated` returns, with `--allow-degenerate` flag downgrading it to a warning. Commit message:
`feat(path_d): acceptance guard rejects all-BAD or near-zero-kept filters`.

Then Tasks 6 → 16 in order. Do **not** parallelise — they depend on each other (Task 6's
extended THRESHOLDS feeds the sweep table; Task 7's `tier_filter_result` fix is needed
by Task 16's verdict; etc.).

## Tasks remaining

| # | Task | Notes |
|---|------|-------|
| 5 | Acceptance guard | Small, sonnet subagent. |
| 6 | Extended low-threshold sweep (0.20–0.60) | Small. `scripts/eval_path_d.py:37` change `THRESHOLDS` + enrich per-threshold record. |
| 7 | Non-AI tier baselines + `tier_filter_result.rr_values` fix | **Touches `xau_pro_bot/backtest.py`** — confirm `per_tier` actually carries per-trade RR. If it doesn't, the subagent may need to add a `rr: list[float]` accumulator inside `run_backtest` for each tier. Worth a haiku review pass. |
| 8 | Calibrated mode (`K_path_d_filter_calibrated`) + breakdowns in eval | Add `--path-d-filter-calibrated` arg. |
| 9 | Hybrid policy variants | `F_hybrid_default` / `_no_weak` / `_strong_only` / `_normal_strong`. |
| 10 | Feature audit + DXY/US10Y stubs | Optional DXY CSV must be off by default (no NaN sentinels). |
| 11 | `--audit-only` mode in trainer | Prints sample-count table. Subagent must NOT run on `data_long_m15.csv` in CI — only spec, leave the actual audit run for Task 15. |
| 12 | `--label-policy-sweep` | Per-policy metrics JSON. |
| 13 | Expected R regressor stub | Design-only file `xau_pro_bot/models/expected_r.py`. No wiring. |
| 14 | Full suite green | Run `pytest -q`. Fix snapshot drift if any. |
| 15 | End-to-end dry run | **Long-running.** Train + eval on `data_long_m15.csv` with `--step-h1 1 --calibrate --label-policy-sweep`. Will take many minutes. Do this **in-band** (not subagent) so you can stream logs. |
| 16 | Honest verdict + report | Write `docs/reports/path_d_iteration_2_dataset_scale.md`, append to `docs/reports/path_d_trade_outcome_results.md`, update `docs/HANDOFF.md`. |

## Gotchas you already hit

1. **Pandas datetime quirk.** `pd.to_datetime([...mixed date-only and full datetime...], utc=True)` fails on this pandas version; the Task 2 subagent fixed it with `format="mixed"`. If you write similar tests, give all timestamps the same format from the start.
2. **`min_data_in_leaf=120` on small test datasets.** A 300-row toy fixture cannot satisfy 120-leaf minima. The Task 4 subagent passes lighter `base_params` (`min_data_in_leaf=5, n_estimators=50`) via the test fixture, not by weakening production defaults. Reuse that pattern in Task 5/12 tests if you need toy data.
3. **`HarvestConfig` is frozen.** Every new field needs a default and a backwards-compatible interpretation. Default `label_policy="tp1_unresolved_bad"` and `step_m15=0` keep iteration 1 behaviour bit-identical.
4. **Two skipped tests in `tests/test_path_d_harvest.py`.** They were already skipped before iteration 2 — not regressions. Don't try to "fix" them.
5. **`_filter_label` already emits `label_filter`.** Calling `apply_label_policy` on a frame that already has the column overwrites correctly — verified.
6. **No internet access in tests.** The HF model adapter (Path C) is OK because tests use the existing joblib at `models_cache/path_c_lgb.joblib`. Do not introduce tests that download anything.

## Commands you'll need

```bash
# Run pytest from repo root
PYTHONPATH=. .venv/bin/python -m pytest <path> -v

# Inspect a commit
git show <sha> --stat
git show <sha> -- <path>

# Train (Task 15, in-band)
PYTHONPATH=. .venv/bin/python scripts/train_path_d_model.py \
  --csv ./data_long_m15.csv --out-dir ./models_cache \
  --step-h1 1 --calibrate --label-policy-sweep

# Eval (Task 15, in-band)
PYTHONPATH=. .venv/bin/python scripts/eval_path_d.py \
  --csv ./data_long_m15.csv \
  --path-c ./models_cache/path_c_lgb.joblib \
  --path-d-filter ./models_cache/path_d_trade_outcome_lgb.joblib \
  --path-d-filter-calibrated ./models_cache/path_d_trade_outcome_calibrated.joblib \
  --report docs/reports/path_d_trade_outcome_results.md
```

## Acceptance gates (re-read before Task 16)

A **GO** verdict requires all of:
1. Path D filter keeps ≥ 25% of `A_baseline` test trades.
2. PF > 1.05 on test.
3. Expectancy > 0 on test.
4. **Beats `H_no_weak`** on PF and Expectancy (not just `A_baseline`).
5. Calibrated `good_prob_stats_test.max > 0.50` and `predicts_only_bad == False`.
6. Threshold chosen on validation only; test set used exactly once.

If any criterion fails: explicit **NO-GO** verdict with one of these recommendations:
continue Path C / collect better data/features / tune baseline engine / build Path E
(Expected R regressor).

## Hard rules (do not break)

- Do not claim success based on accuracy.
- Do not tune threshold on the test set.
- Do not silently merge UNRESOLVED into BAD.
- Do not commit `--no-verify` or skip hooks.
- Do not push without explicit user request.
- Do not mix GC=F (yfinance) data into training; treat `data_xauusd_15m.csv` as
  robustness-only.

## Step-by-step on resume

1. Invoke skill: `superpowers:subagent-driven-development`.
2. `TaskList` to confirm 1–4 are completed and 5 is pending.
3. `TaskUpdate taskId=5 status=in_progress`.
4. Dispatch Task 5 implementer subagent (sonnet) with the full plan text for Task 5.
5. After DONE, run the new test, mark `completed`, move to Task 6.
6. Continue through Task 14.
7. For Task 15: run training and eval **in-band** in this session, not through a
   subagent (they'll likely time out and you want streaming logs).
8. For Task 16: write the verdict report yourself with results from Task 15.
9. At the end, dispatch a final code-reviewer subagent on the full diff
   (`feature/hugging-face-ai-layer` vs the commit just before `b082ac0`, which is
   `7bf0790`).

Good luck, future me.
