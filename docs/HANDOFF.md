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
