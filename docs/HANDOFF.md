# XAU Pro Bot — Execution Handoff Log

> **Purpose:** This file is the source of truth for execution progress. If the active agent (Claude Opus 4.7) runs out of context, Codex (or another agent) picks up from here.
>
> **Plan path:** `docs/superpowers/plans/2026-05-11-xau-pro-bot.md`
> **Spec path:** `docs/superpowers/specs/2026-05-11-xau-pro-bot-design.md` (Revision 3)
> **Working dir:** `/Users/aktan.kasymalievicloud.com/Projects/self-projects/signals/`

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
| 0 | Scaffolding + git init                   | pending     | |
| 1 | Config module                            | pending     | |
| 2 | Test fixtures (conftest.py)              | pending     | |
| 3 | State module (SQLite)                    | pending     | |
| 4 | Data module (Twelve Data + cache)        | pending     | |
| 5 | Classic indicators                       | pending     | |
| 6 | ICT indicators                           | pending     | |
| 7 | SMC indicators                           | pending     | |
| 8 | Wyckoff (soft bias)                      | pending     | |
| 9 | SR levels helpers                        | pending     | |
| 10| Signal engine (5-layer)                  | pending     | |
| 11| Filters (dedup/ratelimit)                | pending     | |
| 12| Formatter                                | pending     | |
| 13| Bot entrypoint                           | pending     | |
| 14| Backtest                                 | pending     | |
| 15| README                                   | pending     | |
| 16| Acceptance smoke                         | pending     | |
| 17| Pip constant + state migration           | pending     | R3 |
| 18| SR zones module                          | pending     | R3 |
| 19| SR zones into intraday engine            | pending     | R3 |
| 20| Swing stream                             | pending     | R3 |
| 21| Scalp stream                             | pending     | R3 |
| 22| StreamRouter + per-stream filters        | pending     | R3 |
| 23| Wire router + formatter labels           | pending     | R3 |
| 24| Per-stream backtest                      | pending     | R3 |

## Recent events

(append latest at the top)
- Task 0 in progress: scaffolded files, venv created, deps installed via `pandas-ta-classic` (see deviation).

## Deviations from plan

- **Task 0 / requirements:** `pandas-ta==0.3.14b` is no longer installable on Python 3.11 (pulled from PyPI; latest 0.4.x requires Python 3.12+). Switched to maintained fork **`pandas-ta-classic>=0.3.15,<0.5`** with `pandas>=2.2,<3.0`. The numpy<2.0 pin was dropped because pandas-ta-classic uses `numpy.nan` correctly. The `numpy.NaN` monkey-patch in `xau_pro_bot/indicators/__init__.py` is kept defensively but is a no-op with the new fork. All call sites import via `from xau_pro_bot.indicators import pandas_ta as ta` — the package re-exports `pandas_ta_classic` under the `pandas_ta` name to keep code identical to the plan.

## Known issues / TODO for follow-up agent

(empty for now)
