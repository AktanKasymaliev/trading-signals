# Operator smoke checklist — Path C legacy in AI_EXPLAIN mode

> ⚠️ **Manual analysis only. Do NOT use the bot output as auto-trading
> signals.** The AI block is an explanation aid: it tells you *why* a
> deterministic signal was kept, blocked, or downgraded so a human
> operator can decide. The AI gate is Path C legacy (PF 1.109, +0.041 R,
> 36.2% WR, 161 trades on the hold-out window) — useful as a second
> opinion, not as a trading authority.

Production model: **Path C legacy** (`AI_FEATURE_SET=internal`).
Research-only paths (Path D / E / F stationary): keep disabled.

---

## 1. Required `.env` values

```bash
# Telegram + market data (must already be set)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TWELVE_DATA_API_KEY=...

# AI gate — Path C legacy
AI_ENABLED=true
AI_EXPLAIN=true
AI_FEATURE_SET=internal
AI_MODEL_TYPE=sklearn
AI_MODEL_LOCAL_PATH=./models_cache/path_c_lgb.joblib

# Decision thresholds (defaults are sane)
AI_MIN_CONFIDENCE=0.65
AI_STRONG_CONFIDENCE=0.75
AI_NO_TRADE_THRESHOLD=0.60

# Leave these empty when using LOCAL_PATH:
AI_MODEL_ID=
AI_MODEL_REVISION=
```

**Sanity:** `AI_MODEL_LOCAL_PATH` takes precedence over `AI_MODEL_ID`. If
both are empty and `AI_ENABLED=true`, the engine logs a warning and
emits a `NO_TRADE` adjustment for every bar.

---

## 2. Local model path check

```bash
ls -lh ./models_cache/path_c_lgb.joblib
.venv/bin/python -c "
import joblib, sys
m = joblib.load('./models_cache/path_c_lgb.joblib')
print('type:', type(m).__name__)
print('feature_set:', getattr(m, 'feature_set', '(default → legacy)'))
print('n_features:', getattr(m, 'n_features_in_', '?'))
"
```

Expected:
- File exists, non-zero, recently modified.
- `feature_set` is unset (legacy) — Path C legacy intentionally does
  **not** tag itself `stationary`. If it prints `stationary`, you have
  the wrong bundle loaded; check Path F training output.
- `n_features_in_` matches `REQUIRED_AI_FEATURES` length in
  `xau_pro_bot/models/features.py` (29 at last check).

---

## 3. One backtest command

Run baseline vs. AI side-by-side on the local H1 CSV; verify Path C
legacy preserves PF ≥ 1.05 and blocks are non-zero on the hold-out
window.

```bash
.venv/bin/python -m xau_pro_bot.backtest \
    --csv data_xauusd_h1.csv \
    --stream intraday \
    --compare-ai \
    --use-ai \
    --ai-model-type sklearn
```

What to read in the printed summary:
- `signals_generated`, `wins`, `losses`, `timeouts`, `blocked_signals`.
- `profit_factor` ≥ 1.05 on the AI side (target: 1.109 ± noise).
- `blocked_signals` > 0 (gate must actually fire). 0 blocks = misconfig
  or wrong feature set.

For ad-hoc inspection of blocked signals, drop into a REPL after a run
and read `BacktestResult.blocked_details` — each entry has
`original_direction`, `tier_before_block`, `ai_reason`, `ai_action`,
`ai_risk_label`.

---

## 4. Telegram `/signal` manual test

1. Start the bot locally:
   ```bash
   .venv/bin/python -m xau_pro_bot.bot
   ```
2. In the configured Telegram chat send `/signal`.
3. Confirm the reply contains either a STRONG/WEAK signal block **with
   the AI explanation block** (see §5), or a "no signal" reply with
   killzone + RSI.

If the AI block is missing while `AI_ENABLED=true` and `AI_EXPLAIN=true`
are set, the model failed to load — check `signals.log` and
`errors.log`.

---

## 5. What a good AI block should look like

KEEP path (production happy case):

```text
🧠 AI filter: KEEP
Модель: Path C legacy
Риск: MEDIUM
Причина: направление совпадает, score gap нормальный, RR приемлемый
```

BLOCK path (gate refused):

```text
🧠 AI filter: BLOCK
Модель: Path C legacy
Риск: HIGH
Причина: AI conflicts with deterministic signal
```

DOWNGRADE path (AI direction != deterministic, no hard block):

```text
🧠 AI filter: DOWNGRADE
Модель: Path C legacy
Риск: HIGH
Причина: AI prefers opposite direction, penalty applied
```

Sanity checks per block:
- `Модель: Path C legacy` (NOT `Path F stationary` — that is research).
- `Риск:` ∈ {CLEAN, MEDIUM, HIGH}.
- `Причина:` ≤ 80 chars, no truncated UTF-8.
- For STRONG signals with no penalties and KEEP action → `Риск: CLEAN`.

---

## 6. Logs to inspect

Two log files in repo root:

```bash
tail -f signals.log
tail -f errors.log
```

`signals.log` — every signal/no-signal emitted, JSON-ish single line.
Look for:
- `tier=STRONG|NORMAL|WEAK|NO_SIGNAL`
- `ai_action`, `ai_blocked`, `ai_confidence`, `ai_pre_block_tier`
- `ai_model_name=Path C legacy`

`errors.log` — model load failures, Twelve Data outages, Telegram send
errors. If `AI_ENABLED=true` and you see repeated
`HFTradingModel._load` failures, the local path is wrong or the
artifact is corrupted.

Quick grep recipes:

```bash
grep '"ai_action": "BLOCK"'    signals.log | wc -l   # blocks today
grep '"ai_action": "DOWNGRADE"' signals.log | wc -l
grep -i 'HFTradingModel\|joblib' errors.log | tail
```

---

## 7. Explicit warning

**This is an analysis-assistant build, not auto-trading.**

- The bot does **not** place broker orders. It emits Telegram messages
  to a human reader.
- Path C legacy backtest PF is 1.109 on the hold-out window — useful as
  a filter, **not** a license to fire trades blindly.
- The AI block exists so the operator can read *why* a signal passed or
  was blocked, cross-check against their own view, and decide manually.
- Never wire `/signal` output into an auto-execution layer without a
  separate eval phase, capital risk gate, and human approval step.

If anyone reading this is tempted to "just hook it up to MT5" — stop,
re-read the Path F report (`docs/reports/path_f_stationary_macro_results.md`)
and the Path C numbers above. The edge is too thin for unattended
execution.
