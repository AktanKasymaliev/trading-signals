# Railway Paper-Run Deployment

Long-running worker on [Railway](https://railway.app/) for the XAU
analysis-assistant bot. Paper-trading only — no broker execution.

## Prerequisites

- Railway account with a Hobby plan (or higher) for always-on workers.
- Telegram bot token + target chat ID.
- Twelve Data API key (free tier is fine for paper-run).
- Local working copy of this repo on the branch you want to deploy.

## 1. Verify the build manifest

Already in the repo:

| File | Expected content |
|---|---|
| `Procfile` | `worker: python -m xau_pro_bot.bot` |
| `runtime.txt` | `python-3.11.10` |
| `requirements.txt` | python-telegram-bot, apscheduler, lightgbm, joblib, … |

If you fork or rename, keep `Procfile` exactly as above. Railway
detects `worker:` and runs it as a non-web process — no `PORT` binding
required.

## 2. Required environment variables

Set these in Railway → *Variables* (or via `railway variables set`).
**Never paste a real token into a git-tracked file or a chat message.**

| Variable | Value | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | from @BotFather | Treat as secret |
| `TELEGRAM_CHAT_ID` | numeric chat id | Where signals are sent |
| `TWELVE_DATA_API_KEY` | from twelvedata.com | Treat as secret |
| `AI_ENABLED` | `true` | Path C legacy gate |
| `AI_EXPLAIN` | `true` | Analysis-assistant mode |
| `AI_FEATURE_SET` | `internal` | 29-feature legacy set |
| `AI_MODEL_TYPE` | `sklearn` | joblib-loaded LightGBM |
| `AI_MODEL_LOCAL_PATH` | `./models_cache/path_c_lgb.joblib` | Path C artifact |
| `AI_MIN_CONFIDENCE` | `0.65` | Production default |
| `AI_STRONG_CONFIDENCE` | `0.75` | Production default |
| `AI_NO_TRADE_THRESHOLD` | `0.60` | Production default |
| `STATE_DB_PATH` | `/data/state.db` | If volume is mounted at `/data`; else omit and accept reset on redeploy |
| `DAILY_REPORT_ENABLED` | `false` for the first paper-run, flip to `true` once daily volume looks healthy | Sends report at 23:55 UTC |

Verify locally with `env | grep AI_` (the local `.env` is *not*
uploaded — Railway env wins). Do not commit `.env`.

### Token security

- Rotate `TELEGRAM_BOT_TOKEN` and `TWELVE_DATA_API_KEY` if either was
  ever pasted into a chat, ticket, or CI log.
- Use Railway → *Variables → Sealed* for the two secret keys so they
  are write-only via the dashboard.
- Avoid `railway logs` over screen-share with secrets visible.

## 3. Model artifact deployment

`models_cache/` is in `.gitignore`, so the trained Path C artifact
**will not be present on Railway out of the box**. Pick one of the
three options below. For the first Railway paper-run, **Option A is
the recommended path** — Path C `path_c_lgb.joblib` is ~1.7 MB and
not sensitive, so committing it via a single gitignore exception is
the smallest moving-parts setup.

### Option A — Commit the artifact (recommended for first paper-run)

1. Add a gitignore exception locally:

   ```
   # .gitignore
   models_cache/
   !models_cache/path_c_lgb.joblib
   ```

2. Force-add (the directory rule still applies to siblings):

   ```bash
   git add -f models_cache/path_c_lgb.joblib .gitignore
   git commit -m "chore(deploy): include Path C legacy model artifact"
   ```

3. Push the branch to GitHub; Railway will pick it up automatically.

Trade-off: the model rides in git history. Re-trainings produce new
commits. Acceptable while we only have Path C in production.

### Option B — Download from Hugging Face on startup

If/when we promote a larger artifact, switch to a pinned HF release.
This requires a small wrapper that downloads on first boot and caches
to disk; not in scope for this commit. Track as future work.

### Option C — Railway volume pre-populated at deploy

Mount `/data` as a persistent volume and `scp` (or `railway run`) the
joblib once. Then point `AI_MODEL_LOCAL_PATH=/data/path_c_lgb.joblib`.
Useful when artifacts grow beyond what's reasonable in git but you
still want zero network dependency at startup.

Do **not** deploy silently without verifying the artifact path. The
bot logs `path_c_lgb` model name on every signal; a startup that loads
nothing should be caught in the smoke step below.

## 4. State persistence

`state.db` holds dedup history and the full lifecycle of every sent
signal (status, MFE/MAE, final R). Two options:

1. **With Railway volume** (recommended for paper-run analysis):
   - Add a volume mounted at `/data`.
   - Set `STATE_DB_PATH=/data/state.db`.
   - Lifecycle history survives redeploys, so daily/weekly reports
     keep accumulating signal.

2. **Without volume**: leave `STATE_DB_PATH` unset (defaults to
   `./state.db`). Every redeploy wipes the DB. Acceptable for a one-
   week smoke run; not acceptable if you want a real performance
   curve. The bot logs a fresh `state.db` creation at startup —
   that's the signal you lost history.

`xau_pro_bot/bot.py:393` already reads `STATE_DB_PATH` from env, so no
code change is needed for either option.

## 5. Smoke test after deploy

Right after the first successful deploy:

1. Open Railway → *Deployments → latest → Logs*. Confirm:
   - `Starting XAU Pro Bot…`
   - `Scheduler started.`
   - No traceback in the first 60 seconds.
   - On the first scheduled scan, the loaded AI model name appears
     (look for `path_c_lgb`).
2. In Telegram, send `/start`. Expect the welcome message listing all
   commands including `/active`, `/history`, `/daily_report`.
3. Send `/signal`. Either a signal is delivered (and persisted to
   `state.db`) or a "no signal in killzone" message is shown.
4. Send `/active`. With no live signals yet, expect `📭 Активных
   сигналов нет.`
5. Send `/daily_report`. With an empty DB, expect `Нет данных за
   период.`
6. Wait at least one full M15 scan cycle, then `/stats` should show
   `Активных: 0` and the AI risk breakdown (empty for the first day).

If any step fails, **do not rotate tokens blindly** — read the logs
first. A wrong `AI_MODEL_LOCAL_PATH` raises a clear
`FileNotFoundError`; a bad `TWELVE_DATA_API_KEY` shows `401` from
twelvedata.

## 6. Operating notes

### Logs

Railway logs are streamed live in the dashboard. The bot writes two
extra files locally (`errors.log`, `signals.log`) inside the
container; both rotate at 2 MB. To download a snapshot:

```bash
railway run -- cat errors.log
```

Do not leave `railway logs` open in a public screen-share — the
welcome banner contains the chat ID.

### Rollback

1. Find the last green deployment in *Deployments*.
2. Click *Redeploy* on that build.
3. If the bad deploy already updated `state.db` schema (it shouldn't —
   migrations are additive), restore the volume snapshot via
   Railway → *Backups* before redeploying.

### Disabling the bot temporarily

Set `TELEGRAM_BOT_TOKEN=disabled` (or any invalid value) and redeploy.
The polling loop will fail fast and the worker will idle without
sending or persisting anything. Restore the real token to resume.

## 7. What this deploy is **not**

- It is not a live trading bot. No broker connection, no order
  execution, no position sizing.
- The AI gate runs in analysis-assistant mode (`AI_EXPLAIN=true`). It
  annotates signals with risk labels; it does not size or skip trades
  for you.
- Path C is the only model loaded. Path F / Path E stay as research
  artifacts until a longer test window or macro L3 confirms a
  promotion (see `docs/superpowers/RESUME-2026-05-15-path-f.md`).
