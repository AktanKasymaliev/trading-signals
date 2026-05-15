# Railway Deploy from GitLab (Docker)

Deploy the XAU paper-run worker to Railway, sourced from a GitLab
repository, built with the project `Dockerfile`. Companion to
`docs/deploy_railway_paper_run.md` — that doc covers env vars,
secrets, smoke checks and rollback; this one focuses on the GitLab
specifics and the Docker build.

## 1. One-time GitLab → Railway connection

Railway officially supports GitLab as a source (group + project tokens
under the hood). Order matters:

1. Sign in to https://railway.app → *New Project*.
2. Choose *Deploy from GitLab repo*. If the option is missing or
   greyed-out, go to *Account Settings → Integrations → GitLab* and
   complete the OAuth handshake first. Self-hosted GitLab needs a
   custom URL and a personal access token with `read_repository`.
3. Pick the repo and the branch you want Railway to track (typically
   `main` or the current `feature/hugging-face-ai-layer` branch for a
   paper-run smoke).
4. Railway will create one Service per repo. Rename it to something
   like `xau-paper-run` so the dashboard stays readable.

Railway autodeploys on every push to the tracked branch. Disable that
in *Service → Settings → Service → Deployment Triggers* if you prefer
manual deploys for paper-run safety.

## 2. Build configuration (already in the repo)

| File | Purpose |
|---|---|
| `Dockerfile` | python:3.11-slim base, installs requirements, copies `xau_pro_bot/` and `models_cache/path_c_lgb.joblib`, runs `python -m xau_pro_bot.bot` |
| `.dockerignore` | Excludes `.venv`, `tests/`, `docs/`, `scripts/`, logs, `state.db`, datasets, and *all* of `models_cache/` except `path_c_lgb.joblib` |
| `railway.toml` | Tells Railway to use `Dockerfile` (not Nixpacks) and restart on failure up to 10 times |
| `Procfile` | Kept for non-Docker deploys; ignored when Dockerfile is detected |

Verify the image builds locally before pushing:

```bash
docker build -t xau-pro-bot:dev .
docker run --rm \
    -e TELEGRAM_BOT_TOKEN=fake -e TELEGRAM_CHAT_ID=0 \
    -e TWELVE_DATA_API_KEY=fake \
    -e AI_ENABLED=true -e AI_EXPLAIN=true \
    -e AI_FEATURE_SET=internal -e AI_MODEL_TYPE=sklearn \
    -e AI_MODEL_LOCAL_PATH=./models_cache/path_c_lgb.joblib \
    xau-pro-bot:dev python -c "from xau_pro_bot.signals.ai_gate import AIExplanationGate; print(AIExplanationGate().ai_enabled)"
```

A successful run prints `True` and exits. The actual Telegram polling
will fail with a fake token, which is exactly what you want for a
build-only sanity check.

## 3. Model artifact

`models_cache/` is in `.gitignore`. The `Dockerfile` `COPY` line
references `models_cache/path_c_lgb.joblib` explicitly — if that file
is missing from the repo, the Railway build fails with a clear
`COPY failed` error.

For the first paper-run, commit the artifact via a gitignore
exception (1.7 MB, low risk):

```
# .gitignore
models_cache/
!models_cache/path_c_lgb.joblib
```

```bash
git add -f models_cache/path_c_lgb.joblib .gitignore
git commit -m "chore(deploy): include Path C legacy model artifact"
git push
```

Larger artifacts (Path D / Path F) stay out of git — switch to the
Hugging Face download or Railway volume options described in
`docs/deploy_railway_paper_run.md` when they need to ship.

## 4. Environment variables

Set in Railway → *Service → Variables*. Minimum set:

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TWELVE_DATA_API_KEY=...
AI_ENABLED=true
AI_EXPLAIN=true
AI_FEATURE_SET=internal
AI_MODEL_TYPE=sklearn
AI_MODEL_LOCAL_PATH=./models_cache/path_c_lgb.joblib
AI_MIN_CONFIDENCE=0.65
AI_STRONG_CONFIDENCE=0.75
AI_NO_TRADE_THRESHOLD=0.60
STATE_DB_PATH=/data/state.db
DAILY_REPORT_ENABLED=false
```

Mark `TELEGRAM_BOT_TOKEN` and `TWELVE_DATA_API_KEY` as *Sealed*
(Railway → variable → menu → Seal). Sealed variables are write-only
from the dashboard and never appear in build logs.

If you keep secrets in GitLab CI variables for any reason, **never
mirror them into the repo or into `.env.example`**. Railway is the
source of truth at runtime.

## 5. Persistent volume

Lifecycle history (`state.db`) and the signal log live on disk. Add a
volume so they survive redeploys:

1. *Service → Volumes → New Volume*. Mount path `/data`. 1 GB is more
   than enough for a multi-month paper-run.
2. Confirm `STATE_DB_PATH=/data/state.db` is set (see env list above).
3. First boot will create `/data/state.db` automatically and migrate
   the schema in place.

Without a volume, every redeploy wipes lifecycle history. The
`/daily_report` and `/weekly_report` outputs will reset accordingly.

## 6. Deploy and smoke

1. Push to the branch Railway is tracking. The *Deployments* tab will
   show the Docker build streaming. A green build takes ~2-4 minutes
   on a cold cache, ~30 seconds when layers are cached.
2. Once the deploy is *Active*, follow the smoke checklist in
   `docs/deploy_railway_paper_run.md` §5 — same set of `/start`,
   `/signal`, `/active`, `/daily_report` checks, no GitLab-specific
   differences.

## 7. Rollback

Two paths:

- **Railway-native**: *Deployments → previous green build →
  Redeploy*. The container image is already cached; rollback takes
  seconds.
- **GitLab-native**: revert the bad commit on the tracked branch
  (`git revert <sha>` then `git push`). Railway detects the new
  commit and rebuilds. Slower but the history stays linear.

For schema-incompatible regressions on `state.db`, restore the
Railway volume snapshot *before* redeploying — otherwise the older
worker may try to run additive migrations in reverse and crash on
startup.

## 8. GitLab CI (optional)

You do **not** need GitLab CI to deploy — Railway is connected
directly to the repo. CI is only useful if you want pre-deploy gates:

```yaml
# .gitlab-ci.yml (sketch)
stages: [test]

pytest:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install -r requirements.txt -r requirements-dev.txt
  script:
    - pytest -x -q
  only:
    - merge_requests
    - main
```

Pin Railway autodeploys to `main` (or a `release/*` branch) and run
the pytest job on merge requests; the bad branch never makes it to
the deploy trigger. Not in this commit — add when you have a CI
runner attached to the project.

## 9. Non-goals (unchanged from main deploy doc)

- No broker execution.
- Path C is the only model loaded; Path F / Path E remain research
  artifacts.
- The bot operates in analysis-assistant mode (`AI_EXPLAIN=true`).
