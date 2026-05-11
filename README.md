# XAU Pro Bot

Deterministic Telegram signal bot for XAU/USD using ICT, SMC, Wyckoff (soft bias), and classic TA confluence. No AI, no LLM, no broker execution.

After Revision 3 the bot is **multi-stream**: a single scan can emit up to three independent signals — `intraday`, `swing`, `scalp`.

## Features

- Multi-timeframe analysis: W1 / D1 / H4 / H1 / M15.
- 5-layer scoring engine (Macro / Structure / ICT / Classic / Penalties).
- Tiers: STRONG (≥65) / NORMAL (50–64) / WEAK (40–49).
- DST-aware killzones (America/New_York).
- SQLite persistence for dedup, rate limits, signal history.
- `AsyncIOScheduler`: 5 min in killzones, 15 min outside, daily DB prune.
- Walk-forward backtester for weight calibration.

## Quickstart (local)

```bash
git clone <repo>
cd signals
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Fill TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVE_DATA_API_KEY
pytest
python -m xau_pro_bot.bot
```

## Twelve Data API key

Free tier — register at <https://twelvedata.com> (8 req/min, 800 req/day). Symbol used: `XAU/USD` (spot).

## Telegram setup

1. `@BotFather` → `/newbot` → save token.
2. Start your bot in DM, send `/start`.
3. `@userinfobot` or `https://api.telegram.org/bot<token>/getUpdates` → `chat.id`.
4. Put both into `.env`.

## Railway deploy

```bash
railway login
railway init
railway variables set TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... TWELVE_DATA_API_KEY=...
railway up
```

`Procfile` runs `worker: python -m xau_pro_bot.bot`. `state.db` lives on the ephemeral disk — dedup history is reset between deploys (acceptable for v1).

## Backtesting (calibration gate)

Before trusting live signals, run the backtester on a CSV with H1 OHLCV history:

```bash
python -m xau_pro_bot.backtest --csv history_h1.csv
```

Sample output:

```
Signals:    142
Wins/Loss:  61 / 54 (timeouts 27)
Win rate:   53.0%
Expectancy: 0.34 R
Profit f.:  1.42
By tier:
  STRONG: n=38 wr=63.2%
  NORMAL: n=52 wr=51.9%
  WEAK:   n=25 wr=44.0%
```

**Acceptance gate:** STRONG ≥ 45% win rate AND expectancy > 0. If not, hand-tune weights in `signals/{ict,smc,classic}_signals.py` and rerun.

## Commands

| Command   | Description                                |
|-----------|--------------------------------------------|
| /start    | welcome + command list                     |
| /signal   | force analysis now, bypass dedup           |
| /status   | market overview (price, trends, killzone)  |
| /levels   | ICT/SMC level map                          |
| /help     | tier explanation                           |
| /settings | current thresholds                         |
| /stats    | today's signal counters                    |

## Signal format

```
🔴 Сильный сигнал — SELL
🔹 Вход: `3,312.50`
🔺 Stop Loss: `3,324.00` (+11.5 pts)
🎯 Цели:
 •  TP1: `3,298.00` (14.5 pts) — FVG
 •  TP2: `3,280.00` (32.5 pts) — ликвидность
 •  TP3: `3,261.00` (51.5 pts) — D1
📊 R:R → 1:2.8
🧠 Score: 81/100
```

`Score` is the internal confluence number, **not** a win-probability. Calibrate via backtest before treating it as one.

## Module map

- `data.py` — Twelve Data REST + TTL cache + retry.
- `state.py` — SQLite signals/dedup.
- `indicators/` — feature extraction (one file per concept group).
- `signals/engine.py` — 5-layer scoring (intraday stream).
- `signals/filters.py` — dedup, ATR-reprice, rate-limit.
- `formatter.py` — Telegram Markdown.
- `bot.py` — Telegram + AsyncIOScheduler.
- `backtest.py` — walk-forward replay + R-multiple metrics.

## ICT / SMC primer

- **OTE** — Optimal Trade Entry, 0.62–0.79 Fibonacci retracement zone.
- **FVG** — Fair Value Gap (3-candle price imbalance).
- **OB** — Order Block, last opposite candle before strong impulse.
- **Liquidity** — equal highs/lows attracting price.
- **BOS / CHOCH** — Break of Structure / Change of Character.
- **Premium / Discount** — top/bottom of recent range.
- **Killzone** — NY-time windows of historic volatility (London 02:00–05:00 NY, NY AM 08:30–11:00 NY).

## Limits and caveats

- Twelve Data free tier: 800 req/day. The 5-minute TTL cache + 5-min scan cadence keeps usage inside.
- SQLite is ephemeral on Railway — dedup state resets on each deploy.
- This is signal generation only — no broker execution, no position sizing.
- Weights are unverified until you run the backtester.

## Implementation status

See `docs/HANDOFF.md` for live progress, deviations (e.g. `pandas-ta-classic` fork), and any open TODOs.

## License

Private project.
