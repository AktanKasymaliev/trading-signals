# XAU Pro Bot Design

Date: 2026-05-11
Revision: 3 (multi-stream architecture + S/R zones + swing + scalp)

## Goal

Профессиональный одноюзерный Telegram-сигнальный бот по XAU/USD на основе детерминированного алгоритмического анализа: ICT, Smart Money Concepts, Wyckoff и классический технический анализ. Без AI/LLM, без подписок, без бэкендовой БД (только локальный SQLite для состояния).

v1 — runnable MVP полной структуры из исходного ТЗ, c обязательным `backtest.py` для калибровки весов скоринга на исторических данных перед выходом на живые сигналы.

## Scope

In scope:

- Python 3.11, проект `xau_pro_bot/`.
- Один приватный Telegram-чат (`TELEGRAM_CHAT_ID`).
- Railway worker (long-running).
- **Twelve Data API** как основной источник OHLCV для XAU/USD spot (free tier — 8 req/min, 800 req/day).
- TF: W1, D1, H4, H1, M15. In-memory cache TTL 5 минут + retry(3, delay=5s).
- Классические индикаторы, ICT, SMC, Wyckoff, S/R helpers.
- Master scoring engine с tiered порогами, penalties, RR validation, dedup, ATR-reprice, rate-limit.
- Telegram-команды: `/start`, `/signal`, `/status`, `/levels`, `/help`, `/settings`, `/stats`.
- AsyncIOScheduler: 5 мин в киллзонах, 15 мин вне, плюс сессионные алерты.
- Markdown-форматирование на русском под исходный визуал.
- SQLite `state.db` для дедупа и истории сигналов.
- `signals.log` + `errors.log`.
- **`backtest.py`** — обязательный модуль v1, прогон последних 12 месяцев и метрики (win rate, expectancy, Sharpe, max DD) для каждой тира сигнала.
- README с setup, Railway deploy, интерпретацией сигналов.

Out of scope для v1:

- Мультиюзер/подписки.
- AI/LLM.
- Платежи, админка.
- Параметрическая оптимизация (Bayesian/grid search) — только ручная калибровка по бэктесту.
- Брокерское исполнение.

## Architecture

```text
xau_pro_bot/
├── bot.py
├── data.py
├── state.py                # SQLite persistence (dedup, signal history)
├── indicators/
│   ├── __init__.py         # numpy.NaN monkey-patch перед import pandas_ta
│   ├── classic.py
│   ├── ict.py
│   ├── smc.py
│   ├── wyckoff.py
│   └── sr_levels.py
├── signals/
│   ├── engine.py
│   ├── ict_signals.py
│   ├── smc_signals.py
│   ├── classic_signals.py
│   └── filters.py
├── formatter.py
├── backtest.py             # обязательно для v1
├── config.py
├── requirements.txt
├── runtime.txt
├── Procfile
└── .env.example
```

`data.py` — Twelve Data REST client, OHLCV-нормализация, retry, in-memory TTL-cache.
`state.py` — SQLite-обёртка: таблицы `signals`, `dedup_state`, `daily_counter`.
`indicators/` — детерминированные фичи.
`signals/` — оценка confluence, фильтры, выбор уровней.
`formatter.py` — Markdown для Telegram.
`backtest.py` — replay движка на исторических барах + метрики.
`bot.py` — Telegram, scheduler, lifecycle.
`config.py` — пороги, окна, ENV.

### pandas-ta совместимость

`indicators/__init__.py` выполняет monkey-patch **до** любого импорта `pandas_ta`:

```python
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas_ta  # noqa: E402, F401
```

Все модули `indicators/*` импортируют `pandas_ta` только через `indicators/__init__.py` или после собственного guard'а.

## Data Source: Twelve Data

Базовый URL: `https://api.twelvedata.com/time_series`.

Symbol: `XAU/USD` (spot, не futures).

Интервалы и retention:

| TF  | interval | outputsize | Назначение     |
|-----|----------|------------|----------------|
| W1  | `1week`  | 104        | Wyckoff macro  |
| D1  | `1day`   | 365        | bias           |
| H4  | `4h`     | 540        | structure      |
| H1  | `1h`     | 720        | entry          |
| M15 | `15min`  | 672        | precision      |

Параметры запроса: `apikey=<TWELVE_DATA_API_KEY>`, `format=JSON`, `timezone=UTC`.

`data.py`:

- `fetch_all_timeframes() -> dict[str, pd.DataFrame]` возвращает 5 DataFrame с колонками `Open, High, Low, Close, Volume` (Volume может быть NaN — Twelve Data spot его не отдаёт; volume-условия в этом случае skip).
- Retry: 3 попытки, 5s между ними, экспоненциальный jitter.
- TTL-кэш 5 минут в памяти (`(tf, last_fetched)`).
- Rate-limit budget: free tier даёт 8 req/min, 800 req/day. На каждом скане — 5 запросов (по одному на TF). При TTL=5мин и скане 5мин в киллзоне = 60 req/час = в пределах лимита.
- Если API возвращает `code != 200` или `status == "error"` → лог в `errors.log`, скан skip, scheduler продолжает.

## Persistence: SQLite `state.db`

`state.py`:

```sql
CREATE TABLE signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,           -- ISO8601
  direction TEXT NOT NULL,        -- BUY/SELL
  tier TEXT NOT NULL,             -- STRONG/NORMAL/WEAK
  score INTEGER NOT NULL,
  entry REAL NOT NULL,
  sl REAL NOT NULL,
  tp1 REAL,
  tp2 REAL,
  tp3 REAL,
  rr REAL,
  killzone TEXT,
  reasons_json TEXT
);

CREATE INDEX idx_signals_ts ON signals(ts_utc);
```

`dedup_state` и `daily_counter` хранятся в той же БД (single-row tables или query по `signals`).

API модуля:

- `record_signal(sig: dict) -> int`
- `last_signal(direction: str | None) -> dict | None`
- `count_today(tier: str | None) -> int`
- `last_weak_ts() -> datetime | None`
- `prune_old(days: int = 90)` — вызывается раз в сутки.

Файл `state.db` живёт рядом с процессом. На Railway это эфемерный диск — допустимо для dedup, но при деплое state сбрасывается. Это явный trade-off v1; для production-grade нужен volume mount или Postgres (v2).

## Timezone & Killzones

Все киллзоны привязаны к **America/New_York** через `zoneinfo.ZoneInfo("America/New_York")`, что автоматически даёт корректные сдвиги по DST.

```python
KILLZONES_NY = {
    "Asian KZ":   (time(20, 0), time(23, 59)),   # 20:00–24:00 NY (Asia overnight)
    "London KZ":  (time(2, 0),  time(5, 0)),     # 02:00–05:00 NY ★ priority
    "NY AM KZ":   (time(8, 30), time(11, 0)),    # 08:30–11:00 NY ★ priority
    "NY PM KZ":   (time(13, 30), time(16, 0)),   # 13:30–16:00 NY
}
```

`get_killzone(now: datetime | None = None) -> str | None`:

- Если `now` None → `datetime.now(ZoneInfo("America/New_York"))`.
- Если naive datetime → ошибка `ValueError`.
- Сравнивает только `time()` часть в NY-зоне.

## Indicator Design

### Classic (`indicators/classic.py`)

`add_classic(df) -> pd.DataFrame`:

- EMA 8, 21, 50, 200.
- RSI 14.
- MACD 12/26/9.
- Stoch 14/3.
- BBands 20/2.0.
- ATR 14.
- `vol_ratio` = `volume / volume.rolling(20).mean()` — **NaN если volume отсутствует** (Twelve Data spot не отдаёт volume → vol_ratio NaN → все vol-условия пропускаются, не штрафуют).
- Pivot, R1/S1, R2/S2 по prev candle.

Если `len(df) < 50` → возврат `df` с NaN-колонками.

### ICT (`indicators/ict.py`)

- `find_ote(df, lookback=20)` — zone 0.62–0.79 на последнем swing-диапазоне.
- `find_fvg(df, max_gaps=5)` — топ-5 самых свежих **незакрытых** FVG.
- `find_order_blocks(df, lookback=50)` — OB до сильного импульса (close > open * 1.003 для bullish, <0.997 для bearish), флаг `tested`.
- `find_liquidity(df, tolerance=0.002, lookback=30)` — equal highs/lows.
- `get_killzone(now=None)` — см. секцию Timezone выше.

### SMC (`indicators/smc.py`)

- `detect_structure(df, swing_len=5)` → `last_event ∈ {BOS_bull, BOS_bear, CHOCH_bull, CHOCH_bear, None}`.
- `premium_discount(df, lookback=50)` → zone, pct, equilibrium, range.

- **`detect_stop_hunt(df, atr)`** — обновлённая формула:

  ```
  wick = upper_wick или lower_wick конкретной свечи
  body = abs(close - open)
  свеча считается stop hunt, если:
      wick > 2 * body
      AND wick > 0.5 * atr
  ```

  Дополнительно: нижний хвост должен пробить недавний swing low и закрыться выше (bull hunt), либо верхний хвост пробить swing high и закрыться ниже (bear hunt). Проверяется на последних 3 свечах.

### Wyckoff (`indicators/wyckoff.py`) — soft bias

**Wyckoff в v1 — это вспомогательный bias, не первичный драйвер сигнала.**

`detect_wyckoff(df) -> dict`:
- На 60 барах H4/D1 определяет phase ∈ {accumulation, distribution, markup, markdown, neutral}.
- Возвращает `{phase, bias, strength}`.
- В скоринге даёт максимум **±5** (см. Layer 1 ниже). Бонус за strength > 70 убран.

Причина: автоматический Wyckoff на 60 барах + tick-volume — это эвристика низкой надёжности, особенно для XAU/USD spot где volume часто отсутствует. До бэктестинга нельзя давать ему большой вес.

### SR Levels (`indicators/sr_levels.py`)

Хелперы для swing-уровней, ближайшего S/R, pivot-уровней, D1-целей.

## Signal Engine

`MasterSignalEngine.analyze(data: dict[str, pd.DataFrame]) -> dict` строит независимо `bull_score` и `bear_score`. Penalty всегда вычитается **из счёта рассматриваемого направления**, не агрегируется отдельно.

### Layer 1: Macro bias (D1/W1) — max ±33

- ±20: D1 EMA50 vs EMA200.
- ±8: W1 EMA50 наклон.
- ±5: Wyckoff D1 bias (soft, см. выше).

### Layer 2: Structure (H4 SMC) — max +45

- +15: CHOCH в направлении.
- +10: BOS подтверждает.
- +8: Premium/Discount aligned.
- +7: Untested H4 OB касается цены.
- +5: Цена входит в unfilled H4 FVG.

### Layer 3: ICT precision (H1/M15) — max +45

- +12: OTE zone.
- +10: London/NY AM killzone.
- +6: Asian/NY PM killzone.
- +9: Liquidity sweep (новый stop_hunt).
- +8: H1 FVG midpoint вход.
- +6: H1 OB первый тест.

### Layer 4: Classic confirmation (H1) — max +37

- +8: RSI OS/OB.
- +6: MACD cross.
- +6: Stoch cross в OS/OB.
- +5: BB rejection.
- +5: vol_ratio > 1.5 (skip если NaN).
- +4: M15 EMA8/EMA21 cross.
- +3: Pivot S1/R1 confluence.

### Layer 5: Penalties (вычитаются из соответствующего направления)

- −20: D1 trend против направления.
- −12: Вне всех киллзон.
- −10: Premium/Discount wrong.
- −8: RSI нейтральный (40–60).
- −6: vol_ratio < 0.6 (skip если NaN).
- −5: Не в OTE.

### Decision

```
net_bull = bull_score - bull_penalties
net_bear = bear_score - bear_penalties
direction = "BUY" if net_bull > net_bear else "SELL"
final_score = max(net_bull, net_bear)
```

Tiered thresholds:

- ≥ 65 → STRONG.
- 50–64 → NORMAL.
- 40–49 → WEAK.
- < 40 → NO_SIGNAL.

**Confidence убран.** В Telegram-сообщениях выводится `Score: X/100`. После прогона `backtest.py` и измерения win-rate per tier можно вернуть calibrated confidence в v1.1.

## Level Calculation

Entry = текущий M15 close.

**SELL:**
- SL = ближайший H1 bearish OB.high или bearish FVG.top (что ближе) + `ATR_M15 * 0.3`.
- TP1 = ближайший bearish target снизу (preferring FVG midpoint).
- TP2 = ближайший sell-side liquidity pool снизу.
- TP3 = D1 support / Wyckoff accumulation zone.

**BUY:** зеркально.

**RR rule:**
- TP2 должен давать RR ≥ 1.8.
- Если TP2 fail → **сигнал отправляется, но в форматтер передаётся флаг `tp2_unavailable=True`**, в сообщении показывается `TP2: недоступен (RR < 1.8)`. Сигнал не отклоняется.
- TP1 обязателен. Если TP1 не находится — сигнал отклоняется.

## Filters & Frequency

### Dedup и ATR-reprice

Правило приоритета: **ATR-reprice побеждает dedup** (early-exit).

```
last = state.last_signal(direction)
if last is None:
    allow
elif abs(current_entry - last.entry) >= 1.5 * atr_h1:
    allow                         # ATR-reprice unlocked
elif now - last.ts >= 2 hours:
    allow                         # dedup window passed
else:
    block
```

### Rate limits (поверх dedup)

- **Не более 6 сигналов в сутки суммарно** (UTC day boundary).
- **Не более 1 WEAK-сигнала за 4 часа.**
- STRONG-сигналы всегда eligible (если dedup/RR прошли) и считаются в дневной cap.
- При превышении лимитов → лог `signals.log` с пометкой `skipped: rate_limit`, в Telegram не отправляется.

### Сканы

- 5 минут в активной киллзоне.
- 15 минут вне.
- WEAK сигналы только в киллзонах.
- STRONG — всегда eligible (dedup/RR/rate-limit могут блокировать).

## Backtest (`backtest.py`) — обязательно в v1

Цель: дать пользователю объективные метрики **до** включения live-сигналов.

CLI:

```bash
python backtest.py --tf-base H1 --months 12 --tier STRONG
python backtest.py --all-tiers --export csv
```

Что делает:

1. Тянет 12 месяцев истории через Twelve Data (по chunks с учётом rate-limit).
2. Прогоняет `MasterSignalEngine.analyze()` walk-forward на каждой M15-свече (на минутном движке — слишком много запросов, на M15 достаточно).
3. Для каждого сгенерированного сигнала отслеживает исход: hit TP1 / hit SL / hit TP2 / timeout (48h).
4. Считает per tier:
   - Win rate (TP1 hit before SL).
   - Expectancy (avg R per trade).
   - Profit factor.
   - Max consecutive losses.
   - Sharpe (на R-multiple ряду).
   - Распределение score → outcome (для калибровки порогов).
5. Выводит таблицу + сохраняет `backtest_results.csv`.

Acceptance criteria для v1 перед live: STRONG должен показать **win rate ≥ 45% при avg RR ≥ 1.8 на TP2**, иначе пороги/веса перекалибровать вручную и повторить.

## Telegram Bot

`python-telegram-bot[job-queue]==21.6` (Application builder, async).

Scheduler: **`AsyncIOScheduler`** из `apscheduler.schedulers.asyncio` — критично для async-совместимости с PTB v21. Все job'ы — async-функции, шарят event loop с ботом.

Команды:

- `/start` — welcome + список.
- `/signal` — force analysis сейчас, bypass dedup и rate-limit (но не RR-валидацию).
- `/status` — обзор: price, trend по TF, killzone, Wyckoff phase, ближайшие OB/FVG.
- `/levels` — текстовая карта ICT/SMC уровней.
- `/help` — пояснение тиров, как читать ICT/SMC.
- `/settings` — текущие пороги и интервалы.
- `/stats` — сводка из `state.db`: signals today, win-rate последних N (если есть outcome tracking — v1.1).

Scheduled jobs:

- Каждые 5 мин в активной киллзоне.
- Каждые 15 мин вне.
- Сессионные алерты по NY-времени с учётом DST:
  - London KZ open (02:00 NY).
  - NY AM KZ open (08:30 NY).
  - Asian session prep (19:00 NY).

Стартует `python bot.py` на Railway.

## Formatting

STRONG/NORMAL: полный формат (header, entry, SL, TP1/2/3, RR, Score, session, analysis breakdown, cancel condition, UTC ts).

`Уверенность: X%` → **`Score: X/100`** во всех шаблонах.

WEAK: короткий формат (entry, SL, TP1, TP2, Score).

NO_SIGNAL в киллзоне: brief market update.

Если `tp2_unavailable=True`:
```
🎯 Цели:
 •  TP1: `3,298.00` (14.5 pts) — FVG закрытие
 •  TP2: недоступен (RR < 1.8)
 •  TP3: `3,261.00` (51.5 pts) — D1 поддержка
```

Цены — `{:,.2f}`, разделитель тысяч запятая, как в исходном промте.

## Error Handling

- Twelve Data 4xx/5xx → retry → лог в `errors.log` → scheduler продолжает.
- Indicator modules возвращают neutral при коротких/NaN-данных.
- Telegram send failures → лог + retry (1 раз через 10s).
- `/status`, `/levels` — partial response при сбое отдельного модуля.
- Все исключения в scheduled jobs обёрнуты `try/except` с traceback в `errors.log`.

Скалярный доступ к ценам: `float(df.iloc[-1]["Close"])` или `.item()`.

## Logging

`signals.log`:
```
ts_utc | direction | tier | score | entry | sl | tp1 | tp2 | rr | killzone | status
```
`status ∈ {sent, skipped:dedup, skipped:rate_limit, skipped:rr}`.

`errors.log` — полные traceback'и.

Console-логи остаются для Railway observability.

## Configuration

`config.py`:

```python
# Tiers
STRONG_SIGNAL = 65
NORMAL_SIGNAL = 50
WEAK_SIGNAL = 40

# Risk
MIN_RR = 1.8

# Dedup & reprice
DEDUP_HOURS = 2
REPRICE_ATR_MULT = 1.5

# Rate limits
MAX_SIGNALS_PER_DAY = 6
WEAK_COOLDOWN_HOURS = 4

# Scan intervals (seconds)
KILLZONE_SCAN_INTERVAL = 300
BACKGROUND_SCAN_INTERVAL = 900

# ICT/SMC
OTE_LOW = 0.62
OTE_HIGH = 0.79
FVG_LOOKBACK = 30
OB_LOOKBACK = 50
LIQUIDITY_TOL = 0.002
SWING_LOOKBACK = 15
WYCKOFF_BARS = 60

# Killzones — NY time, DST-aware
TIMEZONE = "America/New_York"
```

ENV:

- `TELEGRAM_BOT_TOKEN` (required).
- `TELEGRAM_CHAT_ID` (required).
- `TWELVE_DATA_API_KEY` (required).
- `STATE_DB_PATH` (default `./state.db`).

## Deployment

Railway, Python 3.11.

- `runtime.txt`: `python-3.11.x`.
- `Procfile`: `worker: python bot.py`.
- `.env.example`: все три обязательные переменные.
- `requirements.txt`:

```
python-telegram-bot[job-queue]==21.6
yfinance==0.2.43          # удалить, не используется
twelvedata==1.2.18
pandas==2.2.2
numpy<2.0
pandas-ta==0.3.14b
apscheduler==3.10.4
python-dotenv==1.0.1
httpx==0.27.0
```

(Перед коммитом — убрать `yfinance` если он действительно не нужен; оставлен здесь как явная пометка отказа.)

README:

- Локальный setup.
- Получение Twelve Data API key.
- Telegram bot token + chat id.
- Railway deploy.
- Команды.
- Тиры и интерпретация ICT/SMC/Wyckoff.
- Раздел "Calibration": как прогнать `backtest.py` и подстроить веса.

## Testing & Validation

- Unit-тесты индикаторов на edge-cases: короткий df, flat market, no volume, no swings, no FVG.
- Engine tests: tier thresholds, penalty arithmetic (penalty вычитается из своего направления), direction selection, RR rejection of TP1, TP2-unavailable flow.
- Filters: dedup, ATR-reprice early-exit, rate-limit (day cap, WEAK cooldown), /signal bypass.
- Formatter: STRONG/WEAK/NO_SIGNAL/TP2-unavailable шаблоны.
- Smoke: каждый `indicators/*` имеет `__main__` с офлайн-данными.
- **Acceptance gate перед live:** `backtest.py` показывает STRONG win-rate ≥ 45% и expectancy > 0 на 12-месячном окне. Без этого live-режим помечается как "uncalibrated" в `/start` и `/help`.

## Multi-Stream Architecture (Revision 3)

Один скан движка теперь может породить **до трёх независимых сигналов** — по одному на каждый stream. Стримы не складывают очки в общий score; каждый имеет собственный детектор, фильтры, формат, лимиты.

### Streams

| Stream     | TF base   | Horizon       | Daily cap | Cooldown                    | Тираж в сообщении        |
|------------|-----------|---------------|-----------|-----------------------------|--------------------------|
| `intraday` | H1+M15    | 1–24h         | 6         | dedup 2h, WEAK 4h           | "Сигнал (intraday)"      |
| `swing`    | D1+H4     | 2 дня–4 нед.  | 2         | 1 в направлении / 24h       | "Свинг 500" / "Свинг 1000" |
| `scalp`    | M15       | 15–60 мин     | 4         | 30 мин между, killzone-only | "Скальп M15"             |

### StreamRouter

```python
class StreamRouter:
    def __init__(self):
        self.analyzers = {
            "intraday": IntradayAnalyzer(),  # текущий MasterSignalEngine
            "swing":    SwingAnalyzer(),
            "scalp":    ScalpAnalyzer(),
        }

    def analyze(self, data: dict[str, pd.DataFrame]) -> list[SignalResult]:
        results = []
        for stream, analyzer in self.analyzers.items():
            try:
                sig = analyzer.analyze(data)
                if sig is not None and sig["tier"] != "NO_SIGNAL":
                    sig["stream"] = stream
                    results.append(sig)
            except Exception:
                log.exception("Stream %s failed", stream)
        return results
```

Каждый сигнал проходит свой `should_send(stream=...)` и отправляется отдельным Telegram-сообщением.

### Pip Definition

Для XAU/USD во всём проекте: **1 pip = 0.10 USD**. То есть `pips(price_diff) = price_diff / 0.10`. Зафиксировано в `config.XAU_PIP_VALUE = 0.10`.

### Swing Stream (`indicators/swing.py`)

`find_swing_setup(d1_df, h4_df) -> dict | None`:

- Окно 200 D1-баров. Находит major swing high `SH` и major swing low `SL` (через `swing_highs_lows`, выбирается экстремум).
- `range_pips = (SH - SL) / XAU_PIP_VALUE`.
- D1-trend: EMA50 vs EMA200 (`bull`/`bear`).
- Setup type:
  - `range_pips ≥ 1000` → `1000pip`, entry fib 0.20 от противоположного экстремума, **TP = противоположный swing extreme** (не +1000 от entry), SL за extremum + 50 pip buffer.
  - `500 ≤ range_pips < 1000` → `500pip`, entry fib 0.236, TP = противоположный swing extreme, SL + 30 pip buffer.
- Direction берётся из D1-trend (нельзя выдать swing-сигнал против D1-bias).
- Минимальный RR (по правилу): risk vs reward от entry до TP должно быть ≥ 2.0 для swing (стандарт выше, чем у intraday).

Возвращает:
```python
{
    "type": "1000pip" | "500pip",
    "direction": "BUY" | "SELL",
    "entry": float, "sl": float, "tp": float,
    "range_pips": float, "rr": float,
}
```
или `None`.

`SwingAnalyzer` обёртывает это в стандартный `SignalResult`:
- `tier`: `STRONG` для 1000pip, `NORMAL` для 500pip.
- `tp1 = tp`, `tp2 = None`, `tp3 = None` (свинг по дизайну one-target).
- `score` = условный 80 для 1000pip, 65 для 500pip (фиксированный, не аддитивный).
- `horizon_label` = `"2-7 дней"` (500) / `"1-4 недели"` (1000).

### Scalp Stream (`indicators/scalping.py`)

`scalp_signal(m15_df, h1_df, h4_df) -> dict | None`:

- Активен только если `get_killzone() in {"London KZ", "NY AM KZ"}` (priority kz only).
- 4 условия:
  1. EMA8/EMA21 cross на M15 в последних 2 барах.
  2. RSI M15: <35 для BUY / >65 для SELL.
  3. Цена касается/отскакивает от BB-edge на M15 (close ≤ BBL для BUY, ≥ BBU для SELL).
  4. `vol_ratio_m15 > 1.3` (skip если NaN).
- Минимум 3 из 4 → setup active.
- Direction определяется условиями 1+2 (должны указывать в одну сторону).
- **H4 trend filter**: если scalp-сигнал против H4 EMA50/EMA200 trend → `tier = WEAK` (не отменяем, понижаем; counter-trend scalp возможен, но рисковее). Если по тренду — `tier = NORMAL`. STRONG для scalp не выдаём (один scalp ≠ high-conviction).
- Уровни:
  - `SL = entry ± ATR_M15 * 1.0` (≈ 15–25 pip обычно).
  - `TP1 = entry ± ATR_M15 * 1.5`.
  - `TP2 = entry ± ATR_M15 * 2.5`.
  - `TP3 = None`.

Возвращает `SignalResult` с `horizon_label = "15-60 минут"`.

### S/R Zones (`indicators/sr_zones.py`)

Этот модуль **не отдельный stream**, он обогащает context для всех трёх анализаторов.

`find_sr_zones(h4_df, d1_df, current_price) -> dict`:

1. **Historical key levels (D1)**:
   - Сканируется 365 D1 баров, найдены swing highs/lows.
   - Уровень считается key, если в радиусе 0.3% касались ≥2 раза.
   - `MAJOR` = 3+ касаний, `MINOR` = 2.
2. **Psychological round levels**: каждые $50 (3000, 3050, 3100, ...) и каждые $100 (3000, 3100, ...). Берутся ближайшие 3 выше и 3 ниже.
3. **Zone width**: `level ± ATR_H4 * 0.5`. Цена `inside zone` если попадает в этот диапазон.
4. **Zone strength (0–100, clamped)**:
   ```
   raw = min(touches, 5) * 15  +  recency_bonus(0..10)  +  tf_bonus(D1=20, H4=10, psych=8)
   strength = min(100, raw)
   ```

Возвращает:
```python
{
    "resistance_zones": [...],  # каждый: level, strength, zone_top, zone_bot, type, touches
    "support_zones":    [...],
    "at_resistance": bool, "at_support": bool,
    "nearest_resistance": float | None,
    "nearest_support":    float | None,
}
```

**Интеграция с intraday-движком (IntradayAnalyzer)** — добавляются condition checks в Layer 2 (Structure):
- +12 если цена внутри MAJOR support zone и direction = BUY (или MAJOR resistance + SELL).
- +8 для MINOR zone.
- +6 за psychological round level.
- +5 если zone strength > 70.
- −8 если сигнал направлен в strong opposing zone (strength > 70) ближе 30 pip.

**Anti-double-count rule**: если zone level совпадает с liquidity pool (отличие < 0.1%), бонус за zone применяется, а liquidity-bonus **не суммируется** для этого уровня (берётся макс из двух).

### Score Normalization (Revision 3)

Текущая шкала остаётся (макс ~150 raw), но после S/R-zones bonuses сместится. Пороги STRONG/NORMAL/WEAK **остаются 65/50/40** для intraday — после калибровки бэктестом можно поднять до 75/60/45 если STRONG слишком частый. Swing и scalp используют **фиксированные** tier-присваивания (см. выше), не зависят от порогов intraday.

### Backtest per stream

`backtest.py` принимает флаг `--stream {intraday,swing,scalp,all}`:
- intraday: `timeout_bars=48` (48h на H1).
- swing: `timeout_bars=336` (~2 недели на H1).
- scalp: `timeout_bars=8` (2 часа на M15-эквивалент = 8 H1 шагов = используем M15-данные напрямую).

Acceptance gates перед live:
- intraday STRONG: win rate ≥ 45%, expectancy > 0.
- swing: win rate ≥ 40% (выборка маленькая, поэтому терпимее), expectancy ≥ 0.5R.
- scalp: win rate ≥ 50% (короткий TP), expectancy > 0.

### Formatter updates

Все сигналы получают новые строки после TP-блока:

```
📐 Стратегия: <stream label>
⏳ Горизонт: <horizon>
```

Где `stream label` для intraday это композит из `reasons` ("SMC+ICT" если оба сработали), для swing — "Swing 500" / "Swing 1000", для scalp — "Scalp M15".

### State schema (Revision 3)

Таблица `signals` получает колонку `stream TEXT NOT NULL DEFAULT 'intraday'`. Все методы `State` принимают опциональный `stream` параметр.

```sql
ALTER TABLE signals ADD COLUMN stream TEXT NOT NULL DEFAULT 'intraday';
CREATE INDEX IF NOT EXISTS idx_signals_stream ON signals(stream);
```

Миграция автоматическая при старте `State.__init__` через `PRAGMA user_version` или try/except на `ALTER`.

### Rate limits & dedup per stream

`should_send(sig, state)` читает `sig["stream"]` и применяет соответствующий лимит:
- intraday → `MAX_INTRADAY_PER_DAY=6`, dedup 2h, WEAK cooldown 4h.
- swing → `MAX_SWING_PER_DAY=2`, плюс правило "один в одном направлении / 24h".
- scalp → `MAX_SCALP_PER_DAY=4`, плюс min 30 мин между scalp-сигналами, kz-only enforce.

---

## Open Decisions Resolved

- Source: **Twelve Data** (XAU/USD spot, REST, free tier).
- Persistence: **SQLite `state.db`** для dedup и истории.
- Killzones: **America/New_York** через `zoneinfo`, DST-aware.
- pandas-ta: **monkey-patch `numpy.NaN = numpy.nan`** в `indicators/__init__.py`.
- Scheduler: **AsyncIOScheduler** для PTB v21.
- Wyckoff: **soft bias ±5**, не первичный драйвер.
- Stop hunt: `wick > 2*body AND wick > 0.5*atr`.
- Dedup vs reprice: **ATR-reprice wins**.
- Rate limits: **6/день, WEAK 1/4ч**.
- "Уверенность %" → **"Score X/100"** до калибровки.
- TP2 missing: **сигнал идёт, TP2 помечается недоступным**.
- Backtest: **обязательная часть v1**, не v2.
- Multi-stream: **3 независимых потока** (intraday/swing/scalp), не объединяются в один score.
- Pip definition: **1 pip = 0.10 USD** (`config.XAU_PIP_VALUE`).
- S/R zones: **обогащают context**, не отдельный stream; anti-double-count с liquidity pools.
- Score normalization: пороги intraday остаются 65/50/40 в v1; swing/scalp используют фиксированные tier (1000pip=STRONG, 500pip=NORMAL, scalp=NORMAL/WEAK).
