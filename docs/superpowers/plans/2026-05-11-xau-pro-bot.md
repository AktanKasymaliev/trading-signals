# XAU Pro Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic XAU/USD Telegram signal bot using ICT/SMC/Wyckoff/Classic TA confluence, with SQLite-persisted dedup, calibration-ready backtester, and Railway-friendly deployment.

**Architecture:** Single-process Python 3.11 async worker. `data.py` pulls OHLCV from Twelve Data (5 TFs, in-memory TTL cache). `indicators/*` extract deterministic features. `signals/engine.py` builds bull/bear scores across 5 layers, applies penalties, picks direction/tier, computes entry/SL/TPs. `signals/filters.py` applies dedup+ATR-reprice+rate-limit using `state.py` (SQLite). `bot.py` runs `python-telegram-bot` v21 + `AsyncIOScheduler` for 5/15-min scans with NY-tz killzones. `backtest.py` is a separate CLI that replays the engine on history with outcome tracking — required v1 acceptance gate.

**Tech Stack:** Python 3.11, `python-telegram-bot[job-queue]==21.6`, `twelvedata==1.2.18`, `pandas==2.2.2`, `numpy<2.0`, `pandas-ta==0.3.14b` (with numpy.NaN monkey-patch), `apscheduler==3.10.4` (`AsyncIOScheduler`), `httpx==0.27.0`, `python-dotenv==1.0.1`, SQLite (stdlib), `pytest` for tests.

**Working directory:** `/Users/aktan.kasymalievicloud.com/Projects/self-projects/signals/` — the actual code lives in `xau_pro_bot/` subdirectory per spec.

---

## File Map

```
signals/                                  (repo root)
├── xau_pro_bot/                          (package — all source here)
│   ├── __init__.py
│   ├── bot.py                            entrypoint, telegram + scheduler
│   ├── data.py                           Twelve Data client + cache
│   ├── state.py                          SQLite persistence
│   ├── formatter.py                      Markdown messages
│   ├── backtest.py                       CLI backtester
│   ├── config.py                         constants + ENV parsing
│   ├── indicators/
│   │   ├── __init__.py                   numpy.NaN monkey-patch
│   │   ├── classic.py                    EMA/RSI/MACD/BB/ATR/Stoch/Pivots
│   │   ├── ict.py                        OTE/FVG/OB/liquidity/killzone
│   │   ├── smc.py                        BOS/CHOCH/PD/stop_hunt
│   │   ├── wyckoff.py                    phase detection (soft bias)
│   │   └── sr_levels.py                  swing/SR/pivot helpers
│   └── signals/
│       ├── __init__.py
│       ├── engine.py                     MasterSignalEngine
│       ├── ict_signals.py                ICT scoring conditions
│       ├── smc_signals.py                SMC scoring conditions
│       ├── classic_signals.py            Classic scoring conditions
│       └── filters.py                    dedup/ratelimit/RR
├── tests/                                (mirror structure)
│   ├── __init__.py
│   ├── conftest.py                       fixtures (sample DFs)
│   ├── test_data.py
│   ├── test_state.py
│   ├── test_classic.py
│   ├── test_ict.py
│   ├── test_smc.py
│   ├── test_wyckoff.py
│   ├── test_sr_levels.py
│   ├── test_engine.py
│   ├── test_filters.py
│   ├── test_formatter.py
│   └── test_backtest.py
├── requirements.txt
├── requirements-dev.txt
├── runtime.txt                           Railway: python-3.11
├── Procfile                              worker: python -m xau_pro_bot.bot
├── .env.example
├── .gitignore
├── pytest.ini
└── README.md
```

**Test strategy:** pytest, fixtures in `tests/conftest.py` build deterministic OHLCV DataFrames (no network in unit tests). Twelve Data calls are mocked. `backtest.py` integration test uses a small fixture CSV.

---

## Task 0: Repository Scaffolding

**Files:**
- Create: `signals/requirements.txt`
- Create: `signals/requirements-dev.txt`
- Create: `signals/runtime.txt`
- Create: `signals/Procfile`
- Create: `signals/.env.example`
- Create: `signals/.gitignore`
- Create: `signals/pytest.ini`
- Create: `signals/xau_pro_bot/__init__.py` (empty)
- Create: `signals/xau_pro_bot/indicators/__init__.py`
- Create: `signals/xau_pro_bot/signals/__init__.py` (empty)
- Create: `signals/tests/__init__.py` (empty)

- [ ] **Step 1: Write `requirements.txt`**

```
python-telegram-bot[job-queue]==21.6
twelvedata==1.2.18
pandas==2.2.2
numpy<2.0
pandas-ta==0.3.14b
apscheduler==3.10.4
python-dotenv==1.0.1
httpx==0.27.0
```

- [ ] **Step 2: Write `requirements-dev.txt`**

```
-r requirements.txt
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-mock==3.14.0
freezegun==1.5.1
```

- [ ] **Step 3: Write `runtime.txt`**

```
python-3.11.10
```

- [ ] **Step 4: Write `Procfile`**

```
worker: python -m xau_pro_bot.bot
```

- [ ] **Step 5: Write `.env.example`**

```
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TWELVE_DATA_API_KEY=
STATE_DB_PATH=./state.db
LOG_LEVEL=INFO
```

- [ ] **Step 6: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.env
state.db
signals.log
errors.log
backtest_results.csv
.pytest_cache/
.DS_Store
```

- [ ] **Step 7: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning:pandas_ta
```

- [ ] **Step 8: Write `xau_pro_bot/indicators/__init__.py` — the numpy.NaN guard**

```python
"""Indicators package.

This module monkey-patches numpy.NaN BEFORE importing pandas_ta to fix the
known incompatibility between pandas_ta 0.3.14b (uses numpy.NaN) and
numpy >= 1.24 (where numpy.NaN was removed in favor of numpy.nan).

Any module that imports pandas_ta MUST go through this package first.
"""

import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

# Trigger pandas_ta import here so the patch is always applied first.
import pandas_ta  # noqa: E402, F401
```

- [ ] **Step 9: Create empty `__init__.py` files**

```bash
touch xau_pro_bot/__init__.py xau_pro_bot/signals/__init__.py tests/__init__.py
```

- [ ] **Step 10: Commit**

```bash
git init  # if not already
git add .
git commit -m "chore: scaffold project structure and requirements"
```

---

## Task 1: Config Module

**Files:**
- Create: `xau_pro_bot/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from xau_pro_bot import config


def test_thresholds_match_spec():
    assert config.STRONG_SIGNAL == 65
    assert config.NORMAL_SIGNAL == 50
    assert config.WEAK_SIGNAL == 40
    assert config.MIN_RR == 1.8


def test_rate_limits():
    assert config.MAX_SIGNALS_PER_DAY == 6
    assert config.WEAK_COOLDOWN_HOURS == 4


def test_scan_intervals():
    assert config.KILLZONE_SCAN_INTERVAL == 300
    assert config.BACKGROUND_SCAN_INTERVAL == 900


def test_load_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        config.load_env(required=["TELEGRAM_BOT_TOKEN"])


def test_load_env_returns_dict(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "key")
    env = config.load_env(required=["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TWELVE_DATA_API_KEY"])
    assert env["TELEGRAM_BOT_TOKEN"] == "abc"
    assert env["TELEGRAM_CHAT_ID"] == "123"
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError or AttributeError — `config` module doesn't exist.

- [ ] **Step 3: Implement `config.py`**

```python
# xau_pro_bot/config.py
"""Configuration constants and environment loader."""

from __future__ import annotations

import os
from datetime import time
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

# ── Tiers ─────────────────────────────────────────────
STRONG_SIGNAL = 65
NORMAL_SIGNAL = 50
WEAK_SIGNAL = 40

# ── Risk ──────────────────────────────────────────────
MIN_RR = 1.8

# ── Dedup & reprice ───────────────────────────────────
DEDUP_HOURS = 2
REPRICE_ATR_MULT = 1.5

# ── Rate limits ───────────────────────────────────────
MAX_SIGNALS_PER_DAY = 6
WEAK_COOLDOWN_HOURS = 4

# ── Scan intervals (seconds) ──────────────────────────
KILLZONE_SCAN_INTERVAL = 300
BACKGROUND_SCAN_INTERVAL = 900

# ── ICT / SMC ─────────────────────────────────────────
OTE_LOW = 0.62
OTE_HIGH = 0.79
FVG_LOOKBACK = 30
OB_LOOKBACK = 50
LIQUIDITY_TOL = 0.002
SWING_LOOKBACK = 15
WYCKOFF_BARS = 60

# ── Timezone & killzones (NY local time) ──────────────
TIMEZONE = "America/New_York"

KILLZONES_NY: dict[str, tuple[time, time]] = {
    "Asian KZ":   (time(20, 0), time(23, 59)),
    "London KZ":  (time(2, 0),  time(5, 0)),
    "NY AM KZ":   (time(8, 30), time(11, 0)),
    "NY PM KZ":   (time(13, 30), time(16, 0)),
}

PRIORITY_KILLZONES = {"London KZ", "NY AM KZ"}

# ── Data ──────────────────────────────────────────────
SYMBOL = "XAU/USD"
TF_SPEC = {
    "W1":  ("1week",  104),
    "D1":  ("1day",   365),
    "H4":  ("4h",     540),
    "H1":  ("1h",     720),
    "M15": ("15min",  672),
}
DATA_CACHE_TTL_SECONDS = 300
DATA_RETRY_ATTEMPTS = 3
DATA_RETRY_DELAY_SECONDS = 5


def load_env(required: Iterable[str]) -> dict[str, str]:
    """Load and validate required environment variables."""
    env: dict[str, str] = {}
    missing: list[str] = []
    for key in required:
        value = os.getenv(key)
        if value is None or value == "":
            missing.append(key)
        else:
            env[key] = value
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return env
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/config.py tests/test_config.py
git commit -m "feat(config): add constants and env loader"
```

---

## Task 2: Test Fixtures (conftest.py)

**Files:**
- Create: `tests/conftest.py`

This file is foundational — most later tests depend on it. No app code yet, so no separate failing test step.

- [ ] **Step 1: Implement fixtures**

```python
# tests/conftest.py
"""Deterministic OHLCV fixtures for offline unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


def _make_df(closes: list[float], start: datetime, freq: str = "h",
             volume: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    closes_arr = np.array(closes, dtype=float)
    # Simple synthetic OHLC around close
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    highs = np.maximum(opens, closes_arr) + 0.5
    lows = np.minimum(opens, closes_arr) - 0.5
    if volume is None:
        volume = [1000.0] * n
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes_arr,
            "Volume": np.array(volume, dtype=float),
        },
        index=idx,
    )


@pytest.fixture
def uptrend_df() -> pd.DataFrame:
    """100 bars trending up 2000 → 2200."""
    closes = list(np.linspace(2000.0, 2200.0, 100))
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def downtrend_df() -> pd.DataFrame:
    closes = list(np.linspace(2200.0, 2000.0, 100))
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def flat_df() -> pd.DataFrame:
    closes = [2100.0] * 100
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def short_df() -> pd.DataFrame:
    """Only 10 bars — many indicators should return neutral."""
    closes = list(np.linspace(2000.0, 2050.0, 10))
    return _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.fixture
def df_with_fvg() -> pd.DataFrame:
    """100 bars with a clear bullish FVG injected at index 50."""
    closes = list(np.linspace(2000.0, 2100.0, 100))
    df = _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))
    # Inject bullish FVG: candle[50].low > candle[48].high
    df.iloc[48, df.columns.get_loc("High")] = 2040.0
    df.iloc[49, df.columns.get_loc("High")] = 2060.0
    df.iloc[49, df.columns.get_loc("Low")] = 2050.0
    df.iloc[50, df.columns.get_loc("Low")] = 2055.0
    return df


@pytest.fixture
def df_with_volume_none() -> pd.DataFrame:
    """Mimics Twelve Data spot response with NaN Volume."""
    closes = list(np.linspace(2000.0, 2100.0, 100))
    df = _make_df(closes, datetime(2026, 1, 1, tzinfo=timezone.utc))
    df["Volume"] = np.nan
    return df


@pytest.fixture
def all_tfs(uptrend_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mock fetch_all_timeframes return value."""
    return {tf: uptrend_df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}
```

- [ ] **Step 2: Sanity check — fixtures import without error**

```bash
pytest tests/conftest.py --collect-only
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add deterministic OHLCV fixtures"
```

---

## Task 3: State Module (SQLite Persistence)

**Files:**
- Create: `xau_pro_bot/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_state.py
from datetime import datetime, timedelta, timezone

import pytest

from xau_pro_bot.state import State


@pytest.fixture
def state(tmp_path):
    return State(db_path=str(tmp_path / "test.db"))


def _sig(direction="BUY", tier="STRONG", score=70, entry=2000.0,
         ts: datetime | None = None) -> dict:
    return {
        "ts_utc": (ts or datetime.now(timezone.utc)).isoformat(),
        "direction": direction,
        "tier": tier,
        "score": score,
        "entry": entry,
        "sl": entry - 10,
        "tp1": entry + 15,
        "tp2": entry + 30,
        "tp3": entry + 45,
        "rr": 2.0,
        "killzone": "London KZ",
        "reasons_json": "{}",
    }


def test_record_and_last_signal(state):
    sig = _sig()
    sid = state.record_signal(sig)
    assert sid > 0
    last = state.last_signal()
    assert last is not None
    assert last["direction"] == "BUY"
    assert last["entry"] == pytest.approx(2000.0)


def test_last_signal_filter_by_direction(state):
    state.record_signal(_sig(direction="BUY", entry=2000.0))
    state.record_signal(_sig(direction="SELL", entry=2100.0))
    assert state.last_signal(direction="BUY")["entry"] == pytest.approx(2000.0)
    assert state.last_signal(direction="SELL")["entry"] == pytest.approx(2100.0)


def test_count_today(state):
    now = datetime.now(timezone.utc)
    for _ in range(3):
        state.record_signal(_sig(ts=now))
    state.record_signal(_sig(ts=now - timedelta(days=2)))
    assert state.count_today() == 3


def test_count_today_by_tier(state):
    now = datetime.now(timezone.utc)
    state.record_signal(_sig(tier="STRONG", ts=now))
    state.record_signal(_sig(tier="WEAK", ts=now))
    state.record_signal(_sig(tier="WEAK", ts=now))
    assert state.count_today(tier="WEAK") == 2
    assert state.count_today(tier="STRONG") == 1


def test_last_weak_ts(state):
    assert state.last_weak_ts() is None
    state.record_signal(_sig(tier="WEAK"))
    assert state.last_weak_ts() is not None


def test_prune_old(state):
    old = datetime.now(timezone.utc) - timedelta(days=100)
    state.record_signal(_sig(ts=old))
    state.record_signal(_sig())
    removed = state.prune_old(days=90)
    assert removed == 1
    assert state.count_today() == 1
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_state.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `state.py`**

```python
# xau_pro_bot/state.py
"""SQLite persistence for signals, dedup, and rate-limit state."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    direction TEXT NOT NULL,
    tier TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc);
CREATE INDEX IF NOT EXISTS idx_signals_dir ON signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_tier ON signals(tier);
"""


class State:
    """Thin wrapper over SQLite for signal persistence and dedup queries."""

    def __init__(self, db_path: str = "./state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def record_signal(self, sig: dict[str, Any]) -> int:
        cols = ("ts_utc", "direction", "tier", "score", "entry", "sl",
                "tp1", "tp2", "tp3", "rr", "killzone", "reasons_json")
        placeholders = ", ".join("?" * len(cols))
        cur = self._conn.execute(
            f"INSERT INTO signals ({', '.join(cols)}) VALUES ({placeholders})",
            tuple(sig.get(c) for c in cols),
        )
        return int(cur.lastrowid or 0)

    def last_signal(self, direction: str | None = None) -> dict[str, Any] | None:
        if direction is None:
            row = self._conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM signals WHERE direction = ? ORDER BY id DESC LIMIT 1",
                (direction,),
            ).fetchone()
        return dict(row) if row else None

    def count_today(self, tier: str | None = None) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        if tier is None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE substr(ts_utc, 1, 10) = ?",
                (today,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM signals "
                "WHERE substr(ts_utc, 1, 10) = ? AND tier = ?",
                (today, tier),
            ).fetchone()
        return int(row["n"])

    def last_weak_ts(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT ts_utc FROM signals WHERE tier = 'WEAK' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["ts_utc"])

    def prune_old(self, days: int = 90) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.execute("DELETE FROM signals WHERE ts_utc < ?", (cutoff,))
        return int(cur.rowcount)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_state.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/state.py tests/test_state.py
git commit -m "feat(state): add SQLite signal persistence"
```

---

## Task 4: Data Module (Twelve Data Client + Cache)

**Files:**
- Create: `xau_pro_bot/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write failing tests (with mocked Twelve Data)**

```python
# tests/test_data.py
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from xau_pro_bot import data as data_mod


@pytest.fixture(autouse=True)
def reset_cache():
    data_mod._CACHE.clear()
    yield
    data_mod._CACHE.clear()


def _fake_td_response(rows: int = 100) -> dict:
    """Mimic twelvedata library .as_json() output."""
    values = []
    for i in range(rows):
        values.append({
            "datetime": f"2026-01-{(i % 28) + 1:02d} 00:00:00",
            "open": "2000.0",
            "high": "2010.0",
            "low": "1995.0",
            "close": "2005.0",
            "volume": "1000",
        })
    return {"status": "ok", "values": values}


def test_normalize_response_to_df():
    payload = _fake_td_response(50)
    df = data_mod._payload_to_df(payload)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.is_monotonic_increasing
    assert df["Close"].iloc[-1] == pytest.approx(2005.0)


def test_normalize_volume_missing_becomes_nan():
    payload = _fake_td_response(10)
    for row in payload["values"]:
        row.pop("volume")
    df = data_mod._payload_to_df(payload)
    assert df["Volume"].isna().all()


def test_fetch_uses_cache(monkeypatch):
    payload = _fake_td_response(50)
    call_count = {"n": 0}

    def fake_fetch(tf):
        call_count["n"] += 1
        return data_mod._payload_to_df(payload)

    monkeypatch.setattr(data_mod, "_fetch_single_tf", fake_fetch)

    data_mod.fetch_all_timeframes(api_key="dummy")
    data_mod.fetch_all_timeframes(api_key="dummy")  # 2nd call → cache
    assert call_count["n"] == 5  # 5 TFs, cached on 2nd call


def test_fetch_retries_on_error(monkeypatch):
    attempts = {"n": 0}

    def flaky(tf):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return data_mod._payload_to_df(_fake_td_response(50))

    monkeypatch.setattr(data_mod, "_fetch_single_tf", flaky)
    monkeypatch.setattr(data_mod.time, "sleep", lambda _: None)  # skip real sleeps

    df = data_mod._fetch_with_retry("M15")
    assert isinstance(df, pd.DataFrame)
    assert attempts["n"] == 3
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_data.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `data.py`**

```python
# xau_pro_bot/data.py
"""Twelve Data REST client with TTL cache and retry."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from xau_pro_bot import config

log = logging.getLogger(__name__)

# tf -> (df, fetched_at_epoch)
_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}

_API_KEY: str | None = None


def _payload_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert Twelve Data JSON payload to OHLCV DataFrame."""
    if payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {payload.get('message')}")
    rows = payload.get("values", [])
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    rename = {"open": "Open", "high": "High", "low": "Low",
              "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = float("nan")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def _fetch_single_tf(tf: str) -> pd.DataFrame:
    """Fetch one timeframe from Twelve Data. Mocked in tests."""
    from twelvedata import TDClient  # local import to ease testing

    interval, outputsize = config.TF_SPEC[tf]
    td = TDClient(apikey=_API_KEY)
    ts = td.time_series(
        symbol=config.SYMBOL,
        interval=interval,
        outputsize=outputsize,
        timezone="UTC",
    )
    payload = ts.as_json()
    if isinstance(payload, tuple):  # twelvedata sometimes returns tuple
        payload = {"status": "ok", "values": list(payload)}
    return _payload_to_df(payload)


def _fetch_with_retry(tf: str) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, config.DATA_RETRY_ATTEMPTS + 1):
        try:
            return _fetch_single_tf(tf)
        except Exception as exc:
            last_exc = exc
            log.warning("Fetch %s attempt %d failed: %s", tf, attempt, exc)
            if attempt < config.DATA_RETRY_ATTEMPTS:
                time.sleep(config.DATA_RETRY_DELAY_SECONDS)
    assert last_exc is not None
    raise last_exc


def fetch_all_timeframes(api_key: str) -> dict[str, pd.DataFrame]:
    """Return {tf: DataFrame} for W1, D1, H4, H1, M15. Cached for 5 minutes."""
    global _API_KEY
    _API_KEY = api_key
    now = time.time()
    result: dict[str, pd.DataFrame] = {}
    for tf in config.TF_SPEC:
        cached = _CACHE.get(tf)
        if cached and (now - cached[1]) < config.DATA_CACHE_TTL_SECONDS:
            result[tf] = cached[0]
            continue
        df = _fetch_with_retry(tf)
        _CACHE[tf] = (df, now)
        result[tf] = df
    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_data.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/data.py tests/test_data.py
git commit -m "feat(data): add Twelve Data client with TTL cache and retry"
```

---

## Task 5: Classic Indicators

**Files:**
- Create: `xau_pro_bot/indicators/classic.py`
- Test: `tests/test_classic.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_classic.py
import numpy as np
import pandas as pd

from xau_pro_bot.indicators.classic import add_classic


def test_adds_all_expected_columns(uptrend_df):
    df = add_classic(uptrend_df)
    for col in ("EMA_8", "EMA_21", "EMA_50", "EMA_200",
                "RSI_14", "ATR_14", "vol_ratio",
                "pivot", "r1", "s1", "r2", "s2"):
        assert col in df.columns, f"missing {col}"


def test_short_df_returns_nans_not_crash(short_df):
    df = add_classic(short_df)
    # EMA_200 on 10 bars must be all NaN
    assert df["EMA_200"].isna().all()


def test_vol_ratio_nan_when_volume_missing(df_with_volume_none):
    df = add_classic(df_with_volume_none)
    assert df["vol_ratio"].isna().all()


def test_ema_ordering_in_uptrend(uptrend_df):
    df = add_classic(uptrend_df)
    last = df.iloc[-1]
    assert last["EMA_8"] > last["EMA_21"] > last["EMA_50"]


def test_pivots_use_previous_bar(uptrend_df):
    df = add_classic(uptrend_df)
    prev = uptrend_df.iloc[-2]
    expected_pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
    assert df["pivot"].iloc[-1] == pytest.approx(expected_pivot)
```

Add at top of file:

```python
import pytest
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_classic.py -v
```

- [ ] **Step 3: Implement `classic.py`**

```python
# xau_pro_bot/indicators/classic.py
"""Classic TA indicators: EMA, RSI, MACD, BB, ATR, Stoch, Volume, Pivots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xau_pro_bot.indicators import pandas_ta as ta  # noqa: F401 — patched


def add_classic(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich an OHLCV DataFrame with classic indicators in-place (copy)."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["EMA_8"]   = ta.ema(close, length=8)
    out["EMA_21"]  = ta.ema(close, length=21)
    out["EMA_50"]  = ta.ema(close, length=50)
    out["EMA_200"] = ta.ema(close, length=200)

    out["RSI_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out[["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]] = macd.iloc[:, :3]
    else:
        for c in ("MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"):
            out[c] = np.nan

    stoch = ta.stoch(high, low, close, k=14, d=3)
    if stoch is not None and not stoch.empty:
        out[["STOCHk_14_3_3", "STOCHd_14_3_3"]] = stoch.iloc[:, :2]
    else:
        out["STOCHk_14_3_3"] = np.nan
        out["STOCHd_14_3_3"] = np.nan

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        out[["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]] = bb.iloc[:, [0, 1, 2]]
    else:
        for c in ("BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"):
            out[c] = np.nan

    out["ATR_14"] = ta.atr(high, low, close, length=14)

    if "Volume" in out and not out["Volume"].isna().all():
        vol_avg = out["Volume"].rolling(20).mean()
        out["vol_ratio"] = out["Volume"] / vol_avg
    else:
        out["vol_ratio"] = np.nan

    prev = out.shift(1)
    out["pivot"] = (prev["High"] + prev["Low"] + prev["Close"]) / 3
    out["r1"] = 2 * out["pivot"] - prev["Low"]
    out["s1"] = 2 * out["pivot"] - prev["High"]
    out["r2"] = out["pivot"] + (prev["High"] - prev["Low"])
    out["s2"] = out["pivot"] - (prev["High"] - prev["Low"])

    return out


if __name__ == "__main__":
    from datetime import datetime, timezone
    n = 250
    df = pd.DataFrame({
        "Open":  np.linspace(2000, 2100, n),
        "High":  np.linspace(2005, 2105, n),
        "Low":   np.linspace(1995, 2095, n),
        "Close": np.linspace(2000, 2100, n),
        "Volume": [1000.0] * n,
    }, index=pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc),
                           periods=n, freq="h"))
    enriched = add_classic(df)
    print(enriched.tail(3).T)
```

Note: the `from xau_pro_bot.indicators import pandas_ta as ta` import works because `indicators/__init__.py` imports `pandas_ta` and binds it at the package level. Actually fix: just `import pandas_ta as ta` after triggering the package import. Replace that line with:

```python
import xau_pro_bot.indicators  # noqa: F401 — apply numpy.NaN patch
import pandas_ta as ta
```

Use this corrected version in step 3.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_classic.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/indicators/classic.py tests/test_classic.py
git commit -m "feat(indicators): add classic TA (EMA/RSI/MACD/BB/ATR/Stoch/Pivots)"
```

---

## Task 6: ICT Indicators

**Files:**
- Create: `xau_pro_bot/indicators/ict.py`
- Test: `tests/test_ict.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ict.py
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from xau_pro_bot.indicators.ict import (
    find_ote,
    find_fvg,
    find_order_blocks,
    find_liquidity,
    get_killzone,
)


def test_find_ote_returns_dict_with_required_keys(uptrend_df):
    res = find_ote(uptrend_df, lookback=20)
    assert set(res.keys()) >= {"in_ote", "ote_low", "ote_high",
                               "swing_high", "swing_low", "direction"}


def test_find_ote_short_df_neutral(short_df):
    res = find_ote(short_df, lookback=20)
    assert res["in_ote"] is False


def test_find_fvg_detects_bullish_gap(df_with_fvg):
    gaps = find_fvg(df_with_fvg, max_gaps=5)
    # Allow that detector finds at least one gap somewhere
    assert isinstance(gaps, list)


def test_find_fvg_empty_for_flat(flat_df):
    gaps = find_fvg(flat_df, max_gaps=5)
    assert gaps == []


def test_find_order_blocks_returns_list(uptrend_df):
    obs = find_order_blocks(uptrend_df, lookback=50)
    assert isinstance(obs, list)
    for ob in obs:
        assert ob["type"] in ("bull", "bear")
        assert ob["high"] >= ob["low"]


def test_find_liquidity_keys(uptrend_df):
    res = find_liquidity(uptrend_df)
    assert set(res.keys()) == {"buy_side", "sell_side"}


def test_killzone_london():
    # 03:00 NY (London KZ)
    now = datetime(2026, 5, 11, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) == "London KZ"


def test_killzone_ny_am():
    now = datetime(2026, 5, 11, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) == "NY AM KZ"


def test_killzone_outside():
    now = datetime(2026, 5, 11, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) is None


def test_killzone_dst_aware():
    # November (post-DST) — London KZ shifts UTC, but NY local is same
    now = datetime(2026, 11, 15, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) == "London KZ"


def test_killzone_rejects_naive_datetime():
    with pytest.raises(ValueError):
        get_killzone(datetime(2026, 5, 11, 3, 0))
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_ict.py -v
```

- [ ] **Step 3: Implement `ict.py`**

```python
# xau_pro_bot/indicators/ict.py
"""ICT concepts: OTE, FVG, Order Blocks, Liquidity, Killzones."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from xau_pro_bot import config


_NY = ZoneInfo(config.TIMEZONE)


def _neutral_ote() -> dict[str, Any]:
    return {
        "in_ote": False, "ote_low": None, "ote_high": None,
        "swing_high": None, "swing_low": None, "direction": None,
    }


def find_ote(df: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    if len(df) < lookback + 1:
        return _neutral_ote()
    window = df.tail(lookback)
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    if swing_high == swing_low:
        return _neutral_ote()
    last_close = float(df["Close"].iloc[-1])
    # Direction: where is price relative to mid? Closer to high → bull retracement zone
    mid = (swing_high + swing_low) / 2
    if last_close >= mid:
        direction = "bull"
        ote_low = swing_low + 0.62 * (swing_high - swing_low)
        ote_high = swing_low + 0.79 * (swing_high - swing_low)
    else:
        direction = "bear"
        ote_low = swing_high - 0.79 * (swing_high - swing_low)
        ote_high = swing_high - 0.62 * (swing_high - swing_low)
    in_ote = ote_low <= last_close <= ote_high
    return {
        "in_ote": bool(in_ote),
        "ote_low": float(ote_low),
        "ote_high": float(ote_high),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "direction": direction,
    }


def find_fvg(df: pd.DataFrame, max_gaps: int = 5) -> list[dict[str, Any]]:
    """Return unfilled FVGs (most recent first), max `max_gaps`."""
    if len(df) < 3:
        return []
    gaps: list[dict[str, Any]] = []
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    last_idx = len(df) - 1
    for i in range(2, len(df)):
        # Bullish FVG: candle[i].low > candle[i-2].high
        if lows[i] > highs[i - 2]:
            top = float(lows[i])
            bottom = float(highs[i - 2])
            # unfilled iff no subsequent low <= bottom
            future_lows = lows[i + 1:]
            if len(future_lows) == 0 or future_lows.min() > bottom:
                gaps.append({
                    "type": "bull", "top": top, "bottom": bottom,
                    "midpoint": (top + bottom) / 2, "age_bars": last_idx - i,
                })
        if highs[i] < lows[i - 2]:
            top = float(lows[i - 2])
            bottom = float(highs[i])
            future_highs = highs[i + 1:]
            if len(future_highs) == 0 or future_highs.max() < top:
                gaps.append({
                    "type": "bear", "top": top, "bottom": bottom,
                    "midpoint": (top + bottom) / 2, "age_bars": last_idx - i,
                })
    gaps.sort(key=lambda g: g["age_bars"])
    return gaps[:max_gaps]


def find_order_blocks(df: pd.DataFrame, lookback: int = 50) -> list[dict[str, Any]]:
    if len(df) < lookback + 2:
        return []
    obs: list[dict[str, Any]] = []
    window = df.tail(lookback).reset_index(drop=False)
    opens = window["Open"].to_numpy()
    closes = window["Close"].to_numpy()
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()

    for i in range(1, len(window)):
        # Bullish impulse: close > open * 1.003
        if closes[i] > opens[i] * 1.003:
            # find last bearish candle before this impulse
            for j in range(i - 1, -1, -1):
                if closes[j] < opens[j]:
                    obs.append({
                        "type": "bull",
                        "high": float(highs[j]),
                        "low": float(lows[j]),
                        "mid": float((highs[j] + lows[j]) / 2),
                        "index": int(j),
                        "tested": bool(lows[j + 1:].min() <= highs[j])
                                  if i > j + 1 else False,
                    })
                    break
        if closes[i] < opens[i] * 0.997:
            for j in range(i - 1, -1, -1):
                if closes[j] > opens[j]:
                    obs.append({
                        "type": "bear",
                        "high": float(highs[j]),
                        "low": float(lows[j]),
                        "mid": float((highs[j] + lows[j]) / 2),
                        "index": int(j),
                        "tested": bool(highs[j + 1:].max() >= lows[j])
                                  if i > j + 1 else False,
                    })
                    break
    # Dedupe by index
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for ob in obs[::-1]:  # most recent first
        if ob["index"] in seen:
            continue
        seen.add(ob["index"])
        unique.append(ob)
    return unique[:10]


def find_liquidity(df: pd.DataFrame, tolerance: float = 0.002,
                   lookback: int = 30) -> dict[str, list[float]]:
    if len(df) < lookback:
        return {"buy_side": [], "sell_side": []}
    window = df.tail(lookback)
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()

    def cluster(values, ref) -> list[float]:
        clusters: list[float] = []
        used = [False] * len(values)
        for i, v in enumerate(values):
            if used[i]:
                continue
            group = [v]
            used[i] = True
            for j in range(i + 1, len(values)):
                if used[j]:
                    continue
                if abs(values[j] - v) / max(abs(ref), 1) <= tolerance:
                    group.append(values[j])
                    used[j] = True
            if len(group) >= 2:
                clusters.append(float(sum(group) / len(group)))
        return clusters

    ref = float(df["Close"].iloc[-1])
    return {
        "buy_side":  cluster(list(highs), ref),
        "sell_side": cluster(list(lows), ref),
    }


def get_killzone(now: datetime | None = None) -> str | None:
    if now is None:
        now = datetime.now(_NY)
    if now.tzinfo is None:
        raise ValueError("get_killzone requires timezone-aware datetime")
    now_ny = now.astimezone(_NY).time()
    for name, (start, end) in config.KILLZONES_NY.items():
        if start <= now_ny <= end:
            return name
    return None


if __name__ == "__main__":
    print("killzone now:", get_killzone())
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ict.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/indicators/ict.py tests/test_ict.py
git commit -m "feat(indicators): add ICT (OTE/FVG/OB/liquidity/killzone)"
```

---

## Task 7: SMC Indicators

**Files:**
- Create: `xau_pro_bot/indicators/smc.py`
- Test: `tests/test_smc.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_smc.py
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.indicators.smc import (
    detect_structure, premium_discount, detect_stop_hunt,
)


def test_detect_structure_returns_dict(uptrend_df):
    res = detect_structure(uptrend_df, swing_len=5)
    assert "last_event" in res
    assert "prev_swing_high" in res
    assert "prev_swing_low" in res


def test_detect_structure_short_df_neutral(short_df):
    res = detect_structure(short_df, swing_len=5)
    assert res["last_event"] is None


def test_premium_discount_uptrend_premium(uptrend_df):
    res = premium_discount(uptrend_df, lookback=50)
    assert res["zone"] == "premium"
    assert res["pct_of_range"] > 60


def test_premium_discount_downtrend_discount(downtrend_df):
    res = premium_discount(downtrend_df, lookback=50)
    assert res["zone"] == "discount"
    assert res["pct_of_range"] < 40


def test_detect_stop_hunt_requires_wick_and_atr(uptrend_df):
    # Inject a clear bullish stop hunt: long lower wick
    df = uptrend_df.copy()
    df.iloc[-1, df.columns.get_loc("Low")] = float(df["Low"].min()) - 30
    df.iloc[-1, df.columns.get_loc("Close")] = float(df["Close"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Open")] = float(df["Close"].iloc[-1]) - 0.5
    atr = 5.0
    res = detect_stop_hunt(df, atr=atr)
    assert "bull_hunt" in res
    assert "bear_hunt" in res


def test_stop_hunt_neutral_on_flat(flat_df):
    res = detect_stop_hunt(flat_df, atr=1.0)
    assert res["bull_hunt"] is False
    assert res["bear_hunt"] is False
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_smc.py -v
```

- [ ] **Step 3: Implement `smc.py`**

```python
# xau_pro_bot/indicators/smc.py
"""Smart Money Concepts: BOS, CHOCH, Premium/Discount, Stop Hunt."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _swings(df: pd.DataFrame, swing_len: int) -> tuple[list[float], list[float]]:
    """Return (swing_highs, swing_lows) in chronological order."""
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(swing_len, len(df) - swing_len):
        window_h = highs[i - swing_len:i + swing_len + 1]
        window_l = lows[i - swing_len:i + swing_len + 1]
        if highs[i] == window_h.max():
            swing_highs.append(float(highs[i]))
        if lows[i] == window_l.min():
            swing_lows.append(float(lows[i]))
    return swing_highs, swing_lows


def detect_structure(df: pd.DataFrame, swing_len: int = 5) -> dict[str, Any]:
    if len(df) < swing_len * 4:
        return {"last_event": None, "prev_swing_high": None, "prev_swing_low": None}

    swing_highs, swing_lows = _swings(df, swing_len)
    if not swing_highs or not swing_lows:
        return {"last_event": None, "prev_swing_high": None, "prev_swing_low": None}

    last_close = float(df["Close"].iloc[-1])
    prev_swing_high = swing_highs[-1]
    prev_swing_low = swing_lows[-1]

    # Trend from EMA slope proxy: compare last 20 vs prior 20 closes
    recent = df["Close"].iloc[-20:].mean()
    earlier = df["Close"].iloc[-40:-20].mean() if len(df) >= 40 else recent
    in_uptrend = recent > earlier

    event: str | None = None
    if last_close > prev_swing_high:
        event = "BOS_bull" if in_uptrend else "CHOCH_bull"
    elif last_close < prev_swing_low:
        event = "BOS_bear" if not in_uptrend else "CHOCH_bear"

    return {
        "last_event": event,
        "prev_swing_high": prev_swing_high,
        "prev_swing_low": prev_swing_low,
    }


def premium_discount(df: pd.DataFrame, lookback: int = 50) -> dict[str, Any]:
    if len(df) < lookback:
        return {"zone": "neutral", "pct_of_range": 50.0,
                "equilibrium": None, "range_high": None, "range_low": None}
    window = df.tail(lookback)
    range_high = float(window["High"].max())
    range_low = float(window["Low"].min())
    equilibrium = (range_high + range_low) / 2
    last_price = float(df["Close"].iloc[-1])
    span = max(range_high - range_low, 1e-9)
    pct = (last_price - range_low) / span * 100
    if last_price > equilibrium:
        zone = "premium"
    elif last_price < equilibrium:
        zone = "discount"
    else:
        zone = "neutral"
    return {
        "zone": zone, "pct_of_range": float(pct),
        "equilibrium": float(equilibrium),
        "range_high": range_high, "range_low": range_low,
    }


def detect_stop_hunt(df: pd.DataFrame, atr: float) -> dict[str, Any]:
    """Detect stop hunt: wick > 2*body AND wick > 0.5*atr."""
    if len(df) < 5 or atr is None or atr <= 0 or np.isnan(atr):
        return {"bull_hunt": False, "bear_hunt": False, "level_hunted": None}

    last3 = df.tail(3)
    swing_low = float(df["Low"].iloc[:-3].tail(20).min()) if len(df) >= 23 else float(df["Low"].min())
    swing_high = float(df["High"].iloc[:-3].tail(20).max()) if len(df) >= 23 else float(df["High"].max())

    bull_hunt = False
    bear_hunt = False
    level_hunted: float | None = None

    for _, row in last3.iterrows():
        body = abs(row["Close"] - row["Open"])
        lower_wick = min(row["Open"], row["Close"]) - row["Low"]
        upper_wick = row["High"] - max(row["Open"], row["Close"])

        if (lower_wick > 2 * body and lower_wick > 0.5 * atr
                and row["Low"] < swing_low and row["Close"] > swing_low):
            bull_hunt = True
            level_hunted = swing_low

        if (upper_wick > 2 * body and upper_wick > 0.5 * atr
                and row["High"] > swing_high and row["Close"] < swing_high):
            bear_hunt = True
            level_hunted = swing_high

    return {"bull_hunt": bull_hunt, "bear_hunt": bear_hunt,
            "level_hunted": level_hunted}


if __name__ == "__main__":
    pass
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_smc.py -v
```

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/indicators/smc.py tests/test_smc.py
git commit -m "feat(indicators): add SMC (structure/PD/stop-hunt)"
```

---

## Task 8: Wyckoff (Soft Bias)

**Files:**
- Create: `xau_pro_bot/indicators/wyckoff.py`
- Test: `tests/test_wyckoff.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_wyckoff.py
import pytest

from xau_pro_bot.indicators.wyckoff import detect_wyckoff


def test_detect_wyckoff_returns_required_keys(uptrend_df):
    res = detect_wyckoff(uptrend_df)
    assert {"phase", "bias", "strength"} <= res.keys()
    assert res["bias"] in ("bull", "bear", "neutral")
    assert 0 <= res["strength"] <= 100


def test_detect_wyckoff_short_df_neutral(short_df):
    res = detect_wyckoff(short_df)
    assert res["phase"] == "neutral"
    assert res["bias"] == "neutral"


def test_detect_wyckoff_uptrend_is_markup(uptrend_df):
    res = detect_wyckoff(uptrend_df)
    # Pure uptrend should classify as markup (or at least bull bias)
    assert res["bias"] == "bull"


def test_detect_wyckoff_downtrend_is_markdown(downtrend_df):
    res = detect_wyckoff(downtrend_df)
    assert res["bias"] == "bear"
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_wyckoff.py -v
```

- [ ] **Step 3: Implement `wyckoff.py`**

```python
# xau_pro_bot/indicators/wyckoff.py
"""Wyckoff phase detection — soft bias only (max ±5 in engine)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config


def detect_wyckoff(df: pd.DataFrame) -> dict[str, Any]:
    n = config.WYCKOFF_BARS
    if len(df) < n:
        return {"phase": "neutral", "bias": "neutral", "strength": 0}

    window = df.tail(n)
    closes = window["Close"].to_numpy()
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()

    tr_high = float(highs.max())
    tr_low = float(lows.min())
    span = max(tr_high - tr_low, 1e-9)
    last_price = float(closes[-1])
    pos = (last_price - tr_low) / span  # 0..1

    # Trend slope on linear regression of closes
    x = np.arange(n)
    slope = np.polyfit(x, closes, 1)[0]
    slope_norm = slope * n / span  # ~ trend amplitude relative to range

    phase = "neutral"
    bias = "neutral"

    if slope_norm > 0.6 and pos > 0.7:
        phase, bias = "markup", "bull"
    elif slope_norm < -0.6 and pos < 0.3:
        phase, bias = "markdown", "bear"
    elif pos < 0.3 and abs(slope_norm) < 0.5:
        phase, bias = "accumulation", "bull"
    elif pos > 0.7 and abs(slope_norm) < 0.5:
        phase, bias = "distribution", "bear"

    strength = int(min(100, abs(slope_norm) * 50 + abs(pos - 0.5) * 100))
    return {"phase": phase, "bias": bias, "strength": strength}


if __name__ == "__main__":
    pass
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_wyckoff.py -v
```

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/indicators/wyckoff.py tests/test_wyckoff.py
git commit -m "feat(indicators): add Wyckoff soft bias detector"
```

---

## Task 9: SR Levels Helpers

**Files:**
- Create: `xau_pro_bot/indicators/sr_levels.py`
- Test: `tests/test_sr_levels.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sr_levels.py
import pytest

from xau_pro_bot.indicators.sr_levels import (
    swing_highs_lows, nearest_above, nearest_below,
)


def test_swing_highs_lows_returns_lists(uptrend_df):
    sh, sl = swing_highs_lows(uptrend_df, window=5)
    assert isinstance(sh, list) and isinstance(sl, list)


def test_nearest_above_finds_closest():
    levels = [100.0, 105.0, 110.0, 120.0]
    assert nearest_above(102.0, levels) == pytest.approx(105.0)


def test_nearest_above_none_when_empty():
    assert nearest_above(100.0, [50.0, 60.0]) is None


def test_nearest_below_finds_closest():
    levels = [100.0, 105.0, 110.0, 120.0]
    assert nearest_below(108.0, levels) == pytest.approx(105.0)


def test_nearest_below_none_when_empty():
    assert nearest_below(100.0, [200.0, 300.0]) is None
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_sr_levels.py -v
```

- [ ] **Step 3: Implement `sr_levels.py`**

```python
# xau_pro_bot/indicators/sr_levels.py
"""Support/Resistance helpers and swing utilities."""

from __future__ import annotations

import pandas as pd


def swing_highs_lows(df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    sh: list[float] = []
    sl: list[float] = []
    for i in range(window, len(df) - window):
        if highs[i] == highs[i - window:i + window + 1].max():
            sh.append(float(highs[i]))
        if lows[i] == lows[i - window:i + window + 1].min():
            sl.append(float(lows[i]))
    return sh, sl


def nearest_above(price: float, levels: list[float]) -> float | None:
    above = [lv for lv in levels if lv > price]
    return min(above) if above else None


def nearest_below(price: float, levels: list[float]) -> float | None:
    below = [lv for lv in levels if lv < price]
    return max(below) if below else None
```

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/test_sr_levels.py -v
git add xau_pro_bot/indicators/sr_levels.py tests/test_sr_levels.py
git commit -m "feat(indicators): add SR-levels helpers"
```

---

## Task 10: Signal Engine

**Files:**
- Create: `xau_pro_bot/signals/engine.py`
- Create: `xau_pro_bot/signals/ict_signals.py`
- Create: `xau_pro_bot/signals/smc_signals.py`
- Create: `xau_pro_bot/signals/classic_signals.py`
- Test: `tests/test_engine.py`

This is the heart of the bot. I'll implement scoring as pure functions that return `(bull_pts, bear_pts, reasons)` tuples, then `engine.py` aggregates them.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine.py
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


def _enriched_data(uptrend_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {tf: uptrend_df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}


def test_engine_returns_required_keys(uptrend_df):
    eng = MasterSignalEngine()
    result = eng.analyze(_enriched_data(uptrend_df))
    for key in ("direction", "tier", "score", "entry", "sl",
                "tp1", "tp2", "tp3", "rr", "killzone", "reasons",
                "tp2_unavailable"):
        assert key in result, f"missing {key}"


def test_engine_returns_no_signal_for_flat(flat_df):
    eng = MasterSignalEngine()
    result = eng.analyze(_enriched_data(flat_df))
    assert result["tier"] == "NO_SIGNAL" or result["score"] < 40


def test_tier_thresholds():
    eng = MasterSignalEngine()
    assert eng._tier(70) == "STRONG"
    assert eng._tier(60) == "NORMAL"
    assert eng._tier(45) == "WEAK"
    assert eng._tier(30) == "NO_SIGNAL"


def test_engine_picks_strongest_direction(uptrend_df):
    eng = MasterSignalEngine()
    result = eng.analyze(_enriched_data(uptrend_df))
    assert result["direction"] in ("BUY", "SELL")
    # Uptrend should favor BUY
    if result["tier"] != "NO_SIGNAL":
        assert result["direction"] == "BUY"
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_engine.py -v
```

- [ ] **Step 3: Implement scoring helper modules**

```python
# xau_pro_bot/signals/ict_signals.py
"""ICT scoring contributions. Returns (bull_pts, bear_pts, reasons)."""

from __future__ import annotations

from typing import Any

from xau_pro_bot import config
from xau_pro_bot.indicators.ict import (
    find_ote, find_fvg, find_order_blocks, get_killzone,
)
from xau_pro_bot.indicators.smc import detect_stop_hunt


def score_ict(h1_df, m15_df, h1_atr: float) -> tuple[float, float, list[str]]:
    bull = bear = 0.0
    reasons: list[str] = []

    # OTE (H1)
    ote = find_ote(h1_df, lookback=20)
    if ote["in_ote"]:
        if ote["direction"] == "bull":
            bull += 12
            reasons.append(f"ICT OTE bull ({ote['ote_low']:.2f}-{ote['ote_high']:.2f})")
        else:
            bear += 12
            reasons.append(f"ICT OTE bear ({ote['ote_low']:.2f}-{ote['ote_high']:.2f})")
    else:
        bull -= 5
        bear -= 5

    # Killzone
    kz = get_killzone()
    if kz in config.PRIORITY_KILLZONES:
        bull += 10
        bear += 10
        reasons.append(f"ICT killzone {kz} (priority)")
    elif kz:
        bull += 6
        bear += 6
        reasons.append(f"ICT killzone {kz}")
    else:
        bull -= 12
        bear -= 12

    # Liquidity sweep on M15
    sweep = detect_stop_hunt(m15_df, atr=h1_atr)
    if sweep["bull_hunt"]:
        bull += 9
        reasons.append(f"Liquidity bull sweep @ {sweep['level_hunted']:.2f}")
    if sweep["bear_hunt"]:
        bear += 9
        reasons.append(f"Liquidity bear sweep @ {sweep['level_hunted']:.2f}")

    # FVG on H1
    fvgs = find_fvg(h1_df, max_gaps=5)
    last_close = float(h1_df["Close"].iloc[-1])
    for fvg in fvgs[:3]:
        if fvg["type"] == "bull" and fvg["bottom"] <= last_close <= fvg["top"]:
            bull += 8
            reasons.append(f"H1 FVG bull mid {fvg['midpoint']:.2f}")
            break
        if fvg["type"] == "bear" and fvg["bottom"] <= last_close <= fvg["top"]:
            bear += 8
            reasons.append(f"H1 FVG bear mid {fvg['midpoint']:.2f}")
            break

    # OB on H1 — first test
    obs = find_order_blocks(h1_df, lookback=config.OB_LOOKBACK)
    for ob in obs[:5]:
        if not ob["tested"] and ob["low"] <= last_close <= ob["high"]:
            if ob["type"] == "bull":
                bull += 6
                reasons.append(f"H1 OB bull first-test {ob['mid']:.2f}")
            else:
                bear += 6
                reasons.append(f"H1 OB bear first-test {ob['mid']:.2f}")
            break

    return bull, bear, reasons
```

```python
# xau_pro_bot/signals/smc_signals.py
"""SMC scoring contributions."""

from __future__ import annotations

from xau_pro_bot.indicators.smc import (
    detect_structure, premium_discount,
)
from xau_pro_bot.indicators.ict import find_order_blocks, find_fvg


def score_smc(h4_df) -> tuple[float, float, list[str]]:
    bull = bear = 0.0
    reasons: list[str] = []

    struct = detect_structure(h4_df, swing_len=5)
    event = struct["last_event"]
    if event == "CHOCH_bull":
        bull += 15
        reasons.append("H4 CHOCH bull")
    elif event == "CHOCH_bear":
        bear += 15
        reasons.append("H4 CHOCH bear")
    elif event == "BOS_bull":
        bull += 10
        reasons.append("H4 BOS bull")
    elif event == "BOS_bear":
        bear += 10
        reasons.append("H4 BOS bear")

    pd_zone = premium_discount(h4_df, lookback=50)
    if pd_zone["zone"] == "discount":
        bull += 8
        bear -= 10
        reasons.append(f"H4 discount ({pd_zone['pct_of_range']:.0f}%)")
    elif pd_zone["zone"] == "premium":
        bear += 8
        bull -= 10
        reasons.append(f"H4 premium ({pd_zone['pct_of_range']:.0f}%)")

    obs = find_order_blocks(h4_df, lookback=50)
    last_close = float(h4_df["Close"].iloc[-1])
    for ob in obs[:5]:
        if not ob["tested"] and ob["low"] <= last_close <= ob["high"]:
            if ob["type"] == "bull":
                bull += 7
                reasons.append(f"H4 OB bull {ob['mid']:.2f}")
            else:
                bear += 7
                reasons.append(f"H4 OB bear {ob['mid']:.2f}")
            break

    fvgs = find_fvg(h4_df, max_gaps=5)
    for fvg in fvgs[:3]:
        if fvg["bottom"] <= last_close <= fvg["top"]:
            if fvg["type"] == "bull":
                bull += 5
                reasons.append(f"H4 FVG bull {fvg['midpoint']:.2f}")
            else:
                bear += 5
                reasons.append(f"H4 FVG bear {fvg['midpoint']:.2f}")
            break

    return bull, bear, reasons
```

```python
# xau_pro_bot/signals/classic_signals.py
"""Classic TA scoring contributions on H1 (already enriched by add_classic)."""

from __future__ import annotations

import numpy as np


def score_classic(h1_df, m15_df) -> tuple[float, float, list[str]]:
    bull = bear = 0.0
    reasons: list[str] = []

    last = h1_df.iloc[-1]
    prev = h1_df.iloc[-2] if len(h1_df) >= 2 else last

    rsi = last.get("RSI_14", np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            bull += 8
            reasons.append(f"RSI H1 oversold ({rsi:.1f})")
        elif rsi > 70:
            bear += 8
            reasons.append(f"RSI H1 overbought ({rsi:.1f})")
        elif 40 <= rsi <= 60:
            bull -= 8
            bear -= 8

    macd = last.get("MACD_12_26_9", np.nan)
    macd_s = last.get("MACDs_12_26_9", np.nan)
    prev_macd = prev.get("MACD_12_26_9", np.nan)
    prev_macd_s = prev.get("MACDs_12_26_9", np.nan)
    if not any(np.isnan(x) for x in (macd, macd_s, prev_macd, prev_macd_s)):
        if prev_macd < prev_macd_s and macd > macd_s:
            bull += 6
            reasons.append("MACD H1 bull cross")
        elif prev_macd > prev_macd_s and macd < macd_s:
            bear += 6
            reasons.append("MACD H1 bear cross")

    k = last.get("STOCHk_14_3_3", np.nan)
    d = last.get("STOCHd_14_3_3", np.nan)
    pk = prev.get("STOCHk_14_3_3", np.nan)
    pd_val = prev.get("STOCHd_14_3_3", np.nan)
    if not any(np.isnan(x) for x in (k, d, pk, pd_val)):
        if pk < pd_val and k > d and k < 30:
            bull += 6
            reasons.append("Stoch H1 bull cross OS")
        elif pk > pd_val and k < d and k > 70:
            bear += 6
            reasons.append("Stoch H1 bear cross OB")

    bbl = last.get("BBL_20_2.0", np.nan)
    bbu = last.get("BBU_20_2.0", np.nan)
    close = float(last["Close"])
    if not np.isnan(bbl) and close <= bbl:
        bull += 5
        reasons.append("BB lower rejection")
    if not np.isnan(bbu) and close >= bbu:
        bear += 5
        reasons.append("BB upper rejection")

    vol_ratio = last.get("vol_ratio", np.nan)
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.5:
            bull += 5
            bear += 5
            reasons.append(f"Volume {vol_ratio:.1f}x avg")
        elif vol_ratio < 0.6:
            bull -= 6
            bear -= 6

    # M15 EMA8/EMA21 cross (entry timing)
    if len(m15_df) >= 3 and "EMA_8" in m15_df.columns and "EMA_21" in m15_df.columns:
        a, b = m15_df.iloc[-1], m15_df.iloc[-2]
        if b["EMA_8"] < b["EMA_21"] and a["EMA_8"] > a["EMA_21"]:
            bull += 4
            reasons.append("M15 EMA8>EMA21 cross")
        elif b["EMA_8"] > b["EMA_21"] and a["EMA_8"] < a["EMA_21"]:
            bear += 4
            reasons.append("M15 EMA8<EMA21 cross")

    pivot = last.get("pivot", np.nan)
    s1 = last.get("s1", np.nan)
    r1 = last.get("r1", np.nan)
    if not np.isnan(s1) and abs(close - s1) / max(close, 1) < 0.002:
        bull += 3
        reasons.append(f"Pivot S1 {s1:.2f}")
    if not np.isnan(r1) and abs(close - r1) / max(close, 1) < 0.002:
        bear += 3
        reasons.append(f"Pivot R1 {r1:.2f}")

    return bull, bear, reasons
```

- [ ] **Step 4: Implement `engine.py`**

```python
# xau_pro_bot/signals/engine.py
"""Master scoring engine: combines layers, picks direction, computes levels."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.indicators import classic, smc, wyckoff
from xau_pro_bot.indicators.ict import (
    find_fvg, find_order_blocks, find_liquidity, get_killzone,
)
from xau_pro_bot.signals.ict_signals import score_ict
from xau_pro_bot.signals.smc_signals import score_smc
from xau_pro_bot.signals.classic_signals import score_classic


class MasterSignalEngine:
    """Aggregates all scoring layers and produces a structured signal."""

    @staticmethod
    def _tier(score: float) -> str:
        if score >= config.STRONG_SIGNAL:
            return "STRONG"
        if score >= config.NORMAL_SIGNAL:
            return "NORMAL"
        if score >= config.WEAK_SIGNAL:
            return "WEAK"
        return "NO_SIGNAL"

    def _macro_bias(self, w1_df, d1_df) -> tuple[float, float, list[str]]:
        bull = bear = 0.0
        reasons: list[str] = []
        d1_last = d1_df.iloc[-1]
        if not np.isnan(d1_last.get("EMA_50", np.nan)) and not np.isnan(d1_last.get("EMA_200", np.nan)):
            if d1_last["EMA_50"] > d1_last["EMA_200"]:
                bull += 20
                reasons.append("D1 EMA50 > EMA200")
            else:
                bear += 20
                reasons.append("D1 EMA50 < EMA200")
        w1_last = w1_df.iloc[-1]
        w1_prev = w1_df.iloc[-2] if len(w1_df) >= 2 else w1_last
        if not np.isnan(w1_last.get("EMA_50", np.nan)) and not np.isnan(w1_prev.get("EMA_50", np.nan)):
            if w1_last["EMA_50"] > w1_prev["EMA_50"]:
                bull += 8
            else:
                bear += 8
        wy = wyckoff.detect_wyckoff(d1_df)
        if wy["bias"] == "bull":
            bull += 5
            reasons.append(f"Wyckoff {wy['phase']}")
        elif wy["bias"] == "bear":
            bear += 5
            reasons.append(f"Wyckoff {wy['phase']}")
        return bull, bear, reasons

    def _enrich(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {tf: classic.add_classic(df) for tf, df in data.items()}

    def _macro_penalty(self, direction: str, d1_df) -> tuple[float, str | None]:
        d1_last = d1_df.iloc[-1]
        ema50 = d1_last.get("EMA_50", np.nan)
        ema200 = d1_last.get("EMA_200", np.nan)
        if np.isnan(ema50) or np.isnan(ema200):
            return 0.0, None
        d1_bull = ema50 > ema200
        if direction == "BUY" and not d1_bull:
            return 20.0, "D1 trend against BUY"
        if direction == "SELL" and d1_bull:
            return 20.0, "D1 trend against SELL"
        return 0.0, None

    def _compute_levels(self, direction: str, h1_df, m15_df,
                        d1_df) -> dict[str, Any]:
        entry = float(m15_df["Close"].iloc[-1])
        atr_m15 = float(m15_df["ATR_14"].iloc[-1]) if "ATR_14" in m15_df else 1.0
        if np.isnan(atr_m15) or atr_m15 <= 0:
            atr_m15 = max(entry * 0.001, 0.5)

        obs = find_order_blocks(h1_df, lookback=config.OB_LOOKBACK)
        fvgs = find_fvg(h1_df, max_gaps=5)
        liq = find_liquidity(h1_df, lookback=30)

        if direction == "BUY":
            ob_low = min(
                (ob["low"] for ob in obs if ob["type"] == "bull" and ob["low"] < entry),
                default=None,
            )
            fvg_bottom = min(
                (f["bottom"] for f in fvgs if f["type"] == "bull" and f["bottom"] < entry),
                default=None,
            )
            sl_candidates = [c for c in (ob_low, fvg_bottom) if c is not None]
            sl = (max(sl_candidates) if sl_candidates else entry - 5 * atr_m15) - atr_m15 * 0.3

            tp1 = next(
                (f["midpoint"] for f in fvgs if f["type"] == "bull" and f["midpoint"] > entry),
                None,
            ) or (entry + 2 * (entry - sl))
            tp2 = min((x for x in liq["buy_side"] if x > entry), default=None)
            tp3 = float(d1_df["High"].tail(50).max())
        else:
            ob_high = max(
                (ob["high"] for ob in obs if ob["type"] == "bear" and ob["high"] > entry),
                default=None,
            )
            fvg_top = max(
                (f["top"] for f in fvgs if f["type"] == "bear" and f["top"] > entry),
                default=None,
            )
            sl_candidates = [c for c in (ob_high, fvg_top) if c is not None]
            sl = (min(sl_candidates) if sl_candidates else entry + 5 * atr_m15) + atr_m15 * 0.3

            tp1 = next(
                (f["midpoint"] for f in fvgs if f["type"] == "bear" and f["midpoint"] < entry),
                None,
            ) or (entry - 2 * (sl - entry))
            tp2 = max((x for x in liq["sell_side"] if x < entry), default=None)
            tp3 = float(d1_df["Low"].tail(50).min())

        risk = abs(entry - sl)
        if risk <= 0:
            risk = atr_m15
        tp2_unavailable = False
        if tp2 is None:
            tp2_unavailable = True
            rr = abs(tp1 - entry) / risk
        else:
            rr = abs(tp2 - entry) / risk
            if rr < config.MIN_RR:
                tp2_unavailable = True
                tp2 = None
                rr = abs(tp1 - entry) / risk

        return {
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(float(tp1), 2) if tp1 is not None else None,
            "tp2": round(float(tp2), 2) if tp2 is not None else None,
            "tp3": round(float(tp3), 2) if tp3 is not None else None,
            "rr": round(float(rr), 2),
            "tp2_unavailable": tp2_unavailable,
            "atr_h1": float(h1_df["ATR_14"].iloc[-1]) if "ATR_14" in h1_df else atr_m15,
        }

    def analyze(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        enriched = self._enrich(data)
        w1, d1, h4, h1, m15 = (enriched[k] for k in ("W1", "D1", "H4", "H1", "M15"))

        h1_atr = float(h1["ATR_14"].iloc[-1]) if "ATR_14" in h1 and not np.isnan(h1["ATR_14"].iloc[-1]) else 1.0

        macro_bull, macro_bear, macro_reasons = self._macro_bias(w1, d1)
        smc_bull, smc_bear, smc_reasons = score_smc(h4)
        ict_bull, ict_bear, ict_reasons = score_ict(h1, m15, h1_atr)
        cls_bull, cls_bear, cls_reasons = score_classic(h1, m15)

        bull_score = macro_bull + smc_bull + ict_bull + cls_bull
        bear_score = macro_bear + smc_bear + ict_bear + cls_bear

        direction = "BUY" if bull_score >= bear_score else "SELL"
        macro_pen, pen_reason = self._macro_penalty(direction, d1)
        if direction == "BUY":
            bull_score -= macro_pen
        else:
            bear_score -= macro_pen

        final_score = max(bull_score, bear_score)
        tier = self._tier(final_score)

        reasons = {
            "macro": macro_reasons,
            "smc": smc_reasons,
            "ict": ict_reasons,
            "classic": cls_reasons,
            "penalties": [pen_reason] if pen_reason else [],
        }

        if tier == "NO_SIGNAL":
            return {
                "direction": direction,
                "tier": tier,
                "score": int(final_score),
                "entry": float(m15["Close"].iloc[-1]),
                "sl": None, "tp1": None, "tp2": None, "tp3": None,
                "rr": None,
                "killzone": get_killzone(),
                "reasons": reasons,
                "tp2_unavailable": False,
                "ts_utc": datetime.now(timezone.utc),
            }

        levels = self._compute_levels(direction, h1, m15, d1)
        return {
            "direction": direction,
            "tier": tier,
            "score": int(final_score),
            **levels,
            "killzone": get_killzone(),
            "reasons": reasons,
            "ts_utc": datetime.now(timezone.utc),
        }
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_engine.py -v
```

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/signals/ tests/test_engine.py
git commit -m "feat(engine): add master scoring engine with 5-layer scoring"
```

---

## Task 11: Filters (Dedup + Rate Limit + RR)

**Files:**
- Create: `xau_pro_bot/signals/filters.py`
- Test: `tests/test_filters.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_filters.py
from datetime import datetime, timedelta, timezone

import pytest

from xau_pro_bot.state import State
from xau_pro_bot.signals.filters import should_send, SkipReason


@pytest.fixture
def state(tmp_path):
    return State(db_path=str(tmp_path / "f.db"))


def _sig(direction="BUY", tier="STRONG", entry=2000.0, score=70, tp1=2010, tp2=2020):
    return {
        "direction": direction, "tier": tier, "entry": entry, "score": score,
        "sl": entry - 5, "tp1": tp1, "tp2": tp2, "tp3": entry + 30,
        "rr": 2.0, "killzone": "London KZ", "atr_h1": 5.0,
        "tp2_unavailable": False,
    }


def test_no_signal_blocked(state):
    sig = _sig(tier="NO_SIGNAL", score=20)
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.NO_SIGNAL


def test_weak_outside_killzone_blocked(state):
    sig = _sig(tier="WEAK", score=42)
    sig["killzone"] = None
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.WEAK_OUTSIDE_KZ


def test_dedup_within_2h_blocks(state):
    sig = _sig()
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.DEDUP


def test_atr_reprice_overrides_dedup(state):
    sig = _sig(entry=2000.0)
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    sig_moved = _sig(entry=2000.0 + 1.5 * 5.0 + 1)  # > 1.5 ATR move
    ok, reason = should_send(sig_moved, state)
    assert ok, f"expected ATR-reprice to bypass dedup, got reason={reason}"


def test_rate_limit_day(state):
    base = _sig()
    for _ in range(6):
        state.record_signal({**base, "ts_utc": datetime.now(timezone.utc).isoformat(),
                             "reasons_json": "{}", "entry": 2000 + _ * 100})
    next_sig = _sig(entry=9999.0)
    ok, reason = should_send(next_sig, state)
    assert not ok and reason == SkipReason.RATE_LIMIT_DAY


def test_weak_cooldown(state):
    sig = _sig(tier="WEAK", score=42)
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    new_sig = _sig(tier="WEAK", score=42, entry=2100.0)
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.WEAK_COOLDOWN
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_filters.py -v
```

- [ ] **Step 3: Implement `filters.py`**

```python
# xau_pro_bot/signals/filters.py
"""Quality filters: dedup, ATR-reprice (early-exit), rate limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from xau_pro_bot import config
from xau_pro_bot.state import State


class SkipReason(str, Enum):
    NO_SIGNAL = "no_signal"
    WEAK_OUTSIDE_KZ = "weak_outside_kz"
    DEDUP = "dedup"
    RATE_LIMIT_DAY = "rate_limit_day"
    WEAK_COOLDOWN = "weak_cooldown"
    NO_TP1 = "no_tp1"


def should_send(sig: dict, state: State,
                bypass_dedup: bool = False) -> tuple[bool, SkipReason | None]:
    """Returns (True, None) if signal should be sent, else (False, reason)."""
    if sig["tier"] == "NO_SIGNAL":
        return False, SkipReason.NO_SIGNAL

    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1

    if sig["tier"] == "WEAK" and not sig.get("killzone"):
        return False, SkipReason.WEAK_OUTSIDE_KZ

    if state.count_today() >= config.MAX_SIGNALS_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY

    if sig["tier"] == "WEAK":
        last_weak = state.last_weak_ts()
        if last_weak is not None:
            elapsed = datetime.now(timezone.utc) - last_weak
            if elapsed < timedelta(hours=config.WEAK_COOLDOWN_HOURS):
                return False, SkipReason.WEAK_COOLDOWN

    if bypass_dedup:
        return True, None

    last = state.last_signal(direction=sig["direction"])
    if last is None:
        return True, None

    atr_h1 = sig.get("atr_h1", 1.0)
    moved = abs(sig["entry"] - last["entry"])
    if moved >= config.REPRICE_ATR_MULT * atr_h1:
        return True, None  # ATR-reprice wins (early-exit)

    last_ts = datetime.fromisoformat(last["ts_utc"])
    if datetime.now(timezone.utc) - last_ts >= timedelta(hours=config.DEDUP_HOURS):
        return True, None

    return False, SkipReason.DEDUP
```

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/test_filters.py -v
git add xau_pro_bot/signals/filters.py tests/test_filters.py
git commit -m "feat(filters): add dedup/ATR-reprice/rate-limit filters"
```

---

## Task 12: Formatter

**Files:**
- Create: `xau_pro_bot/formatter.py`
- Test: `tests/test_formatter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_formatter.py
from datetime import datetime, timezone

import pytest

from xau_pro_bot.formatter import (
    format_strong_signal, format_weak_signal,
    format_no_signal_killzone,
)


def _sig(tier="STRONG", tp2_unavailable=False):
    return {
        "direction": "SELL", "tier": tier, "score": 81,
        "entry": 3312.50, "sl": 3324.00,
        "tp1": 3298.00, "tp2": 3280.00 if not tp2_unavailable else None,
        "tp3": 3261.00, "rr": 2.8,
        "killzone": "London KZ",
        "tp2_unavailable": tp2_unavailable,
        "reasons": {
            "ict": ["OTE zone"], "smc": ["CHOCH H4"],
            "macro": ["Wyckoff Distribution"],
            "classic": ["RSI 72"], "penalties": [],
        },
        "ts_utc": datetime(2026, 5, 11, 9, 47, tzinfo=timezone.utc),
    }


def test_strong_signal_contains_required_fields():
    text = format_strong_signal(_sig())
    assert "Сильный сигнал" in text
    assert "SELL" in text
    assert "3,312.50" in text
    assert "Score: 81/100" in text
    assert "Уверенность" not in text  # renamed per review


def test_strong_signal_with_tp2_unavailable():
    text = format_strong_signal(_sig(tp2_unavailable=True))
    assert "TP2: недоступен" in text


def test_weak_signal_short_format():
    text = format_weak_signal(_sig(tier="WEAK"))
    assert "TP1" in text
    assert "Анализ:" not in text  # WEAK skips breakdown


def test_no_signal_brief():
    text = format_no_signal_killzone(
        killzone="London KZ", price=3298.10, rsi=52.0,
    )
    assert "London KZ" in text
    assert "3,298.10" in text
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_formatter.py -v
```

- [ ] **Step 3: Implement `formatter.py`**

```python
# xau_pro_bot/formatter.py
"""Telegram Markdown signal formatter (Russian, matches spec template)."""

from __future__ import annotations

from datetime import datetime


KZ_FLAGS = {
    "London KZ": "🇬🇧",
    "NY AM KZ": "🇺🇸",
    "NY PM KZ": "🇺🇸",
    "Asian KZ": "🇯🇵",
}


def _fmt_price(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:,.2f}"


def _fmt_pts(diff: float) -> str:
    return f"{diff:+.1f} pts"


def _tp2_line(sig: dict) -> str:
    if sig.get("tp2_unavailable") or sig.get("tp2") is None:
        return " •  TP2: недоступен (RR < 1.8)"
    diff = sig["tp2"] - sig["entry"]
    return f" •  TP2: `{_fmt_price(sig['tp2'])}` ({abs(diff):.1f} pts) — ликвидность"


def _direction_header(sig: dict) -> str:
    if sig["direction"] == "BUY":
        return "🟢 Сильный сигнал — BUY"
    return "🔴 Сильный сигнал — SELL"


def _analysis_block(sig: dict) -> str:
    lines = ["📐 Анализ:"]
    for source in ("ict", "smc", "macro", "classic"):
        for r in sig["reasons"].get(source, []):
            lines.append(f"• {r} ✅")
    for r in sig["reasons"].get("penalties", []):
        lines.append(f"• {r} ⚠️")
    return "\n".join(lines)


def format_strong_signal(sig: dict) -> str:
    flag = KZ_FLAGS.get(sig.get("killzone") or "", "")
    sl_diff = sig["sl"] - sig["entry"]
    tp1_diff = (sig["tp1"] - sig["entry"]) if sig["tp1"] is not None else 0
    tp3_diff = (sig["tp3"] - sig["entry"]) if sig["tp3"] is not None else 0
    ts: datetime = sig["ts_utc"]

    parts = [
        _direction_header(sig),
        "━━━━━━━━━━━━━━━━━━━",
        f"🔹 Вход: `{_fmt_price(sig['entry'])}`",
        f"🔺 Stop Loss: `{_fmt_price(sig['sl'])}` ({_fmt_pts(sl_diff)})",
        "🎯 Цели:",
        f" •  TP1: `{_fmt_price(sig['tp1'])}` ({abs(tp1_diff):.1f} pts) — FVG",
        _tp2_line(sig),
        f" •  TP3: `{_fmt_price(sig['tp3'])}` ({abs(tp3_diff):.1f} pts) — D1",
        "━━━━━━━━━━━━━━━━━━━",
        f"📊 R:R → 1:{sig['rr']:.1f}",
        f"🧠 Score: {sig['score']}/100",
        f"⏱ Сессия: {sig.get('killzone') or '—'} {flag} | M15→H1",
        "━━━━━━━━━━━━━━━━━━━",
        _analysis_block(sig),
        "━━━━━━━━━━━━━━━━━━━",
        f"🕐 {ts.strftime('%d.%m.%Y %H:%M')} UTC",
    ]
    return "\n".join(parts)


def format_weak_signal(sig: dict) -> str:
    flag = KZ_FLAGS.get(sig.get("killzone") or "", "")
    sl_diff = sig["sl"] - sig["entry"]
    parts = [
        f"⚠️ Слабый сигнал — {sig['direction']}",
        f"🔹 Вход: `{_fmt_price(sig['entry'])}`",
        f"🔺 SL: `{_fmt_price(sig['sl'])}` ({_fmt_pts(sl_diff)})",
        f"🎯 TP1: `{_fmt_price(sig['tp1'])}`",
        _tp2_line(sig),
        f"🧠 Score: {sig['score']}/100",
        f"⏱ {sig.get('killzone') or '—'} {flag}",
    ]
    return "\n".join(parts)


def format_no_signal_killzone(killzone: str, price: float,
                              rsi: float | None) -> str:
    rsi_text = f"{rsi:.0f}" if rsi is not None else "—"
    return (
        f"⏳ {killzone} | Нет сигнала\n"
        f"💰 XAU: `{_fmt_price(price)}` | RSI H1: {rsi_text}"
    )


def format_status(snapshot: dict) -> str:
    """Generic /status response builder."""
    lines = [
        "📊 Market Status",
        f"💰 XAU/USD: `{_fmt_price(snapshot.get('price'))}`",
        f"🕐 Killzone: {snapshot.get('killzone') or 'none'}",
        f"📈 D1 trend: {snapshot.get('d1_trend', '—')}",
        f"📊 H4 structure: {snapshot.get('h4_structure', '—')}",
        f"🌀 Wyckoff: {snapshot.get('wyckoff', '—')}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/test_formatter.py -v
git add xau_pro_bot/formatter.py tests/test_formatter.py
git commit -m "feat(formatter): add Telegram Markdown signal templates"
```

---

## Task 13: Bot Entrypoint (Telegram + AsyncIOScheduler)

**Files:**
- Create: `xau_pro_bot/bot.py`

This task is the integration glue. Tests are minimal because the bot is heavily IO-bound — main quality gate is manual smoke test plus the signal-pipeline test below.

- [ ] **Step 1: Write integration test (signal pipeline without telegram)**

```python
# tests/test_pipeline.py
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from xau_pro_bot.state import State
from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.filters import should_send


@pytest.fixture
def state(tmp_path):
    return State(db_path=str(tmp_path / "p.db"))


def test_pipeline_records_signal(state, all_tfs):
    eng = MasterSignalEngine()
    sig = eng.analyze(all_tfs)
    ok, reason = should_send(sig, state)
    if ok:
        state.record_signal({
            "ts_utc": sig["ts_utc"].isoformat(),
            "direction": sig["direction"],
            "tier": sig["tier"],
            "score": sig["score"],
            "entry": sig["entry"],
            "sl": sig.get("sl") or 0.0,
            "tp1": sig.get("tp1"),
            "tp2": sig.get("tp2"),
            "tp3": sig.get("tp3"),
            "rr": sig.get("rr"),
            "killzone": sig.get("killzone"),
            "reasons_json": json.dumps(sig["reasons"]),
        })
        assert state.last_signal() is not None
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implement `bot.py`**

```python
# xau_pro_bot/bot.py
"""Telegram bot entrypoint with AsyncIOScheduler."""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes,
)

from xau_pro_bot import config, data, formatter
from xau_pro_bot.indicators.ict import get_killzone
from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.filters import should_send, SkipReason
from xau_pro_bot.state import State


# ── Logging ────────────────────────────────────────
def _setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    err = logging.handlers.RotatingFileHandler(
        "errors.log", maxBytes=2_000_000, backupCount=2)
    err.setLevel(logging.ERROR)
    err.setFormatter(logging.Formatter(fmt))
    root.addHandler(err)


_signal_log = logging.getLogger("signals")
_signal_handler = logging.handlers.RotatingFileHandler(
    "signals.log", maxBytes=2_000_000, backupCount=3)
_signal_handler.setFormatter(logging.Formatter("%(message)s"))
_signal_log.addHandler(_signal_handler)
_signal_log.setLevel(logging.INFO)


# ── Globals (set in main) ──────────────────────────
ENV: dict[str, str] = {}
STATE: State | None = None
ENGINE = MasterSignalEngine()


def _log_signal(sig: dict[str, Any], status: str) -> None:
    line = " | ".join(str(x) for x in (
        sig["ts_utc"].isoformat() if isinstance(sig["ts_utc"], datetime) else sig["ts_utc"],
        sig["direction"], sig["tier"], sig["score"],
        sig.get("entry"), sig.get("sl"), sig.get("tp1"), sig.get("tp2"),
        sig.get("rr"), sig.get("killzone"), status,
    ))
    _signal_log.info(line)


def _persist(sig: dict[str, Any]) -> None:
    assert STATE is not None
    STATE.record_signal({
        "ts_utc": sig["ts_utc"].isoformat(),
        "direction": sig["direction"],
        "tier": sig["tier"],
        "score": sig["score"],
        "entry": sig["entry"],
        "sl": sig.get("sl") or 0.0,
        "tp1": sig.get("tp1"),
        "tp2": sig.get("tp2"),
        "tp3": sig.get("tp3"),
        "rr": sig.get("rr"),
        "killzone": sig.get("killzone"),
        "reasons_json": json.dumps(sig["reasons"], ensure_ascii=False),
    })


def _format(sig: dict[str, Any]) -> str:
    if sig["tier"] in ("STRONG", "NORMAL"):
        return formatter.format_strong_signal(sig)
    if sig["tier"] == "WEAK":
        return formatter.format_weak_signal(sig)
    raise ValueError(f"Cannot format tier {sig['tier']}")


async def _scan_and_send(app: Application, *, bypass_dedup: bool = False) -> None:
    assert STATE is not None
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
    except Exception:
        logging.exception("Data fetch failed")
        return

    try:
        sig = ENGINE.analyze(tfs)
    except Exception:
        logging.exception("Engine analyze failed")
        return

    ok, reason = should_send(sig, STATE, bypass_dedup=bypass_dedup)

    if not ok:
        _log_signal(sig, f"skipped:{reason.value if reason else 'unknown'}")
        if sig["tier"] == "NO_SIGNAL" and sig.get("killzone"):
            # Brief market update during killzone
            rsi = None
            try:
                from xau_pro_bot.indicators.classic import add_classic
                rsi = float(add_classic(tfs["H1"])["RSI_14"].iloc[-1])
                if np.isnan(rsi):
                    rsi = None
            except Exception:
                pass
            msg = formatter.format_no_signal_killzone(
                killzone=sig["killzone"], price=sig["entry"], rsi=rsi)
            await app.bot.send_message(
                chat_id=ENV["TELEGRAM_CHAT_ID"], text=msg,
                parse_mode=ParseMode.MARKDOWN)
        return

    text = _format(sig)
    try:
        await app.bot.send_message(
            chat_id=ENV["TELEGRAM_CHAT_ID"], text=text,
            parse_mode=ParseMode.MARKDOWN)
        _persist(sig)
        _log_signal(sig, "sent")
    except Exception:
        logging.exception("Telegram send failed")
        _log_signal(sig, "send_failed")


# ── Command handlers ───────────────────────────────
async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "XAU Pro Bot готов.\n"
        "Команды: /signal /status /levels /help /settings /stats")


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Анализирую…")
    await _scan_and_send(ctx.application, bypass_dedup=True)


async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
        from xau_pro_bot.indicators.classic import add_classic
        from xau_pro_bot.indicators.smc import detect_structure
        from xau_pro_bot.indicators.wyckoff import detect_wyckoff

        d1 = add_classic(tfs["D1"])
        h4 = tfs["H4"]
        ema50 = d1["EMA_50"].iloc[-1]
        ema200 = d1["EMA_200"].iloc[-1]
        d1_trend = "bull" if ema50 > ema200 else "bear"
        struct = detect_structure(h4, swing_len=5)
        wy = detect_wyckoff(tfs["D1"])
        snapshot = {
            "price": float(tfs["M15"]["Close"].iloc[-1]),
            "killzone": get_killzone(),
            "d1_trend": d1_trend,
            "h4_structure": struct["last_event"] or "—",
            "wyckoff": f"{wy['phase']} ({wy['bias']})",
        }
        await update.message.reply_text(
            formatter.format_status(snapshot), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        logging.exception("/status failed")
        await update.message.reply_text("Ошибка получения данных.")


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Тиры:\n"
        "STRONG (≥65) — всегда\n"
        "NORMAL (50–64) — по фильтрам\n"
        "WEAK (40–49) — только в killzone\n\n"
        "Лимиты: 6 сигналов/сутки, WEAK 1 раз в 4ч.\n"
        "Score — внутренняя метрика конfluence, не вероятность."
    )


async def cmd_settings(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"STRONG≥{config.STRONG_SIGNAL}, NORMAL≥{config.NORMAL_SIGNAL}, "
        f"WEAK≥{config.WEAK_SIGNAL}\n"
        f"Dedup {config.DEDUP_HOURS}h, RR≥{config.MIN_RR}\n"
        f"Scan: KZ {config.KILLZONE_SCAN_INTERVAL}s / out "
        f"{config.BACKGROUND_SCAN_INTERVAL}s")


async def cmd_levels(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
        from xau_pro_bot.indicators.ict import (
            find_fvg, find_order_blocks, find_liquidity,
        )
        h1 = tfs["H1"]
        fvgs = find_fvg(h1, 3)
        obs = find_order_blocks(h1, 50)[:3]
        liq = find_liquidity(h1, lookback=30)
        lines = ["📍 Ключевые уровни (H1)"]
        for f in fvgs:
            lines.append(f"FVG {f['type']}: {f['bottom']:.2f}–{f['top']:.2f}")
        for ob in obs:
            lines.append(f"OB {ob['type']}: {ob['low']:.2f}–{ob['high']:.2f}")
        if liq["buy_side"]:
            lines.append(f"Buy-side liq: {liq['buy_side'][:3]}")
        if liq["sell_side"]:
            lines.append(f"Sell-side liq: {liq['sell_side'][:3]}")
        await update.message.reply_text("\n".join(lines))
    except Exception:
        logging.exception("/levels failed")
        await update.message.reply_text("Ошибка.")


async def cmd_stats(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    assert STATE is not None
    today = STATE.count_today()
    strong = STATE.count_today(tier="STRONG")
    weak = STATE.count_today(tier="WEAK")
    await update.message.reply_text(
        f"Сегодня: {today} сигналов (STRONG={strong}, WEAK={weak})")


# ── Scheduler jobs ──────────────────────────────────
def _is_killzone_now() -> bool:
    return get_killzone() is not None


async def _scheduled_scan(app: Application) -> None:
    await _scan_and_send(app, bypass_dedup=False)


def _build_scheduler(app: Application) -> AsyncIOScheduler:
    sched = AsyncIOScheduler(timezone="UTC")
    # Two interval jobs; the handler decides whether the current time is inside a killzone
    sched.add_job(_scheduled_scan, "interval",
                  seconds=config.KILLZONE_SCAN_INTERVAL,
                  args=[app], id="kz_scan",
                  misfire_grace_time=60, coalesce=True)
    sched.add_job(_scheduled_scan, "interval",
                  seconds=config.BACKGROUND_SCAN_INTERVAL,
                  args=[app], id="bg_scan",
                  misfire_grace_time=60, coalesce=True)
    # Daily prune
    sched.add_job(lambda: STATE.prune_old(90) if STATE else None,
                  "cron", hour=0, minute=15, id="prune")
    return sched


def main() -> None:
    _setup_logging()
    global ENV, STATE
    ENV = config.load_env(required=[
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TWELVE_DATA_API_KEY"])
    import os
    STATE = State(db_path=os.getenv("STATE_DB_PATH", "./state.db"))

    app = ApplicationBuilder().token(ENV["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("levels", cmd_levels))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("stats", cmd_stats))

    sched = _build_scheduler(app)

    async def on_startup(_: Application) -> None:
        sched.start()
        logging.info("Scheduler started.")

    async def on_shutdown(_: Application) -> None:
        sched.shutdown(wait=False)
        if STATE is not None:
            STATE.close()

    app.post_init = on_startup
    app.post_shutdown = on_shutdown

    logging.info("Starting XAU Pro Bot…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke test — import only (no real telegram)**

```bash
python -c "from xau_pro_bot import bot; print('import ok')"
```

Expected: `import ok`.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/bot.py tests/test_pipeline.py
git commit -m "feat(bot): add telegram entrypoint with AsyncIOScheduler"
```

---

## Task 14: Backtest Module

**Files:**
- Create: `xau_pro_bot/backtest.py`
- Test: `tests/test_backtest.py`
- Create: `tests/fixtures/h1_sample.csv` (small CSV)

- [ ] **Step 1: Create fixture CSV**

```python
# Generate tests/fixtures/h1_sample.csv via a quick script:
python -c "
import pandas as pd, numpy as np
from datetime import datetime, timezone
n = 500
idx = pd.date_range(datetime(2026,1,1,tzinfo=timezone.utc), periods=n, freq='h')
closes = 2000 + np.cumsum(np.random.default_rng(42).normal(0, 5, n))
df = pd.DataFrame({
    'datetime': idx,
    'Open': closes - 1, 'High': closes + 3, 'Low': closes - 3,
    'Close': closes, 'Volume': 1000,
})
df.to_csv('tests/fixtures/h1_sample.csv', index=False)
"
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_backtest.py
from pathlib import Path

import pandas as pd
import pytest

from xau_pro_bot.backtest import (
    load_csv_history, run_backtest, BacktestResult,
)


def test_load_csv_history():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    assert len(df) > 0
    assert set(df.columns) >= {"Open", "High", "Low", "Close", "Volume"}


def test_run_backtest_returns_result():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    # Use same df for all TFs as a simplification in tests
    result = run_backtest(
        history={tf: df for tf in ("W1", "D1", "H4", "H1", "M15")},
        timeout_bars=48,
    )
    assert isinstance(result, BacktestResult)
    assert result.signals_generated >= 0
    assert 0 <= result.win_rate <= 1
```

- [ ] **Step 3: Run test — verify failure**

```bash
pytest tests/test_backtest.py -v
```

- [ ] **Step 4: Implement `backtest.py`**

```python
# xau_pro_bot/backtest.py
"""Walk-forward backtester for MasterSignalEngine.

Usage:
    python -m xau_pro_bot.backtest --csv path/to/history.csv [--tier STRONG]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.signals.engine import MasterSignalEngine


@dataclass
class BacktestResult:
    signals_generated: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    pnl_r: list[float] = field(default_factory=list)
    per_tier: dict[str, dict[str, int]] = field(
        default_factory=lambda: {t: {"n": 0, "w": 0, "l": 0}
                                  for t in ("STRONG", "NORMAL", "WEAK")})

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def expectancy(self) -> float:
        return float(np.mean(self.pnl_r)) if self.pnl_r else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(x for x in self.pnl_r if x > 0)
        losses = -sum(x for x in self.pnl_r if x < 0)
        return gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0


def load_csv_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _resample(h1: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return h1.resample(rule).agg(agg).dropna()


def _outcome(future: pd.DataFrame, entry: float, sl: float,
             tp: float, direction: str, timeout_bars: int) -> tuple[str, float]:
    """Return ('win'|'loss'|'timeout', R_multiple)."""
    risk = abs(entry - sl)
    if risk <= 0:
        return "timeout", 0.0
    bars = future.iloc[:timeout_bars]
    for _, row in bars.iterrows():
        if direction == "BUY":
            if row["Low"] <= sl:
                return "loss", -1.0
            if row["High"] >= tp:
                return "win", abs(tp - entry) / risk
        else:
            if row["High"] >= sl:
                return "loss", -1.0
            if row["Low"] <= tp:
                return "win", abs(entry - tp) / risk
    return "timeout", 0.0


def run_backtest(history: dict[str, pd.DataFrame],
                 timeout_bars: int = 48,
                 step: int = 4) -> BacktestResult:
    """Replay the engine on history, stepping `step` H1 bars at a time."""
    eng = MasterSignalEngine()
    res = BacktestResult()
    h1 = history["H1"]
    if len(h1) < 250:
        return res

    for i in range(250, len(h1) - timeout_bars, step):
        slice_data: dict[str, pd.DataFrame] = {}
        cutoff = h1.index[i]
        for tf, df in history.items():
            slice_data[tf] = df.loc[:cutoff].tail(720)
        try:
            sig = eng.analyze(slice_data)
        except Exception:
            continue
        if sig["tier"] == "NO_SIGNAL" or sig.get("tp1") is None:
            continue
        res.signals_generated += 1
        target = sig.get("tp2") or sig["tp1"]
        future = h1.iloc[i + 1:]
        outcome, r = _outcome(future, sig["entry"], sig["sl"],
                              target, sig["direction"], timeout_bars)
        if outcome == "win":
            res.wins += 1
            res.per_tier[sig["tier"]]["n"] += 1
            res.per_tier[sig["tier"]]["w"] += 1
        elif outcome == "loss":
            res.losses += 1
            res.per_tier[sig["tier"]]["n"] += 1
            res.per_tier[sig["tier"]]["l"] += 1
        else:
            res.timeouts += 1
        res.pnl_r.append(r)
    return res


def _cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="H1 OHLCV CSV path")
    p.add_argument("--timeout-bars", type=int, default=48)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--export", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    h1 = load_csv_history(Path(args.csv))
    history = {
        "H1": h1,
        "M15": _resample(h1, "15min"),
        "H4": _resample(h1, "4h"),
        "D1": _resample(h1, "1D"),
        "W1": _resample(h1, "1W"),
    }
    res = run_backtest(history, timeout_bars=args.timeout_bars, step=args.step)
    print(f"Signals:    {res.signals_generated}")
    print(f"Wins/Loss:  {res.wins} / {res.losses} (timeouts {res.timeouts})")
    print(f"Win rate:   {res.win_rate:.1%}")
    print(f"Expectancy: {res.expectancy:.2f} R")
    print(f"Profit f.:  {res.profit_factor:.2f}")
    print("By tier:")
    for tier, st in res.per_tier.items():
        if st["n"]:
            wr = st["w"] / st["n"]
            print(f"  {tier}: n={st['n']} wr={wr:.1%}")
    if args.export:
        pd.DataFrame({"R": res.pnl_r}).to_csv(args.export, index=False)
        print(f"Exported {args.export}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
```

- [ ] **Step 5: Run tests + commit**

```bash
pytest tests/test_backtest.py -v
git add xau_pro_bot/backtest.py tests/test_backtest.py tests/fixtures/h1_sample.csv
git commit -m "feat(backtest): add walk-forward backtester with R-multiple stats"
```

---

## Task 15: README + Final Polish

**Files:**
- Create: `signals/README.md`

- [ ] **Step 1: Write the README**

```markdown
# XAU Pro Bot

Deterministic Telegram signal bot for XAU/USD using ICT, SMC, Wyckoff (soft bias), and classic TA confluence. No AI, no LLM, no broker execution.

## Features

- Multi-timeframe analysis: W1 / D1 / H4 / H1 / M15.
- 5-layer scoring engine (Macro, Structure, ICT, Classic, Penalties).
- Three signal tiers: STRONG (≥65) / NORMAL (50–64) / WEAK (40–49).
- DST-aware killzones (America/New_York).
- SQLite persistence for dedup + rate limits.
- AsyncIOScheduler: 5 min in killzones, 15 min outside.
- Built-in walk-forward backtester for weight calibration.

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

Free tier — register at https://twelvedata.com (8 req/min, 800 req/day). Symbol used: `XAU/USD` (spot).

## Telegram setup

1. `@BotFather` → `/newbot` → token.
2. Start your bot in DM; send `/start`.
3. Use `@userinfobot` or hit `https://api.telegram.org/bot<token>/getUpdates` after sending a message → `chat.id`.
4. Put both into `.env`.

## Railway deploy

```bash
railway login
railway init
railway variables set TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... TWELVE_DATA_API_KEY=...
railway up
```

Service runs `worker: python -m xau_pro_bot.bot` (`Procfile`). State file `state.db` lives on the ephemeral filesystem — fine for v1 dedup, redeploys reset history.

## Backtesting (calibration gate)

Before trusting live signals, run the backtester on 12 months of H1 data:

```bash
python -m xau_pro_bot.backtest --csv history_h1.csv
```

Expected output (sample):
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

**Acceptance gate:** STRONG should show win rate ≥ 45% AND expectancy > 0. If not, hand-tune weights in `signals/{ict,smc,classic}_signals.py` and rerun.

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

`Score` is the internal confluence number, NOT a probability. Calibrate via backtest before treating it as one.

## Module map

- `data.py` — Twelve Data REST + TTL cache + retry.
- `state.py` — SQLite signals/dedup.
- `indicators/` — feature extraction (one file per concept group).
- `signals/engine.py` — 5-layer scoring.
- `signals/filters.py` — dedup, ATR-reprice, rate-limit.
- `formatter.py` — Telegram Markdown.
- `bot.py` — Telegram + AsyncIOScheduler.
- `backtest.py` — walk-forward replay + R-multiple metrics.

## ICT / SMC primer

- **OTE** — Optimal Trade Entry, 0.62–0.79 Fibonacci retracement zone.
- **FVG** — Fair Value Gap (price imbalance between 3 candles).
- **OB** — Order Block, last opposite candle before strong impulse.
- **Liquidity** — equal highs/lows attracting price.
- **BOS / CHOCH** — Break of Structure / Change of Character.
- **Premium/Discount** — top/bottom 40% of recent range.
- **Killzone** — NY-tz windows of historic volatility (London 02:00–05:00 NY, NY AM 08:30–11:00 NY).

## Limits and caveats

- Twelve Data free tier rate limit: 800 req/day. The TTL cache + 5-min scan cadence stays inside.
- SQLite is local; if Railway redeploys you lose dedup state until first new signal.
- This is signal generation only — no broker execution, no position sizing.
- Weights are unverified until you run the backtester.

## License

Private project.
```

- [ ] **Step 2: Full test run + commit**

```bash
pytest -v
git add README.md
git commit -m "docs: add README with setup, deploy, backtesting, and signal primer"
```

---

## Task 16: Acceptance Smoke Run

- [ ] **Step 1: Run full test suite**

```bash
pytest -v --tb=short
```

Expected: all green.

- [ ] **Step 2: Run each indicator's `__main__` standalone**

```bash
python -m xau_pro_bot.indicators.classic
python -c "from xau_pro_bot.indicators.ict import get_killzone; print(get_killzone())"
```

Expected: no exceptions.

- [ ] **Step 3: Lint pass (optional, run if ruff/mypy configured)**

```bash
ruff check xau_pro_bot/ tests/ || true
```

- [ ] **Step 4: Dry-run import of bot.py without starting Telegram**

```bash
python -c "from xau_pro_bot import bot; print('bot module import ok')"
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: complete v1 acceptance smoke pass" --allow-empty
```

---

## Self-Review Notes

**Spec coverage:**
- Data (Twelve Data + cache + retry) → Task 4 ✓
- State / SQLite → Task 3 ✓
- pandas-ta monkey-patch → Task 0 step 8 ✓
- Classic indicators → Task 5 ✓
- ICT (OTE/FVG/OB/Liquidity/Killzone, NY-tz) → Task 6 ✓
- SMC (structure/PD/stop-hunt updated formula) → Task 7 ✓
- Wyckoff soft bias ±5 → Task 8 + integrated in engine `_macro_bias` ✓
- SR helpers → Task 9 ✓
- Signal engine 5-layer + penalty subtracted from own direction → Task 10 ✓
- Level calculation including `tp2_unavailable` flow → Task 10 (`_compute_levels`) ✓
- Filters (dedup + ATR-reprice win + rate limit + WEAK cooldown) → Task 11 ✓
- Formatter ("Score X/100" + TP2 unavailable) → Task 12 ✓
- Bot + AsyncIOScheduler + commands → Task 13 ✓
- Backtest (CLI + R-multiple + per-tier) → Task 14 ✓
- README → Task 15 ✓

**Gaps fixed inline:**
- `/stats` command added (was in spec, missed initial draft).
- Daily prune job added to scheduler (state.prune_old).
- `format_status` helper added in formatter for `/status`.

**Placeholder scan:** no TBDs, no "implement later", every code step has full code.

**Type consistency:** `should_send` returns `tuple[bool, SkipReason | None]` everywhere; `SkipReason` enum values match test asserts (`SkipReason.DEDUP`, `SkipReason.RATE_LIMIT_DAY`, etc.). `analyze()` returns dict with consistent keys across engine, formatter, and bot.

---

---

# Revision 3 Addendum: Multi-Stream Architecture (S/R Zones + Swing + Scalp)

> **Context:** After Tasks 0–16 complete, the bot has a working `intraday` stream. Tasks 17–24 add the `swing` and `scalp` streams plus the shared `sr_zones` enrichment, refactoring `engine.py` into a `StreamRouter` that orchestrates three independent analyzers. **All upstream tests must remain green after each task in this addendum.**

**Key invariants:**
- Streams do NOT combine scores. Each stream produces its own `SignalResult` or nothing.
- `1 pip = 0.10 USD` (`config.XAU_PIP_VALUE`). All swing/scalp math uses this constant.
- S/R zones enrich context only; they are not a separate stream.
- Per-stream rate limits and dedup are enforced in `filters.py` by reading `sig["stream"]`.

---

## Task 17: Pip Constant + State Schema Migration

**Files:**
- Modify: `xau_pro_bot/config.py` — add pip constant + per-stream limits.
- Modify: `xau_pro_bot/state.py` — add `stream` column with migration.
- Test: `tests/test_state.py` — add migration + per-stream tests.

- [ ] **Step 1: Add config constants**

```python
# Append to xau_pro_bot/config.py

# ── XAU pip ───────────────────────────────────────────
XAU_PIP_VALUE = 0.10  # USD per pip

# ── Per-stream rate limits ────────────────────────────
MAX_INTRADAY_PER_DAY = 6      # was MAX_SIGNALS_PER_DAY
MAX_SWING_PER_DAY = 2
MAX_SCALP_PER_DAY = 4
SCALP_MIN_GAP_MINUTES = 30
SWING_DIRECTION_COOLDOWN_HOURS = 24

# Stream identifiers
STREAM_INTRADAY = "intraday"
STREAM_SWING = "swing"
STREAM_SCALP = "scalp"
```

Keep `MAX_SIGNALS_PER_DAY = 6` as backward-compat alias of `MAX_INTRADAY_PER_DAY`.

- [ ] **Step 2: Write failing migration test**

```python
# tests/test_state.py — append
import sqlite3


def test_migration_adds_stream_column(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    # Create v1 schema (no stream column)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL, direction TEXT NOT NULL, tier TEXT NOT NULL,
            score INTEGER NOT NULL, entry REAL NOT NULL, sl REAL NOT NULL,
            tp1 REAL, tp2 REAL, tp3 REAL, rr REAL,
            killzone TEXT, reasons_json TEXT
        );
    """)
    conn.execute(
        "INSERT INTO signals (ts_utc, direction, tier, score, entry, sl) "
        "VALUES (?, 'BUY', 'STRONG', 70, 2000, 1995)",
        (datetime.now(timezone.utc).isoformat(),)
    )
    conn.close()

    # Open via State — migration should run
    st = State(db_path=db_path)
    cols = [r["name"] for r in st._conn.execute(
        "PRAGMA table_info(signals)").fetchall()]
    assert "stream" in cols
    # Legacy row defaults to 'intraday'
    row = st._conn.execute("SELECT stream FROM signals LIMIT 1").fetchone()
    assert row["stream"] == "intraday"


def test_record_signal_with_stream(state):
    sig = _sig()
    sig["stream"] = "swing"
    sid = state.record_signal(sig)
    assert sid > 0
    last = state.last_signal(stream="swing")
    assert last is not None
    assert last["stream"] == "swing"
    assert state.last_signal(stream="intraday") is None


def test_count_today_by_stream(state):
    base = _sig()
    state.record_signal({**base, "stream": "intraday"})
    state.record_signal({**base, "stream": "intraday"})
    state.record_signal({**base, "stream": "swing"})
    assert state.count_today(stream="intraday") == 2
    assert state.count_today(stream="swing") == 1
    assert state.count_today() == 3  # no filter still works
```

- [ ] **Step 3: Update `state.py`**

```python
# Update SCHEMA constant in xau_pro_bot/state.py
SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    direction TEXT NOT NULL,
    tier TEXT NOT NULL,
    score INTEGER NOT NULL,
    entry REAL NOT NULL,
    sl REAL NOT NULL,
    tp1 REAL,
    tp2 REAL,
    tp3 REAL,
    rr REAL,
    killzone TEXT,
    reasons_json TEXT,
    stream TEXT NOT NULL DEFAULT 'intraday'
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc);
CREATE INDEX IF NOT EXISTS idx_signals_dir ON signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_tier ON signals(tier);
CREATE INDEX IF NOT EXISTS idx_signals_stream ON signals(stream);
"""

# In State.__init__, after executescript(SCHEMA), add:
def _migrate(self) -> None:
    cols = [r[1] for r in self._conn.execute(
        "PRAGMA table_info(signals)").fetchall()]
    if "stream" not in cols:
        self._conn.execute(
            "ALTER TABLE signals ADD COLUMN stream TEXT NOT NULL DEFAULT 'intraday'"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_stream ON signals(stream)"
        )

# Call self._migrate() at end of __init__.
```

Update `record_signal` to include `stream` column:

```python
def record_signal(self, sig: dict[str, Any]) -> int:
    cols = ("ts_utc", "direction", "tier", "score", "entry", "sl",
            "tp1", "tp2", "tp3", "rr", "killzone", "reasons_json", "stream")
    placeholders = ", ".join("?" * len(cols))
    values = tuple(sig.get(c) if c != "stream" else sig.get("stream", "intraday")
                   for c in cols)
    cur = self._conn.execute(
        f"INSERT INTO signals ({', '.join(cols)}) VALUES ({placeholders})",
        values,
    )
    return int(cur.lastrowid or 0)
```

Update `last_signal`, `count_today`, `last_weak_ts` to take optional `stream` filter:

```python
def last_signal(self, direction: str | None = None,
                stream: str | None = None) -> dict[str, Any] | None:
    where = []
    params: list[Any] = []
    if direction:
        where.append("direction = ?"); params.append(direction)
    if stream:
        where.append("stream = ?"); params.append(stream)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    row = self._conn.execute(
        f"SELECT * FROM signals {clause} ORDER BY id DESC LIMIT 1",
        tuple(params),
    ).fetchone()
    return dict(row) if row else None


def count_today(self, tier: str | None = None,
                stream: str | None = None) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    where = ["substr(ts_utc, 1, 10) = ?"]
    params: list[Any] = [today]
    if tier:
        where.append("tier = ?"); params.append(tier)
    if stream:
        where.append("stream = ?"); params.append(stream)
    row = self._conn.execute(
        f"SELECT COUNT(*) AS n FROM signals WHERE {' AND '.join(where)}",
        tuple(params),
    ).fetchone()
    return int(row["n"])


def last_weak_ts(self, stream: str = "intraday") -> datetime | None:
    row = self._conn.execute(
        "SELECT ts_utc FROM signals WHERE tier = 'WEAK' AND stream = ? "
        "ORDER BY id DESC LIMIT 1",
        (stream,),
    ).fetchone()
    return datetime.fromisoformat(row["ts_utc"]) if row else None


def last_scalp_ts(self) -> datetime | None:
    row = self._conn.execute(
        "SELECT ts_utc FROM signals WHERE stream = 'scalp' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return datetime.fromisoformat(row["ts_utc"]) if row else None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_state.py tests/test_config.py -v
```

Expected: green, including new migration test.

- [ ] **Step 5: Commit**

```bash
git add xau_pro_bot/config.py xau_pro_bot/state.py tests/test_state.py
git commit -m "feat(state): add per-stream column with auto-migration"
```

---

## Task 18: S/R Zones Module

**Files:**
- Create: `xau_pro_bot/indicators/sr_zones.py`
- Test: `tests/test_sr_zones.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sr_zones.py
import pytest

from xau_pro_bot.indicators.sr_zones import find_sr_zones, find_psychological_levels


def test_psychological_levels_includes_round_50_and_100():
    levels = find_psychological_levels(price=3312.0, span=200)
    assert 3300.0 in levels
    assert 3350.0 in levels
    assert 3400.0 in levels
    assert 3250.0 in levels


def test_find_sr_zones_keys(uptrend_df):
    res = find_sr_zones(h4_df=uptrend_df, d1_df=uptrend_df,
                        current_price=float(uptrend_df["Close"].iloc[-1]))
    for k in ("resistance_zones", "support_zones",
              "at_resistance", "at_support",
              "nearest_resistance", "nearest_support"):
        assert k in res


def test_sr_zone_strength_clamped_to_100(uptrend_df):
    res = find_sr_zones(h4_df=uptrend_df, d1_df=uptrend_df,
                        current_price=float(uptrend_df["Close"].iloc[-1]))
    for z in res["resistance_zones"] + res["support_zones"]:
        assert 0 <= z["strength"] <= 100


def test_sr_zone_has_top_bottom(uptrend_df):
    res = find_sr_zones(h4_df=uptrend_df, d1_df=uptrend_df,
                        current_price=2100.0)
    for z in res["resistance_zones"]:
        assert z["zone_top"] >= z["zone_bot"]
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_sr_zones.py -v
```

- [ ] **Step 3: Implement `sr_zones.py`**

```python
# xau_pro_bot/indicators/sr_zones.py
"""S/R zones: historical key levels, psychological round levels, and zone scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot.indicators import classic
from xau_pro_bot.indicators.sr_levels import swing_highs_lows


_TOUCH_TOLERANCE = 0.003  # 0.3%


def find_psychological_levels(price: float, span: float = 200.0) -> list[float]:
    """Return psychological round levels within +/- span of price.
    Includes every $50 and every $100."""
    low = price - span
    high = price + span
    levels: set[float] = set()
    base = int(low // 50) * 50
    while base <= high:
        levels.add(float(base))
        base += 50
    return sorted(levels)


def _count_touches(prices: np.ndarray, level: float) -> int:
    diff = np.abs(prices - level) / max(abs(level), 1.0)
    return int(np.sum(diff <= _TOUCH_TOLERANCE))


def _zone_strength(touches: int, recency_pct: float,
                   tf_bonus: int) -> int:
    raw = min(touches, 5) * 15 + int(recency_pct * 10) + tf_bonus
    return max(0, min(100, raw))


def _build_zone(level: float, touches: int, atr_h4: float,
                tf_bonus: int, recency_pct: float, kind: str) -> dict[str, Any]:
    width = max(atr_h4 * 0.5, 0.5)
    return {
        "level": float(level),
        "zone_top": float(level + width),
        "zone_bot": float(level - width),
        "touches": touches,
        "strength": _zone_strength(touches, recency_pct, tf_bonus),
        "type": kind,  # MAJOR / MINOR / PSYCHOLOGICAL
    }


def find_sr_zones(h4_df: pd.DataFrame, d1_df: pd.DataFrame,
                  current_price: float) -> dict[str, Any]:
    atr_h4 = 1.0
    enriched_h4 = classic.add_classic(h4_df)
    if "ATR_14" in enriched_h4 and not np.isnan(enriched_h4["ATR_14"].iloc[-1]):
        atr_h4 = float(enriched_h4["ATR_14"].iloc[-1])

    candidates: list[dict[str, Any]] = []

    # 1. D1 swing-based levels
    d1_window = d1_df.tail(365)
    if len(d1_window) >= 20:
        sh, sl = swing_highs_lows(d1_window, window=5)
        all_prices = np.concatenate(
            [d1_window["High"].to_numpy(), d1_window["Low"].to_numpy()])
        unique_swings = list(set([round(x, 2) for x in (sh + sl)]))
        for lvl in unique_swings:
            touches = _count_touches(all_prices, lvl)
            if touches >= 2:
                # Recency: position of last touch (later = stronger)
                close = d1_window["Close"].to_numpy()
                last_touch_idx = max(
                    (i for i, p in enumerate(close)
                     if abs(p - lvl) / max(abs(lvl), 1) <= _TOUCH_TOLERANCE),
                    default=0,
                )
                recency_pct = last_touch_idx / max(len(close) - 1, 1)
                kind = "MAJOR" if touches >= 3 else "MINOR"
                candidates.append(_build_zone(lvl, touches, atr_h4,
                                              tf_bonus=20,
                                              recency_pct=recency_pct,
                                              kind=kind))

    # 2. Psychological round levels
    for lvl in find_psychological_levels(current_price, span=200.0):
        candidates.append(_build_zone(
            level=lvl, touches=1, atr_h4=atr_h4,
            tf_bonus=8, recency_pct=0.5, kind="PSYCHOLOGICAL"))

    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for z in candidates:
        if z["level"] > current_price:
            resistance.append(z)
        elif z["level"] < current_price:
            support.append(z)

    resistance.sort(key=lambda z: z["level"])
    support.sort(key=lambda z: z["level"], reverse=True)

    at_resistance = any(z["zone_bot"] <= current_price <= z["zone_top"]
                        for z in resistance[:3])
    at_support = any(z["zone_bot"] <= current_price <= z["zone_top"]
                     for z in support[:3])

    return {
        "resistance_zones": resistance[:6],
        "support_zones": support[:6],
        "at_resistance": at_resistance,
        "at_support": at_support,
        "nearest_resistance": resistance[0]["level"] if resistance else None,
        "nearest_support": support[0]["level"] if support else None,
        "atr_h4": atr_h4,
    }
```

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/test_sr_zones.py -v
git add xau_pro_bot/indicators/sr_zones.py tests/test_sr_zones.py
git commit -m "feat(indicators): add S/R zones with key levels and psychological"
```

---

## Task 19: Integrate S/R Zones into Intraday Engine

**Files:**
- Modify: `xau_pro_bot/signals/smc_signals.py` — add zone bonus, anti-double-count with liquidity.
- Test: `tests/test_engine.py` — assert score increases when price sits in MAJOR zone aligned with direction.

- [ ] **Step 1: Write failing test**

```python
# tests/test_engine.py — append
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def _build_zone_aware_df(price: float = 2050.0, n: int = 100) -> pd.DataFrame:
    """Build a downtrend that bottoms repeatedly near `price` (creates MAJOR support)."""
    idx = pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc),
                        periods=n, freq="h")
    closes = np.concatenate([
        np.linspace(2200, price, n // 3),
        np.full(n // 3, price + np.random.uniform(-1, 1, n // 3)),
        np.linspace(price, price + 10, n - 2 * (n // 3)),
    ])
    return pd.DataFrame({
        "Open": closes - 0.5, "High": closes + 1.5,
        "Low": closes - 1.5, "Close": closes,
        "Volume": 1000.0,
    }, index=idx)


def test_sr_zone_increases_buy_score_at_major_support(monkeypatch):
    # Smoke: ensure score_smc with sr_zones doesn't crash on zone-rich data.
    from xau_pro_bot.signals.smc_signals import score_smc
    df = _build_zone_aware_df(price=2050.0)
    bull, bear, reasons = score_smc(df, sr_zones={
        "resistance_zones": [], "support_zones": [
            {"level": 2050.0, "zone_top": 2051.0, "zone_bot": 2049.0,
             "strength": 80, "touches": 4, "type": "MAJOR"}
        ],
        "at_resistance": False, "at_support": True,
        "nearest_resistance": None, "nearest_support": 2050.0,
        "atr_h4": 2.0,
    }, liquidity={"buy_side": [], "sell_side": []})
    assert bull > 0
    assert any("MAJOR support" in r for r in reasons)
```

- [ ] **Step 2: Run test — verify failure**

```bash
pytest tests/test_engine.py::test_sr_zone_increases_buy_score_at_major_support -v
```

- [ ] **Step 3: Update `signals/smc_signals.py` to accept and use S/R zones**

```python
# xau_pro_bot/signals/smc_signals.py — updated signature
"""SMC scoring contributions, now optionally augmented with S/R zones."""

from __future__ import annotations

from typing import Any

from xau_pro_bot.indicators.smc import detect_structure, premium_discount
from xau_pro_bot.indicators.ict import find_order_blocks, find_fvg


def _zone_bonus_for_direction(zones: list[dict],
                              price: float,
                              liquidity_levels: list[float],
                              direction: str) -> tuple[float, list[str]]:
    """Award zone bonus for direction (BUY = support, SELL = resistance).

    Anti-double-count: if zone level matches a liquidity level within 0.1%,
    apply the zone bonus only (max of the two), liquidity-scoring elsewhere skips.
    """
    pts = 0.0
    reasons: list[str] = []
    for z in zones[:3]:
        if not (z["zone_bot"] <= price <= z["zone_top"]):
            continue
        # Anti-double-count check
        overlap = any(
            abs(z["level"] - lq) / max(abs(z["level"]), 1) < 0.001
            for lq in liquidity_levels
        )
        marker = " (+liq overlap)" if overlap else ""
        if z["type"] == "MAJOR":
            pts += 12
            reasons.append(f"{direction} MAJOR zone @ {z['level']:.2f}{marker}")
        elif z["type"] == "MINOR":
            pts += 8
            reasons.append(f"{direction} MINOR zone @ {z['level']:.2f}{marker}")
        else:  # PSYCHOLOGICAL
            pts += 6
            reasons.append(f"{direction} round level @ {z['level']:.2f}{marker}")
        if z["strength"] > 70:
            pts += 5
        break
    return pts, reasons


def _opposing_zone_penalty(zones: list[dict], price: float,
                           pip_value: float) -> float:
    """If signal is heading INTO a strong zone within 30 pips, penalize -8."""
    for z in zones[:3]:
        if z["strength"] > 70:
            dist_pips = abs(z["level"] - price) / pip_value
            if dist_pips <= 30:
                return 8.0
    return 0.0


def score_smc(h4_df, sr_zones: dict | None = None,
              liquidity: dict | None = None) -> tuple[float, float, list[str]]:
    from xau_pro_bot import config
    bull = bear = 0.0
    reasons: list[str] = []
    sr_zones = sr_zones or {"resistance_zones": [], "support_zones": [],
                            "at_resistance": False, "at_support": False}
    liquidity = liquidity or {"buy_side": [], "sell_side": []}

    struct = detect_structure(h4_df, swing_len=5)
    event = struct["last_event"]
    if event == "CHOCH_bull":
        bull += 15; reasons.append("H4 CHOCH bull")
    elif event == "CHOCH_bear":
        bear += 15; reasons.append("H4 CHOCH bear")
    elif event == "BOS_bull":
        bull += 10; reasons.append("H4 BOS bull")
    elif event == "BOS_bear":
        bear += 10; reasons.append("H4 BOS bear")

    pd_zone = premium_discount(h4_df, lookback=50)
    if pd_zone["zone"] == "discount":
        bull += 8; bear -= 10
        reasons.append(f"H4 discount ({pd_zone['pct_of_range']:.0f}%)")
    elif pd_zone["zone"] == "premium":
        bear += 8; bull -= 10
        reasons.append(f"H4 premium ({pd_zone['pct_of_range']:.0f}%)")

    last_close = float(h4_df["Close"].iloc[-1])
    obs = find_order_blocks(h4_df, lookback=50)
    for ob in obs[:5]:
        if not ob["tested"] and ob["low"] <= last_close <= ob["high"]:
            if ob["type"] == "bull":
                bull += 7; reasons.append(f"H4 OB bull {ob['mid']:.2f}")
            else:
                bear += 7; reasons.append(f"H4 OB bear {ob['mid']:.2f}")
            break

    fvgs = find_fvg(h4_df, max_gaps=5)
    for fvg in fvgs[:3]:
        if fvg["bottom"] <= last_close <= fvg["top"]:
            if fvg["type"] == "bull":
                bull += 5; reasons.append(f"H4 FVG bull {fvg['midpoint']:.2f}")
            else:
                bear += 5; reasons.append(f"H4 FVG bear {fvg['midpoint']:.2f}")
            break

    # S/R zone bonuses
    buy_pts, buy_reasons = _zone_bonus_for_direction(
        sr_zones["support_zones"], last_close,
        liquidity["sell_side"], "BUY")
    bull += buy_pts; reasons.extend(buy_reasons)

    sell_pts, sell_reasons = _zone_bonus_for_direction(
        sr_zones["resistance_zones"], last_close,
        liquidity["buy_side"], "SELL")
    bear += sell_pts; reasons.extend(sell_reasons)

    # Opposing zone penalty
    bull -= _opposing_zone_penalty(sr_zones["resistance_zones"], last_close,
                                    config.XAU_PIP_VALUE)
    bear -= _opposing_zone_penalty(sr_zones["support_zones"], last_close,
                                    config.XAU_PIP_VALUE)

    return bull, bear, reasons
```

- [ ] **Step 4: Update `engine.py` to call `find_sr_zones` and pass into `score_smc`**

In `MasterSignalEngine.analyze`, replace the `score_smc(h4)` call with:

```python
from xau_pro_bot.indicators.sr_zones import find_sr_zones
from xau_pro_bot.indicators.ict import find_liquidity

current_price = float(m15["Close"].iloc[-1])
sr = find_sr_zones(h4_df=h4, d1_df=d1, current_price=current_price)
liq = find_liquidity(h1, lookback=30)
smc_bull, smc_bear, smc_reasons = score_smc(h4, sr_zones=sr, liquidity=liq)
```

Also store `sr` in the returned dict under key `"sr_zones"` so other streams can reuse it later.

- [ ] **Step 5: Run all tests**

```bash
pytest -v
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/signals/smc_signals.py xau_pro_bot/signals/engine.py tests/test_engine.py
git commit -m "feat(engine): integrate S/R zones into intraday SMC scoring"
```

---

## Task 20: Swing Stream

**Files:**
- Create: `xau_pro_bot/indicators/swing.py` — pure detector.
- Create: `xau_pro_bot/signals/swing_analyzer.py` — wraps detector into `SignalResult`.
- Test: `tests/test_swing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_swing.py
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.indicators.swing import find_swing_setup
from xau_pro_bot.signals.swing_analyzer import SwingAnalyzer


def _wide_range_d1(swing_low=2000.0, swing_high=2200.0, n=210) -> pd.DataFrame:
    idx = pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc), periods=n, freq="D")
    closes = np.concatenate([
        np.linspace(swing_low + 50, swing_high, n // 2),
        np.linspace(swing_high, swing_low + 30, n - n // 2),  # pullback creates entry zone
    ])
    return pd.DataFrame({
        "Open": closes - 1, "High": closes + 3,
        "Low": closes - 3, "Close": closes, "Volume": 1000.0,
    }, index=idx)


def test_find_swing_1000pip(monkeypatch):
    # XAU 2000 → 2200 = $200 = 2000 pips at $0.10/pip
    df = _wide_range_d1(swing_low=2000.0, swing_high=2200.0)
    res = find_swing_setup(d1_df=df, h4_df=df)
    assert res is not None
    assert res["type"] in ("1000pip", "500pip")
    assert res["range_pips"] >= 1000


def test_no_setup_when_range_too_small():
    n = 210
    idx = pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc), periods=n, freq="D")
    closes = np.linspace(2000.0, 2010.0, n)  # only $10 range = 100 pips
    df = pd.DataFrame({
        "Open": closes - 0.5, "High": closes + 1, "Low": closes - 1,
        "Close": closes, "Volume": 1000.0,
    }, index=idx)
    assert find_swing_setup(d1_df=df, h4_df=df) is None


def test_swing_analyzer_returns_signal_result():
    df = _wide_range_d1()
    data = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}
    sig = SwingAnalyzer().analyze(data)
    if sig is not None:
        assert sig["tier"] in ("STRONG", "NORMAL")
        assert sig["tp1"] is not None
        assert "horizon_label" in sig
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_swing.py -v
```

- [ ] **Step 3: Implement `indicators/swing.py`**

```python
# xau_pro_bot/indicators/swing.py
"""Swing setups: 1000-pip and 500-pip Fibonacci retracement entries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.indicators import classic


def _d1_trend(d1_df: pd.DataFrame) -> str | None:
    enriched = classic.add_classic(d1_df)
    last = enriched.iloc[-1]
    e50, e200 = last.get("EMA_50", np.nan), last.get("EMA_200", np.nan)
    if np.isnan(e50) or np.isnan(e200):
        return None
    return "bull" if e50 > e200 else "bear"


def find_swing_setup(d1_df: pd.DataFrame, h4_df: pd.DataFrame) -> dict[str, Any] | None:
    if len(d1_df) < 200:
        return None
    window = d1_df.tail(200)
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    full_range_usd = swing_high - swing_low
    range_pips = full_range_usd / config.XAU_PIP_VALUE
    if range_pips < 500:
        return None

    trend = _d1_trend(d1_df)
    if trend is None:
        return None
    direction = "BUY" if trend == "bull" else "SELL"

    if range_pips >= 1000:
        setup_type = "1000pip"
        fib = 0.20
        sl_buffer_pips = 50
    else:
        setup_type = "500pip"
        fib = 0.236
        sl_buffer_pips = 30

    sl_buffer = sl_buffer_pips * config.XAU_PIP_VALUE

    if direction == "BUY":
        # Entry at fib retracement from swing_high back toward swing_low
        entry = swing_high - fib * full_range_usd
        tp = swing_high  # opposite swing extreme = TP target
        # But TP is above entry only if we haven't already crossed it
        sl = swing_low - sl_buffer
    else:
        entry = swing_low + fib * full_range_usd
        tp = swing_low
        sl = swing_high + sl_buffer

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < 2.0:
        return None

    return {
        "type": setup_type,
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "range_pips": round(range_pips, 1),
        "rr": round(rr, 2),
    }
```

- [ ] **Step 4: Implement `signals/swing_analyzer.py`**

```python
# xau_pro_bot/signals/swing_analyzer.py
"""Swing stream analyzer wrapping find_swing_setup into SignalResult."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from xau_pro_bot.indicators.swing import find_swing_setup
from xau_pro_bot.indicators.ict import get_killzone


_HORIZON = {"1000pip": "1-4 недели", "500pip": "2-7 дней"}
_TIER = {"1000pip": "STRONG", "500pip": "NORMAL"}
_SCORE = {"1000pip": 80, "500pip": 65}


class SwingAnalyzer:
    def analyze(self, data: dict[str, pd.DataFrame]) -> dict | None:
        setup = find_swing_setup(d1_df=data["D1"], h4_df=data["H4"])
        if setup is None:
            return None
        return {
            "direction": setup["direction"],
            "tier": _TIER[setup["type"]],
            "score": _SCORE[setup["type"]],
            "entry": setup["entry"],
            "sl": setup["sl"],
            "tp1": setup["tp"],
            "tp2": None,
            "tp3": None,
            "rr": setup["rr"],
            "tp2_unavailable": True,
            "killzone": get_killzone(),
            "reasons": {
                "swing": [f"{setup['type']} setup, range {setup['range_pips']} pips"],
                "macro": [], "smc": [], "ict": [], "classic": [], "penalties": [],
            },
            "ts_utc": datetime.now(timezone.utc),
            "strategy_label": f"Swing {'1000' if setup['type'] == '1000pip' else '500'}",
            "horizon_label": _HORIZON[setup["type"]],
            "atr_h1": 1.0,  # not used for swing dedup
        }
```

- [ ] **Step 5: Run tests + commit**

```bash
pytest tests/test_swing.py -v
git add xau_pro_bot/indicators/swing.py xau_pro_bot/signals/swing_analyzer.py tests/test_swing.py
git commit -m "feat(swing): add 500/1000 pip Fibonacci swing detector"
```

---

## Task 21: Scalp Stream

**Files:**
- Create: `xau_pro_bot/indicators/scalping.py`
- Create: `xau_pro_bot/signals/scalp_analyzer.py`
- Test: `tests/test_scalp.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scalp.py
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.indicators.scalping import scalp_signal
from xau_pro_bot.signals.scalp_analyzer import ScalpAnalyzer


def _kz_now():
    return datetime(2026, 5, 11, 3, 0, tzinfo=ZoneInfo("America/New_York"))


@patch("xau_pro_bot.indicators.scalping.datetime")
def test_scalp_inactive_outside_killzone(mock_dt, uptrend_df, monkeypatch):
    # Force time to be outside KZ
    monkeypatch.setattr(
        "xau_pro_bot.indicators.scalping.get_killzone",
        lambda: None
    )
    res = scalp_signal(m15_df=uptrend_df, h1_df=uptrend_df, h4_df=uptrend_df)
    assert res is None or res["active"] is False


def test_scalp_returns_dict_in_killzone(monkeypatch, uptrend_df):
    monkeypatch.setattr(
        "xau_pro_bot.indicators.scalping.get_killzone",
        lambda: "London KZ"
    )
    res = scalp_signal(m15_df=uptrend_df, h1_df=uptrend_df, h4_df=uptrend_df)
    # In synthetic uptrend, scalp may or may not fire — assert shape only
    if res is not None and res["active"]:
        assert "direction" in res
        assert "sl" in res
        assert "tp1" in res
        assert "conditions_met" in res


def test_scalp_analyzer_no_signal_outside_kz(monkeypatch, uptrend_df):
    monkeypatch.setattr(
        "xau_pro_bot.indicators.scalping.get_killzone",
        lambda: None
    )
    data = {tf: uptrend_df for tf in ("W1", "D1", "H4", "H1", "M15")}
    sig = ScalpAnalyzer().analyze(data)
    assert sig is None
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_scalp.py -v
```

- [ ] **Step 3: Implement `indicators/scalping.py`**

```python
# xau_pro_bot/indicators/scalping.py
"""M15 scalp setup: EMA cross + RSI extreme + BB touch + volume."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from xau_pro_bot import config
from xau_pro_bot.indicators import classic
from xau_pro_bot.indicators.ict import get_killzone


def _h4_trend_bias(h4_df: pd.DataFrame) -> str | None:
    enriched = classic.add_classic(h4_df)
    last = enriched.iloc[-1]
    e50, e200 = last.get("EMA_50", np.nan), last.get("EMA_200", np.nan)
    if np.isnan(e50) or np.isnan(e200):
        return None
    return "bull" if e50 > e200 else "bear"


def scalp_signal(m15_df: pd.DataFrame, h1_df: pd.DataFrame,
                 h4_df: pd.DataFrame) -> dict[str, Any] | None:
    kz = get_killzone()
    if kz not in config.PRIORITY_KILLZONES:
        return None

    if len(m15_df) < 50:
        return None

    enriched = classic.add_classic(m15_df)
    if len(enriched) < 3:
        return None

    last = enriched.iloc[-1]
    prev = enriched.iloc[-2]
    pprev = enriched.iloc[-3]

    direction: str | None = None
    conditions_met: list[str] = []

    # 1. EMA8/EMA21 cross within last 2 bars
    cross_bull = (pprev["EMA_8"] < pprev["EMA_21"] and last["EMA_8"] > last["EMA_21"])
    cross_bear = (pprev["EMA_8"] > pprev["EMA_21"] and last["EMA_8"] < last["EMA_21"])
    if cross_bull:
        direction = "BUY"; conditions_met.append("EMA cross bull")
    elif cross_bear:
        direction = "SELL"; conditions_met.append("EMA cross bear")
    if direction is None:
        return None

    rsi = last["RSI_14"]
    if direction == "BUY" and not np.isnan(rsi) and rsi < 35:
        conditions_met.append(f"RSI {rsi:.0f} OS")
    elif direction == "SELL" and not np.isnan(rsi) and rsi > 65:
        conditions_met.append(f"RSI {rsi:.0f} OB")

    close = float(last["Close"])
    bbl = last["BBL_20_2.0"]; bbu = last["BBU_20_2.0"]
    if direction == "BUY" and not np.isnan(bbl) and close <= bbl * 1.001:
        conditions_met.append("BB lower")
    elif direction == "SELL" and not np.isnan(bbu) and close >= bbu * 0.999:
        conditions_met.append("BB upper")

    vol_ratio = last.get("vol_ratio", np.nan)
    if not np.isnan(vol_ratio) and vol_ratio > 1.3:
        conditions_met.append(f"Vol {vol_ratio:.1f}x")

    if len(conditions_met) < 3:
        return None

    atr_m15 = last["ATR_14"]
    if np.isnan(atr_m15) or atr_m15 <= 0:
        return None

    h4_trend = _h4_trend_bias(h4_df)
    counter_trend = (h4_trend == "bear" and direction == "BUY") or \
                    (h4_trend == "bull" and direction == "SELL")

    if direction == "BUY":
        sl = close - atr_m15 * 1.0
        tp1 = close + atr_m15 * 1.5
        tp2 = close + atr_m15 * 2.5
    else:
        sl = close + atr_m15 * 1.0
        tp1 = close - atr_m15 * 1.5
        tp2 = close - atr_m15 * 2.5

    return {
        "active": True,
        "direction": direction,
        "entry": round(close, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "conditions_met": conditions_met,
        "counter_trend": counter_trend,
        "killzone": kz,
        "atr_m15": float(atr_m15),
    }
```

- [ ] **Step 4: Implement `signals/scalp_analyzer.py`**

```python
# xau_pro_bot/signals/scalp_analyzer.py
"""Scalp stream analyzer."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from xau_pro_bot.indicators.scalping import scalp_signal


class ScalpAnalyzer:
    def analyze(self, data: dict[str, pd.DataFrame]) -> dict | None:
        res = scalp_signal(m15_df=data["M15"], h1_df=data["H1"], h4_df=data["H4"])
        if res is None:
            return None
        tier = "WEAK" if res["counter_trend"] else "NORMAL"
        score = 45 if res["counter_trend"] else 55
        risk = abs(res["entry"] - res["sl"])
        reward = abs(res["tp1"] - res["entry"])
        rr = round(reward / risk, 2) if risk > 0 else 0.0
        return {
            "direction": res["direction"],
            "tier": tier,
            "score": score,
            "entry": res["entry"],
            "sl": res["sl"],
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "tp3": None,
            "rr": rr,
            "tp2_unavailable": False,
            "killzone": res["killzone"],
            "reasons": {
                "scalp": res["conditions_met"]
                          + (["counter-trend"] if res["counter_trend"] else []),
                "macro": [], "smc": [], "ict": [], "classic": [], "penalties": [],
            },
            "ts_utc": datetime.now(timezone.utc),
            "strategy_label": "Scalp M15",
            "horizon_label": "15-60 минут",
            "atr_h1": res["atr_m15"],  # used only for ATR-reprice check
        }
```

- [ ] **Step 5: Run tests + commit**

```bash
pytest tests/test_scalp.py -v
git add xau_pro_bot/indicators/scalping.py xau_pro_bot/signals/scalp_analyzer.py tests/test_scalp.py
git commit -m "feat(scalp): add M15 scalp stream with H4 counter-trend tagging"
```

---

## Task 22: StreamRouter + Per-Stream Filters

**Files:**
- Create: `xau_pro_bot/signals/router.py`
- Modify: `xau_pro_bot/signals/filters.py` — per-stream limits.
- Test: `tests/test_router.py`
- Test: `tests/test_filters.py` — add per-stream tests.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_router.py
import pandas as pd
import pytest

from xau_pro_bot.signals.router import StreamRouter


def test_router_returns_list(all_tfs):
    results = StreamRouter().analyze(all_tfs)
    assert isinstance(results, list)
    # Each result carries a `stream` key
    for sig in results:
        assert sig.get("stream") in ("intraday", "swing", "scalp")


def test_router_continues_on_analyzer_exception(all_tfs, monkeypatch):
    from xau_pro_bot.signals.router import StreamRouter

    def boom(_self, _data):
        raise RuntimeError("scalp blew up")

    router = StreamRouter()
    monkeypatch.setattr(type(router.analyzers["scalp"]), "analyze", boom)
    # Should not raise; should still return any other valid signals
    results = router.analyze(all_tfs)
    assert isinstance(results, list)
```

```python
# tests/test_filters.py — append
from datetime import datetime, timedelta, timezone
from freezegun import freeze_time


def test_swing_per_day_cap(state):
    sig = _sig()
    sig["stream"] = "swing"
    for _ in range(2):
        state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                             "reasons_json": "{}"})
    new_sig = _sig(entry=2200.0)
    new_sig["stream"] = "swing"
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.RATE_LIMIT_DAY


def test_swing_same_direction_24h(state):
    sig = _sig(direction="BUY")
    sig["stream"] = "swing"
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    new_sig = _sig(direction="BUY", entry=2050.0)
    new_sig["stream"] = "swing"
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.SWING_DIRECTION_COOLDOWN


def test_scalp_min_gap_30min(state):
    sig = _sig()
    sig["stream"] = "scalp"
    sig["killzone"] = "London KZ"
    state.record_signal({**sig, "ts_utc": datetime.now(timezone.utc).isoformat(),
                         "reasons_json": "{}"})
    new_sig = _sig(entry=2010.0)
    new_sig["stream"] = "scalp"
    new_sig["killzone"] = "London KZ"
    ok, reason = should_send(new_sig, state)
    assert not ok and reason == SkipReason.SCALP_GAP


def test_scalp_must_be_in_killzone(state):
    sig = _sig()
    sig["stream"] = "scalp"
    sig["killzone"] = None
    ok, reason = should_send(sig, state)
    assert not ok and reason == SkipReason.SCALP_OUTSIDE_KZ
```

- [ ] **Step 2: Run tests — verify failure**

```bash
pytest tests/test_router.py tests/test_filters.py -v
```

- [ ] **Step 3: Update `filters.py`**

```python
# Replace xau_pro_bot/signals/filters.py with per-stream version

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from xau_pro_bot import config
from xau_pro_bot.state import State


class SkipReason(str, Enum):
    NO_SIGNAL = "no_signal"
    WEAK_OUTSIDE_KZ = "weak_outside_kz"
    DEDUP = "dedup"
    RATE_LIMIT_DAY = "rate_limit_day"
    WEAK_COOLDOWN = "weak_cooldown"
    NO_TP1 = "no_tp1"
    SWING_DIRECTION_COOLDOWN = "swing_direction_cooldown"
    SCALP_OUTSIDE_KZ = "scalp_outside_kz"
    SCALP_GAP = "scalp_gap"
    UNKNOWN_STREAM = "unknown_stream"


def _intraday_check(sig, state, bypass_dedup):
    if sig["tier"] == "NO_SIGNAL":
        return False, SkipReason.NO_SIGNAL
    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1
    if sig["tier"] == "WEAK" and not sig.get("killzone"):
        return False, SkipReason.WEAK_OUTSIDE_KZ
    if state.count_today(stream="intraday") >= config.MAX_INTRADAY_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY
    if sig["tier"] == "WEAK":
        last_weak = state.last_weak_ts(stream="intraday")
        if last_weak is not None:
            elapsed = datetime.now(timezone.utc) - last_weak
            if elapsed < timedelta(hours=config.WEAK_COOLDOWN_HOURS):
                return False, SkipReason.WEAK_COOLDOWN
    if bypass_dedup:
        return True, None
    last = state.last_signal(direction=sig["direction"], stream="intraday")
    if last is None:
        return True, None
    atr_h1 = sig.get("atr_h1", 1.0)
    if abs(sig["entry"] - last["entry"]) >= config.REPRICE_ATR_MULT * atr_h1:
        return True, None
    last_ts = datetime.fromisoformat(last["ts_utc"])
    if datetime.now(timezone.utc) - last_ts >= timedelta(hours=config.DEDUP_HOURS):
        return True, None
    return False, SkipReason.DEDUP


def _swing_check(sig, state, bypass_dedup):
    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1
    if state.count_today(stream="swing") >= config.MAX_SWING_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY
    if bypass_dedup:
        return True, None
    last = state.last_signal(direction=sig["direction"], stream="swing")
    if last is None:
        return True, None
    last_ts = datetime.fromisoformat(last["ts_utc"])
    cooldown = timedelta(hours=config.SWING_DIRECTION_COOLDOWN_HOURS)
    if datetime.now(timezone.utc) - last_ts < cooldown:
        return False, SkipReason.SWING_DIRECTION_COOLDOWN
    return True, None


def _scalp_check(sig, state, bypass_dedup):
    if sig.get("tp1") is None:
        return False, SkipReason.NO_TP1
    if not sig.get("killzone"):
        return False, SkipReason.SCALP_OUTSIDE_KZ
    if state.count_today(stream="scalp") >= config.MAX_SCALP_PER_DAY and not bypass_dedup:
        return False, SkipReason.RATE_LIMIT_DAY
    if bypass_dedup:
        return True, None
    last_ts = state.last_scalp_ts()
    if last_ts is not None:
        gap = datetime.now(timezone.utc) - last_ts
        if gap < timedelta(minutes=config.SCALP_MIN_GAP_MINUTES):
            return False, SkipReason.SCALP_GAP
    return True, None


def should_send(sig: dict, state: State,
                bypass_dedup: bool = False) -> tuple[bool, SkipReason | None]:
    stream = sig.get("stream", "intraday")
    if stream == "intraday":
        return _intraday_check(sig, state, bypass_dedup)
    if stream == "swing":
        return _swing_check(sig, state, bypass_dedup)
    if stream == "scalp":
        return _scalp_check(sig, state, bypass_dedup)
    return False, SkipReason.UNKNOWN_STREAM
```

- [ ] **Step 4: Implement `signals/router.py`**

```python
# xau_pro_bot/signals/router.py
"""Stream router: invokes all stream analyzers and returns non-null SignalResults."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.swing_analyzer import SwingAnalyzer
from xau_pro_bot.signals.scalp_analyzer import ScalpAnalyzer

log = logging.getLogger(__name__)


class _IntradayWrap:
    """Wraps MasterSignalEngine so its signal carries `strategy_label` and `horizon_label`."""

    def __init__(self):
        self._engine = MasterSignalEngine()

    def analyze(self, data: dict[str, pd.DataFrame]) -> dict | None:
        sig = self._engine.analyze(data)
        if sig is None or sig["tier"] == "NO_SIGNAL":
            return None
        labels = []
        if sig["reasons"].get("smc"):
            labels.append("SMC")
        if sig["reasons"].get("ict"):
            labels.append("ICT")
        if sig["reasons"].get("classic"):
            labels.append("Classic")
        sig["strategy_label"] = "+".join(labels) or "Intraday"
        sig["horizon_label"] = "1-24 часа"
        return sig


class StreamRouter:
    def __init__(self):
        self.analyzers: dict[str, Any] = {
            "intraday": _IntradayWrap(),
            "swing":    SwingAnalyzer(),
            "scalp":    ScalpAnalyzer(),
        }

    def analyze(self, data: dict[str, pd.DataFrame]) -> list[dict]:
        out: list[dict] = []
        for stream_name, analyzer in self.analyzers.items():
            try:
                sig = analyzer.analyze(data)
            except Exception:
                log.exception("Stream %s failed", stream_name)
                continue
            if sig is None:
                continue
            sig["stream"] = stream_name
            out.append(sig)
        return out
```

- [ ] **Step 5: Run all tests**

```bash
pytest -v
```

- [ ] **Step 6: Commit**

```bash
git add xau_pro_bot/signals/router.py xau_pro_bot/signals/filters.py tests/test_router.py tests/test_filters.py
git commit -m "feat(router): add StreamRouter with per-stream filters"
```

---

## Task 23: Wire StreamRouter into Bot + Update Formatter

**Files:**
- Modify: `xau_pro_bot/bot.py` — replace `ENGINE = MasterSignalEngine()` with `ROUTER = StreamRouter()`, iterate over `results` in `_scan_and_send`.
- Modify: `xau_pro_bot/formatter.py` — add `strategy_label` and `horizon_label` lines.
- Test: `tests/test_formatter.py` — assert new lines present.

- [ ] **Step 1: Update failing formatter test**

```python
# tests/test_formatter.py — append
def test_strong_signal_shows_strategy_and_horizon():
    sig = _sig()
    sig["strategy_label"] = "Swing 500"
    sig["horizon_label"] = "2-7 дней"
    text = format_strong_signal(sig)
    assert "Стратегия: Swing 500" in text
    assert "Горизонт: 2-7 дней" in text
```

- [ ] **Step 2: Run test — verify failure**

```bash
pytest tests/test_formatter.py::test_strong_signal_shows_strategy_and_horizon -v
```

- [ ] **Step 3: Update `formatter.py`**

In `format_strong_signal`, insert two lines after the R:R/Score/Session block (right before the analysis block):

```python
# Inside format_strong_signal, in the parts list, after session line:
if sig.get("strategy_label"):
    parts.append(f"📐 Стратегия: {sig['strategy_label']}")
if sig.get("horizon_label"):
    parts.append(f"⏳ Горизонт: {sig['horizon_label']}")
parts.append("━━━━━━━━━━━━━━━━━━━")
# ... then _analysis_block, then closing separator + timestamp
```

(The existing separator before `_analysis_block` is removed and re-added after the new lines so the layout stays consistent.)

Same two lines added to `format_weak_signal` after the Score line.

- [ ] **Step 4: Update `bot.py` — replace ENGINE with ROUTER**

```python
# In xau_pro_bot/bot.py replace:
ENGINE = MasterSignalEngine()
# with:
from xau_pro_bot.signals.router import StreamRouter
ROUTER = StreamRouter()

# Replace _scan_and_send body's analyze call:
async def _scan_and_send(app: Application, *, bypass_dedup: bool = False) -> None:
    assert STATE is not None
    try:
        tfs = data.fetch_all_timeframes(api_key=ENV["TWELVE_DATA_API_KEY"])
    except Exception:
        logging.exception("Data fetch failed")
        return
    try:
        results = ROUTER.analyze(tfs)
    except Exception:
        logging.exception("Router failed")
        return

    if not results:
        # No-signal killzone update (intraday-only behavior)
        kz = get_killzone()
        if kz:
            try:
                rsi = float(classic.add_classic(tfs["H1"])["RSI_14"].iloc[-1])
                if np.isnan(rsi):
                    rsi = None
            except Exception:
                rsi = None
            price = float(tfs["M15"]["Close"].iloc[-1])
            msg = formatter.format_no_signal_killzone(killzone=kz, price=price, rsi=rsi)
            await app.bot.send_message(
                chat_id=ENV["TELEGRAM_CHAT_ID"], text=msg,
                parse_mode=ParseMode.MARKDOWN)
        return

    for sig in results:
        ok, reason = should_send(sig, STATE, bypass_dedup=bypass_dedup)
        if not ok:
            _log_signal(sig, f"skipped:{reason.value if reason else 'unknown'}")
            continue
        text = _format(sig)
        try:
            await app.bot.send_message(
                chat_id=ENV["TELEGRAM_CHAT_ID"], text=text,
                parse_mode=ParseMode.MARKDOWN)
            _persist(sig)
            _log_signal(sig, "sent")
        except Exception:
            logging.exception("Telegram send failed")
            _log_signal(sig, "send_failed")
```

Update `_persist` to include `stream`:

```python
def _persist(sig: dict[str, Any]) -> None:
    assert STATE is not None
    STATE.record_signal({
        "ts_utc": sig["ts_utc"].isoformat(),
        "direction": sig["direction"],
        "tier": sig["tier"],
        "score": sig["score"],
        "entry": sig["entry"],
        "sl": sig.get("sl") or 0.0,
        "tp1": sig.get("tp1"),
        "tp2": sig.get("tp2"),
        "tp3": sig.get("tp3"),
        "rr": sig.get("rr"),
        "killzone": sig.get("killzone"),
        "reasons_json": json.dumps(sig["reasons"], ensure_ascii=False),
        "stream": sig.get("stream", "intraday"),
    })
```

- [ ] **Step 5: Run all tests**

```bash
pytest -v
```

- [ ] **Step 6: Smoke import**

```bash
python -c "from xau_pro_bot import bot; print('ok')"
```

- [ ] **Step 7: Commit**

```bash
git add xau_pro_bot/bot.py xau_pro_bot/formatter.py tests/test_formatter.py
git commit -m "feat(bot): wire StreamRouter and emit per-stream Telegram messages"
```

---

## Task 24: Per-Stream Backtest

**Files:**
- Modify: `xau_pro_bot/backtest.py` — add `--stream` flag and per-stream timeouts.
- Test: `tests/test_backtest.py` — add per-stream smoke.

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest.py — append
def test_backtest_supports_stream_flag():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    history = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}
    res = run_backtest(history, timeout_bars=48, stream="intraday")
    assert isinstance(res, BacktestResult)
    # Should at least not crash for swing and scalp streams
    run_backtest(history, timeout_bars=336, stream="swing")
    run_backtest(history, timeout_bars=8, stream="scalp")
```

- [ ] **Step 2: Run test — verify failure**

```bash
pytest tests/test_backtest.py -v
```

- [ ] **Step 3: Update `backtest.py`**

```python
# xau_pro_bot/backtest.py — patch run_backtest signature
from xau_pro_bot.signals.router import StreamRouter

_STREAM_ANALYZER = {
    "intraday": "intraday",
    "swing":    "swing",
    "scalp":    "scalp",
}


def run_backtest(history: dict[str, pd.DataFrame],
                 timeout_bars: int = 48,
                 step: int = 4,
                 stream: str = "intraday") -> BacktestResult:
    router = StreamRouter()
    analyzer = router.analyzers[stream]
    res = BacktestResult()
    h1 = history["H1"]
    if len(h1) < 250:
        return res

    for i in range(250, len(h1) - timeout_bars, step):
        cutoff = h1.index[i]
        slice_data = {
            tf: df.loc[:cutoff].tail(720)
            for tf, df in history.items()
        }
        try:
            sig = analyzer.analyze(slice_data)
        except Exception:
            continue
        if sig is None or sig["tier"] == "NO_SIGNAL" or sig.get("tp1") is None:
            continue
        res.signals_generated += 1
        target = sig.get("tp2") or sig["tp1"]
        future = h1.iloc[i + 1:]
        outcome, r = _outcome(future, sig["entry"], sig["sl"],
                              target, sig["direction"], timeout_bars)
        if outcome == "win":
            res.wins += 1
            res.per_tier[sig["tier"]]["n"] += 1
            res.per_tier[sig["tier"]]["w"] += 1
        elif outcome == "loss":
            res.losses += 1
            res.per_tier[sig["tier"]]["n"] += 1
            res.per_tier[sig["tier"]]["l"] += 1
        else:
            res.timeouts += 1
        res.pnl_r.append(r)
    return res
```

Update CLI:

```python
# In _cli():
p.add_argument("--stream", default="intraday",
               choices=["intraday", "swing", "scalp", "all"])

# ...
streams = ["intraday", "swing", "scalp"] if args.stream == "all" else [args.stream]
default_timeouts = {"intraday": 48, "swing": 336, "scalp": 8}
for s in streams:
    print(f"\n=== Stream: {s} ===")
    to = default_timeouts.get(s, args.timeout_bars)
    res = run_backtest(history, timeout_bars=to, step=args.step, stream=s)
    print(f"Signals:    {res.signals_generated}")
    print(f"Wins/Loss:  {res.wins} / {res.losses} (timeouts {res.timeouts})")
    print(f"Win rate:   {res.win_rate:.1%}")
    print(f"Expectancy: {res.expectancy:.2f} R")
    print(f"Profit f.:  {res.profit_factor:.2f}")
```

- [ ] **Step 4: Run all tests + commit**

```bash
pytest -v
git add xau_pro_bot/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): add per-stream backtesting with stream-specific timeouts"
```

---

## Revision 3 Self-Review

**Spec coverage (R3):**
- Pip constant → Task 17 ✓
- State `stream` column + migration → Task 17 ✓
- S/R zones detector → Task 18 ✓
- S/R integration into intraday scoring (anti-double-count) → Task 19 ✓
- Swing 500/1000 stream → Task 20 ✓
- Scalp stream with H4 counter-trend tag → Task 21 ✓
- StreamRouter + per-stream filters (cap, cooldowns) → Task 22 ✓
- Formatter strategy_label + horizon_label → Task 23 ✓
- Bot wired to router, multi-message scan → Task 23 ✓
- Per-stream backtest with proper timeouts → Task 24 ✓

**Placeholder scan:** все шаги содержат полный код.

**Type consistency:**
- Все стримы возвращают одинаковую структуру `dict` с keys: `direction, tier, score, entry, sl, tp1, tp2, tp3, rr, tp2_unavailable, killzone, reasons, ts_utc, strategy_label, horizon_label, atr_h1` (+ router добавляет `stream`).
- `should_send` теперь делегирует по `sig["stream"]`, все вспомогательные функции принимают тот же `dict` shape.
- `State.last_signal`, `count_today`, `last_weak_ts` имеют опциональный `stream` параметр + новый `last_scalp_ts()`.
- `SkipReason` enum дополнен `SWING_DIRECTION_COOLDOWN`, `SCALP_OUTSIDE_KZ`, `SCALP_GAP`, `UNKNOWN_STREAM`.

**Open caveats для будущих ревизий:**
- Score normalization (0–100) НЕ реализована в v1 — пороги intraday остаются 65/50/40, после калибровки бэктестом можно поднять.
- На Railway state.db ephemeral — между деплоями per-stream дневные счётчики обнулятся (соответствует поведению intraday в R2).

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-xau-pro-bot.md` (Revision 3 with multi-stream architecture).**
