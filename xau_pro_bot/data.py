"""Twelve Data REST client with TTL cache and retry."""

from __future__ import annotations

import logging
import time
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
    if isinstance(payload, tuple):
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
