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

# ── XAU pip ───────────────────────────────────────────
XAU_PIP_VALUE = 0.10  # USD per pip

# ── Per-stream rate limits ────────────────────────────
MAX_INTRADAY_PER_DAY = 6
MAX_SWING_PER_DAY = 2
MAX_SCALP_PER_DAY = 4
SCALP_MIN_GAP_MINUTES = 30
SWING_DIRECTION_COOLDOWN_HOURS = 24

# Stream identifiers
STREAM_INTRADAY = "intraday"
STREAM_SWING = "swing"
STREAM_SCALP = "scalp"

# ── Optional AI confirmation layer ────────────────────
AI_ENABLED = os.getenv("AI_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
AI_MODEL_ID = ""
AI_MODEL_TYPE = "sklearn"
AI_MIN_CONFIDENCE = 0.65
AI_STRONG_CONFIDENCE = 0.75
AI_NO_TRADE_THRESHOLD = 0.60
AI_SCORE_BONUS = 8
AI_STRONG_SCORE_BONUS = 12
AI_CONFLICT_PENALTY = 10
AI_CACHE_DIR = "./models_cache"


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse common env bool values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float env var {name}: {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int env var {name}: {raw!r}") from exc


def load_ai_config() -> dict[str, str | bool | float | int]:
    """Return current AI config using live env values.

    Tests mutate env after module import, so this function reads os.environ
    directly instead of returning only module-import constants.
    """
    return {
        "enabled": _env_bool("AI_ENABLED", False),
        "model_id": os.getenv("AI_MODEL_ID", ""),
        "model_type": os.getenv("AI_MODEL_TYPE", "sklearn"),
        "min_confidence": _env_float("AI_MIN_CONFIDENCE", 0.65),
        "strong_confidence": _env_float("AI_STRONG_CONFIDENCE", 0.75),
        "no_trade_threshold": _env_float("AI_NO_TRADE_THRESHOLD", 0.60),
        "score_bonus": _env_int("AI_SCORE_BONUS", 8),
        "strong_score_bonus": _env_int("AI_STRONG_SCORE_BONUS", 12),
        "conflict_penalty": _env_int("AI_CONFLICT_PENALTY", 10),
        "cache_dir": os.getenv("AI_CACHE_DIR", "./models_cache"),
        "revision": os.getenv("AI_MODEL_REVISION", ""),
    }


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
