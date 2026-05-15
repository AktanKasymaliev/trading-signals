"""Smoke test: download SMC v2 model, run engine.analyze() on synthetic data.

Run:
    AI_ENABLED=true \\
    AI_MODEL_ID=JonusNattapong/xauusd-trading-ai-smc-v2 \\
    AI_MODEL_REVISION=d1ee87d058bf714af1b6f4b3979646dd0024b726 \\
    AI_MODEL_FILENAME=trading_model_15m.pkl \\
    AI_FEATURE_SET=smc_v2 \\
    AI_CACHE_DIR=./models_cache \\
    .venv/bin/python scripts/poc_smc_v2_smoke.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from xau_pro_bot.signals.engine import MasterSignalEngine


def _synthetic_tfs(n: int = 300, seed: int = 42) -> dict[str, pd.DataFrame]:
    np.random.seed(seed)
    base = 2000.0 + np.cumsum(np.random.normal(0, 1.5, n))
    m15 = pd.DataFrame({
        "Open": base,
        "High": base + 2.5,
        "Low": base - 2.5,
        "Close": base + np.random.normal(0, 0.8, n),
        "Volume": np.random.randint(100, 1000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    h1 = m15.resample("1h").agg(agg).dropna()
    h4 = m15.resample("4h").agg(agg).dropna()
    d1 = m15.resample("1D").agg(agg).dropna()
    w1 = d1.copy()
    return {"M15": m15, "H1": h1, "H4": h4, "D1": d1, "W1": w1}


def main() -> int:
    if os.getenv("AI_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
        print("AI_ENABLED is not true — set the env vars from the docstring at the top.")
        return 1
    tfs = _synthetic_tfs()
    engine = MasterSignalEngine()
    print("AI enabled:", engine.ai_enabled)
    print("AI feature set:", engine.ai_feature_set)
    print("AI adapter:", type(engine.ai_model).__name__ if engine.ai_model else None)
    sig = engine.analyze(tfs)
    serializable = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in sig.items() if k != "reasons"}
    print(json.dumps(serializable, indent=2, default=str))
    print("\nreasons:", sig["reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
