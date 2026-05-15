"""Audit gate: Path F must never carry raw absolute-price features.

Fails CI if any column matching `close_(m15|h1|h4|d1)` or `ema(8|21|50|200)_h1`
leaks into the stationary feature builder output, or appears in the
feature-column list tagged as `stationary` inside a model artifact.

Also enforces (per user constraint):
- Suspicious raw-price-suggesting names without a stationary suffix
  (`open`, `high`, `low`, `close`, bare `ema\\d+`) are forbidden.
- Builder output values must stay within stationary scale even when
  the underlying instrument trades at XAU-magnitude prices (~3000).
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.features_stationary import (
    STATIONARY_FEATURES,
    build_stationary_features,
)


FORBIDDEN = re.compile(r"^(close_(m15|h1|h4|d1)|ema(8|21|50|200)_h1)$")

# A stationary feature name must either contain one of these tokens, or be
# explicitly enumerated in ALLOWED_BARE. Bare names like `open`/`ema8` would
# fail this gate.
_STATIONARY_SUFFIX_TOKENS = (
    "_atr",
    "_vs_",
    "return_",
    "_percentile",
    "_pct",
    "_ratio",
    "_norm",
    "_z",
    "is_",
)
ALLOWED_BARE: set[str] = set()
_BARE_PRICE_TOKEN = re.compile(
    r"^(open|high|low|close|ema\d+|sma\d+|price|vwap)(_[a-z0-9]+)?$"
)

# Stationary scale ceiling. ATR-normalised distances, returns, ratios, and
# bounded percentiles should all sit well under this. Raw XAU prices (~3000)
# would blow past it. Picked generously to avoid flakiness on edge bars.
STATIONARY_VALUE_CEILING = 50.0


def test_stationary_feature_list_has_no_raw_price_columns():
    leaking = [c for c in STATIONARY_FEATURES if FORBIDDEN.match(c)]
    assert leaking == [], f"raw-price columns in STATIONARY_FEATURES: {leaking}"


def test_build_stationary_features_output_has_no_raw_price_columns():
    df, _ = build_stationary_features({})
    leaking = [c for c in df.columns if FORBIDDEN.match(c)]
    assert leaking == [], f"raw-price columns leaked from builder: {leaking}"


def _is_stationary_name(col: str) -> bool:
    """A name is stationary iff it contains a known stationary suffix token,
    is explicitly allow-listed, or is not a bare price/MA name."""
    if col in ALLOWED_BARE:
        return True
    if _BARE_PRICE_TOKEN.match(col) and not any(
        tok in col for tok in _STATIONARY_SUFFIX_TOKENS
    ):
        return False
    return True


def test_stationary_feature_names_have_no_raw_price_suggestion():
    """Catches names like `open`, `close_h1`, `ema8` even if not in FORBIDDEN."""
    leaking = [c for c in STATIONARY_FEATURES if not _is_stationary_name(c)]
    assert leaking == [], (
        f"raw-price-suggesting names in STATIONARY_FEATURES: {leaking}"
    )


def _synthetic_xau_tfs() -> dict[str, pd.DataFrame]:
    """Construct a multi-TF dict at XAU-magnitude price (~3000) so that any
    feature that accidentally returns a raw price level will spike well past
    the stationary value ceiling."""
    rng = np.random.default_rng(42)
    n = 400
    base = 3000.0
    drift = np.cumsum(rng.normal(0.0, 1.5, size=n))
    close = base + drift
    high = close + np.abs(rng.normal(0.0, 1.0, size=n))
    low = close - np.abs(rng.normal(0.0, 1.0, size=n))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    h1 = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                      index=idx)
    m15 = h1.copy()
    h4 = h1.iloc[::4].copy()
    return {"M15": m15, "H1": h1, "H4": h4}


def test_build_stationary_features_values_are_stationary_scale():
    df, complete = build_stationary_features(_synthetic_xau_tfs())
    assert complete, "synthetic XAU tfs should produce a complete feature row"
    vals = df.iloc[0].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals)), f"non-finite values: {df.iloc[0].to_dict()}"
    max_abs = float(np.max(np.abs(vals)))
    assert max_abs < STATIONARY_VALUE_CEILING, (
        f"stationary builder emitted raw-price-scale value {max_abs:.2f} "
        f"(ceiling {STATIONARY_VALUE_CEILING}). Row: {df.iloc[0].to_dict()}"
    )


def _stationary_artifacts() -> list[Path]:
    """Find any joblib artifact under models_cache/ whose bundle dict carries
    feature_set='stationary'. Subdirs are scanned via rglob so multi-bundle
    training output (e.g. models_cache/path_f_stationary/) is covered."""
    matches: list[Path] = []
    for p in sorted(Path("models_cache").rglob("*.joblib")):
        try:
            bundle = joblib.load(p)
        except Exception:
            continue
        if isinstance(bundle, dict) and bundle.get("feature_set") == "stationary":
            matches.append(p)
    return matches


@pytest.mark.parametrize("artifact_path", _stationary_artifacts() or [None])
def test_stationary_model_artifacts_have_no_raw_price_features(artifact_path):
    """Every artifact tagged feature_set='stationary' must list only
    stationary columns. Parametrization yields a single None case when no
    such artifact exists yet — intended behaviour pre-training."""
    if artifact_path is None:
        pytest.skip("no stationary-tagged artifacts on disk yet")
    bundle = joblib.load(artifact_path)
    cols = bundle.get("feature_cols") or []
    leaking = [c for c in cols if FORBIDDEN.match(c)]
    assert leaking == [], (
        f"raw-price columns in {artifact_path}: {leaking}"
    )
