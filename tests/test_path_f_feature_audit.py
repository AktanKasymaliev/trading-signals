"""Audit gate: Path F must never carry raw absolute-price features.

Fails CI if any column matching `close_(m15|h1|h4|d1)` or `ema(8|21|50|200)_h1`
leaks into the stationary feature builder output, or appears in the
feature-column list tagged as `stationary` inside a model artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import pytest

from xau_pro_bot.models.features_stationary import (
    STATIONARY_FEATURES,
    build_stationary_features,
)


FORBIDDEN = re.compile(r"^(close_(m15|h1|h4|d1)|ema(8|21|50|200)_h1)$")


def test_stationary_feature_list_has_no_raw_price_columns():
    leaking = [c for c in STATIONARY_FEATURES if FORBIDDEN.match(c)]
    assert leaking == [], f"raw-price columns in STATIONARY_FEATURES: {leaking}"


def test_build_stationary_features_output_has_no_raw_price_columns():
    df, _ = build_stationary_features({})
    leaking = [c for c in df.columns if FORBIDDEN.match(c)]
    assert leaking == [], f"raw-price columns leaked from builder: {leaking}"


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
