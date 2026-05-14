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


@pytest.mark.parametrize("artifact_path", sorted(
    Path("models_cache").glob("*stationary*.joblib"),
))
def test_stationary_model_artifacts_have_no_raw_price_features(artifact_path):
    """Path F artifacts tagged 'stationary' must list only stationary
    columns. Parametrization yields zero cases when no such artifact
    exists yet — intended behaviour pre-training."""
    bundle = joblib.load(artifact_path)
    cols = bundle.get("feature_cols") or []
    tag = bundle.get("feature_set", "legacy")
    if tag != "stationary":
        pytest.skip(f"artifact {artifact_path.name} tagged {tag!r}, not stationary")
    leaking = [c for c in cols if FORBIDDEN.match(c)]
    assert leaking == [], (
        f"raw-price columns in {artifact_path.name}: {leaking}"
    )
