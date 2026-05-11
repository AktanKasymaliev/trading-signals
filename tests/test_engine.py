import pandas as pd
import pytest

from xau_pro_bot.signals.engine import MasterSignalEngine


def _enriched_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {tf: df.copy() for tf in ("W1", "D1", "H4", "H1", "M15")}


def test_engine_returns_required_keys(uptrend_df):
    eng = MasterSignalEngine()
    result = eng.analyze(_enriched_data(uptrend_df))
    for key in ("direction", "tier", "score", "entry",
                "killzone", "reasons", "tp2_unavailable"):
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
    if result["tier"] != "NO_SIGNAL":
        assert result["direction"] == "BUY"
