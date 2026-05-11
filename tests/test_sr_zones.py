import pytest

from xau_pro_bot.indicators.sr_zones import (
    find_sr_zones, find_psychological_levels,
)


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
