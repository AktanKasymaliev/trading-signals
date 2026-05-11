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


def test_detect_stop_hunt_returns_keys(uptrend_df):
    df = uptrend_df.copy()
    df.iloc[-1, df.columns.get_loc("Low")] = float(df["Low"].min()) - 30
    df.iloc[-1, df.columns.get_loc("Close")] = float(df["Close"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("Open")] = float(df["Close"].iloc[-1]) - 0.5
    res = detect_stop_hunt(df, atr=5.0)
    assert "bull_hunt" in res
    assert "bear_hunt" in res


def test_stop_hunt_neutral_on_flat(flat_df):
    res = detect_stop_hunt(flat_df, atr=1.0)
    assert res["bull_hunt"] is False
    assert res["bear_hunt"] is False
