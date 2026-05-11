import pytest

from xau_pro_bot.indicators.classic import add_classic


def test_adds_all_expected_columns(uptrend_df):
    df = add_classic(uptrend_df)
    for col in ("EMA_8", "EMA_21", "EMA_50", "EMA_200",
                "RSI_14", "ATR_14", "vol_ratio",
                "pivot", "r1", "s1", "r2", "s2"):
        assert col in df.columns, f"missing {col}"


def test_short_df_returns_nans_not_crash(short_df):
    df = add_classic(short_df)
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
