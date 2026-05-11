from xau_pro_bot.indicators.wyckoff import detect_wyckoff


def test_detect_wyckoff_returns_required_keys(uptrend_df):
    res = detect_wyckoff(uptrend_df)
    assert {"phase", "bias", "strength"} <= res.keys()
    assert res["bias"] in ("bull", "bear", "neutral")
    assert 0 <= res["strength"] <= 100


def test_detect_wyckoff_short_df_neutral(short_df):
    res = detect_wyckoff(short_df)
    assert res["phase"] == "neutral"
    assert res["bias"] == "neutral"


def test_detect_wyckoff_uptrend_is_bull(uptrend_df):
    res = detect_wyckoff(uptrend_df)
    assert res["bias"] == "bull"


def test_detect_wyckoff_downtrend_is_bear(downtrend_df):
    res = detect_wyckoff(downtrend_df)
    assert res["bias"] == "bear"
