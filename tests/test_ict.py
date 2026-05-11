from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from xau_pro_bot.indicators.ict import (
    find_ote,
    find_fvg,
    find_order_blocks,
    find_liquidity,
    get_killzone,
)


def test_find_ote_returns_dict_with_required_keys(uptrend_df):
    res = find_ote(uptrend_df, lookback=20)
    assert set(res.keys()) >= {"in_ote", "ote_low", "ote_high",
                               "swing_high", "swing_low", "direction"}


def test_find_ote_short_df_neutral(short_df):
    res = find_ote(short_df, lookback=20)
    assert res["in_ote"] is False


def test_find_fvg_returns_list(df_with_fvg):
    gaps = find_fvg(df_with_fvg, max_gaps=5)
    assert isinstance(gaps, list)


def test_find_fvg_empty_for_flat(flat_df):
    gaps = find_fvg(flat_df, max_gaps=5)
    assert gaps == []


def test_find_order_blocks_returns_list(uptrend_df):
    obs = find_order_blocks(uptrend_df, lookback=50)
    assert isinstance(obs, list)
    for ob in obs:
        assert ob["type"] in ("bull", "bear")
        assert ob["high"] >= ob["low"]


def test_find_liquidity_keys(uptrend_df):
    res = find_liquidity(uptrend_df)
    assert set(res.keys()) == {"buy_side", "sell_side"}


def test_killzone_london():
    now = datetime(2026, 5, 11, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) == "London KZ"


def test_killzone_ny_am():
    now = datetime(2026, 5, 11, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) == "NY AM KZ"


def test_killzone_outside():
    now = datetime(2026, 5, 11, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) is None


def test_killzone_dst_aware():
    now = datetime(2026, 11, 15, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_killzone(now) == "London KZ"


def test_killzone_rejects_naive_datetime():
    with pytest.raises(ValueError):
        get_killzone(datetime(2026, 5, 11, 3, 0))
