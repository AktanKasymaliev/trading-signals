import pytest

from xau_pro_bot.indicators.sr_levels import (
    swing_highs_lows, nearest_above, nearest_below,
)


def test_swing_highs_lows_returns_lists(uptrend_df):
    sh, sl = swing_highs_lows(uptrend_df, window=5)
    assert isinstance(sh, list) and isinstance(sl, list)


def test_nearest_above_finds_closest():
    levels = [100.0, 105.0, 110.0, 120.0]
    assert nearest_above(102.0, levels) == pytest.approx(105.0)


def test_nearest_above_none_when_empty():
    assert nearest_above(100.0, [50.0, 60.0]) is None


def test_nearest_below_finds_closest():
    levels = [100.0, 105.0, 110.0, 120.0]
    assert nearest_below(108.0, levels) == pytest.approx(105.0)


def test_nearest_below_none_when_empty():
    assert nearest_below(100.0, [200.0, 300.0]) is None
