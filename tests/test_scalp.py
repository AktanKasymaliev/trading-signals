import pytest

from xau_pro_bot.indicators.scalping import scalp_signal
from xau_pro_bot.signals.scalp_analyzer import ScalpAnalyzer


def test_scalp_inactive_outside_killzone(uptrend_df, monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.indicators.scalping.get_killzone",
        lambda: None,
    )
    res = scalp_signal(m15_df=uptrend_df, h1_df=uptrend_df, h4_df=uptrend_df)
    assert res is None


def test_scalp_returns_dict_shape_in_killzone(monkeypatch, uptrend_df):
    monkeypatch.setattr(
        "xau_pro_bot.indicators.scalping.get_killzone",
        lambda: "London KZ",
    )
    res = scalp_signal(m15_df=uptrend_df, h1_df=uptrend_df, h4_df=uptrend_df)
    if res is not None:
        assert "direction" in res
        assert "sl" in res
        assert "tp1" in res
        assert "conditions_met" in res


def test_scalp_analyzer_no_signal_outside_kz(monkeypatch, uptrend_df):
    monkeypatch.setattr(
        "xau_pro_bot.indicators.scalping.get_killzone",
        lambda: None,
    )
    data = {tf: uptrend_df for tf in ("W1", "D1", "H4", "H1", "M15")}
    sig = ScalpAnalyzer().analyze(data)
    assert sig is None
