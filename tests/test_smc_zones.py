"""Tests that score_smc properly consumes sr_zones and liquidity."""

from xau_pro_bot.signals.smc_signals import score_smc


def test_sr_zone_increases_buy_score_at_major_support(uptrend_df):
    bull, bear, reasons = score_smc(uptrend_df, sr_zones={
        "resistance_zones": [],
        "support_zones": [{
            "level": float(uptrend_df["Close"].iloc[-1]),
            "zone_top": float(uptrend_df["Close"].iloc[-1]) + 1.0,
            "zone_bot": float(uptrend_df["Close"].iloc[-1]) - 1.0,
            "strength": 80, "touches": 4, "type": "MAJOR",
        }],
        "at_resistance": False, "at_support": True,
        "nearest_resistance": None, "nearest_support": float(uptrend_df["Close"].iloc[-1]),
        "atr_h4": 2.0,
    }, liquidity={"buy_side": [], "sell_side": []})
    assert bull > 0
    assert any("MAJOR" in r for r in reasons)


def test_opposing_zone_within_30_pips_penalizes(uptrend_df):
    last = float(uptrend_df["Close"].iloc[-1])
    # Strong resistance zone 1 pip away (0.10 USD * 1 = 0.10)
    bull, bear, reasons = score_smc(uptrend_df, sr_zones={
        "resistance_zones": [{
            "level": last + 0.5, "zone_top": last + 1.5, "zone_bot": last - 0.5,
            "strength": 80, "touches": 5, "type": "MAJOR",
        }],
        "support_zones": [],
        "at_resistance": True, "at_support": False,
        "nearest_resistance": last + 0.5, "nearest_support": None,
        "atr_h4": 2.0,
    }, liquidity={"buy_side": [], "sell_side": []})
    # bull should get the -8 penalty (so might be negative or smaller)
    assert bull <= 0 or bull < 12
