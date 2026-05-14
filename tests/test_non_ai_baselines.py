from xau_pro_bot.backtest import BacktestResult
from scripts.eval_path_d import tier_filter_result


def test_tier_filter_keeps_only_requested_tiers():
    base = BacktestResult()
    base.signals_generated = 100
    base.wins = 45
    base.losses = 55
    base.per_tier = {
        "WEAK":   {"n": 40, "w": 12, "l": 28, "rr": [-1, 2, -1, 2]},
        "NORMAL": {"n": 40, "w": 20, "l": 20, "rr": [2, -1, 2, -1]},
        "STRONG": {"n": 20, "w": 13, "l":  7, "rr": [2, 2, -1, 2]},
    }
    out = tier_filter_result(base, keep={"STRONG"})
    assert out.signals_generated == 20
    assert out.wins == 13
    assert out.losses == 7
    assert out.rr_values, "tier_filter_result must surface rr for PF/expectancy"
