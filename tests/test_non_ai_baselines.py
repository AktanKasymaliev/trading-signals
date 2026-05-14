from pathlib import Path

from xau_pro_bot.backtest import BacktestResult, run_backtest, load_csv_history
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


def test_run_backtest_populates_per_tier_rr():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    result = run_backtest(
        history={tf: df for tf in ("W1", "D1", "H4", "H1", "M15")},
        timeout_bars=48,
    )
    if result.signals_generated == 0:
        return
    total_rr = sum(len(cnt.get("rr", [])) for cnt in result.per_tier.values())
    assert total_rr == result.wins + result.losses, (
        "per_tier rr lists must accumulate one entry per resolved trade"
    )
    filtered = tier_filter_result(result, keep={"STRONG", "NORMAL"})
    if filtered.signals_generated > 0:
        assert filtered.rr_values, (
            "tier-filtered result must carry rr_values from run_backtest"
        )
