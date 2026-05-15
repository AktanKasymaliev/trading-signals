from pathlib import Path

import pandas as pd
import pytest

from xau_pro_bot.backtest import (
    load_csv_history, run_backtest, BacktestResult, compare_backtests,
)


def test_load_csv_history():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    assert len(df) > 0
    assert set(df.columns) >= {"Open", "High", "Low", "Close", "Volume"}


def test_run_backtest_returns_result():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    result = run_backtest(
        history={tf: df for tf in ("W1", "D1", "H4", "H1", "M15")},
        timeout_bars=48,
    )
    assert isinstance(result, BacktestResult)
    assert result.signals_generated >= 0
    assert 0 <= result.win_rate <= 1


def test_backtest_supports_stream_flag():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    history = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}

    intraday = run_backtest(history, timeout_bars=48, stream="intraday")
    swing = run_backtest(history, timeout_bars=336, stream="swing")
    scalp = run_backtest(history, timeout_bars=8, stream="scalp")

    assert isinstance(intraday, BacktestResult)
    assert isinstance(swing, BacktestResult)
    assert isinstance(scalp, BacktestResult)


class BacktestMockAI:
    def predict(self, features):
        return {
            "direction": "NO_TRADE",
            "confidence": 0.65,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }


def test_run_backtest_accepts_ai_model():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    history = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}

    result = run_backtest(
        history=history,
        timeout_bars=48,
        stream="intraday",
        use_ai=True,
        ai_model=BacktestMockAI(),
    )

    assert isinstance(result, BacktestResult)
    assert result.blocked_signals >= 0


def test_compare_backtests_returns_baseline_and_ai():
    df = load_csv_history(Path("tests/fixtures/h1_sample.csv"))
    history = {tf: df for tf in ("W1", "D1", "H4", "H1", "M15")}

    comparison = compare_backtests(
        history=history,
        timeout_bars=48,
        step=24,
        stream="intraday",
        ai_model=BacktestMockAI(),
    )

    assert set(comparison) == {"baseline", "ai", "summary"}
    assert isinstance(comparison["baseline"], BacktestResult)
    assert isinstance(comparison["ai"], BacktestResult)
    assert "trade_delta" in comparison["summary"]
