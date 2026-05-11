from pathlib import Path

import pandas as pd
import pytest

from xau_pro_bot.backtest import (
    load_csv_history, run_backtest, BacktestResult,
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
