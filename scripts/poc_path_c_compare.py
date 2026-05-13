"""Compare baseline vs Path C LightGBM on long M15 history."""

from __future__ import annotations

import sys

import pandas as pd

from xau_pro_bot.backtest import compare_backtests
from xau_pro_bot.models.hf_model import HFTradingModel


def main() -> int:
    m15 = pd.read_csv("./data_long_m15.csv")
    m15["datetime"] = pd.to_datetime(m15["datetime"], utc=True)
    m15 = m15.set_index("datetime").sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    history = {
        "M15": m15,
        "H1": m15.resample("1h").agg(agg).dropna(),
        "H4": m15.resample("4h").agg(agg).dropna(),
        "D1": m15.resample("1D").agg(agg).dropna(),
        "W1": m15.resample("1W").agg(agg).dropna(),
    }
    ai = HFTradingModel(model_id="", model_type="sklearn",
                        local_path="./models_cache/path_c_lgb.joblib")
    c = compare_backtests(history=history, timeout_bars=48, step=12,
                          stream="intraday", ai_model=ai)
    b, a, s = c["baseline"], c["ai"], c["summary"]
    print(f"BASE: sig={b.signals_generated} W/L={b.wins}/{b.losses} "
          f"WR={b.win_rate:.1%} E={b.expectancy:.2f}R PF={b.profit_factor:.2f}")
    print(f"AI  : sig={a.signals_generated} blocked={a.blocked_signals} "
          f"W/L={a.wins}/{a.losses} WR={a.win_rate:.1%} "
          f"E={a.expectancy:.2f}R PF={a.profit_factor:.2f}")
    print(f"DELTA: trades={s['trade_delta']} WR={s['win_rate_delta']:.1%} "
          f"E={s['expectancy_delta']:.2f}R PF={s['profit_factor_delta']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
