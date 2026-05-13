"""TP/SL outcome resolver on M15 future bars for Path D labeling."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import pandas as pd


class OutcomeClass(str, enum.Enum):
    TP = "TP"
    SL = "SL"
    UNRESOLVED = "UNRESOLVED"
    SAME_CANDLE_SL_FIRST = "SAME_CANDLE_SL_FIRST"


@dataclass(frozen=True)
class Outcome:
    hit_tp: bool
    hit_sl: bool
    unresolved: bool
    same_candle_conflict: bool
    final_R: float
    mfe_R: float
    mae_R: float
    bars_to_outcome: int
    tp_used: float
    outcome_class: OutcomeClass


def resolve_outcome_m15(entry: float, sl: float, tp: float,
                        direction: str, m15_future: pd.DataFrame,
                        timeout_bars: int = 192) -> Outcome:
    """Resolve a hypothetical trade on M15 future bars.

    Conservative rule: if a single candle's range touches both TP and SL,
    treat it as SL-first (and record same_candle_conflict).
    """
    risk = abs(entry - sl)
    if risk <= 0:
        raise ValueError("zero-risk trade: entry == sl")
    R_tp = abs(tp - entry) / risk
    mfe = 0.0
    mae = 0.0
    bars = m15_future.iloc[:timeout_bars]

    for k, (_, bar) in enumerate(bars.iterrows(), start=1):
        if direction == "BUY":
            mfe = max(mfe, (bar.High - entry) / risk)
            mae = min(mae, (bar.Low - entry) / risk)
            hit_sl = bar.Low <= sl
            hit_tp = bar.High >= tp
        else:
            mfe = max(mfe, (entry - bar.Low) / risk)
            mae = min(mae, (entry - bar.High) / risk)
            hit_sl = bar.High >= sl
            hit_tp = bar.Low <= tp

        if hit_sl and hit_tp:
            return Outcome(False, True, False, True, -1.0, mfe, mae, k, tp,
                           OutcomeClass.SAME_CANDLE_SL_FIRST)
        if hit_sl:
            return Outcome(False, True, False, False, -1.0, mfe, mae, k, tp,
                           OutcomeClass.SL)
        if hit_tp:
            return Outcome(True, False, False, False, R_tp, mfe, mae, k, tp,
                           OutcomeClass.TP)

    return Outcome(False, False, True, False, 0.0, mfe, mae,
                   min(timeout_bars, len(bars)), tp, OutcomeClass.UNRESOLVED)
