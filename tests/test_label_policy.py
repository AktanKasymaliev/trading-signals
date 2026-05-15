import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.label_policy import LabelPolicy, apply_label_policy


def _sample_df():
    return pd.DataFrame({
        "outcome_class": ["TP", "SL", "UNRESOLVED", "SAME_CANDLE_SL_FIRST", "TP"],
        "final_R":       [2.0, -1.0, 0.0, -1.0, 1.6],
        "mfe_R":         [2.0, 0.5, 0.3, 0.2, 1.7],
        "tp_used":       [1.0, 1.0, 1.0, 1.0, 2.0],
    })


def test_tp1_unresolved_bad():
    out = apply_label_policy(_sample_df(), LabelPolicy.TP1_UNRESOLVED_BAD)
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_tp1_unresolved_drop():
    out = apply_label_policy(_sample_df(), LabelPolicy.TP1_UNRESOLVED_DROP)
    assert len(out) == 4
    assert "UNRESOLVED" not in out["outcome_class"].tolist()


def test_tp2_unresolved_bad_uses_higher_rr():
    df = _sample_df().assign(tp_used=[2.0, 2.0, 2.0, 2.0, 2.0])
    out = apply_label_policy(df, LabelPolicy.TP2_UNRESOLVED_BAD)
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_plus_1r_before_minus_1r():
    out = apply_label_policy(_sample_df(), LabelPolicy.PLUS_1R_BEFORE_MINUS_1R)
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_plus_1_5r_before_minus_1r():
    out = apply_label_policy(_sample_df(), LabelPolicy.PLUS_1_5R_BEFORE_MINUS_1R)
    assert list(out["label_filter"]) == [1, 0, 0, 0, 1]


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        apply_label_policy(_sample_df(), "garbage")  # type: ignore[arg-type]
