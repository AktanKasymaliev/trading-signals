"""Explicit label policies for the Path D filter dataset.

Each policy maps (outcome_class, final_R) → label_filter in {0, 1} (or drop).
UNRESOLVED is never silently merged into BAD — the policy decides.
"""

from __future__ import annotations

import enum

import pandas as pd


class LabelPolicy(str, enum.Enum):
    TP1_UNRESOLVED_BAD = "tp1_unresolved_bad"
    TP1_UNRESOLVED_DROP = "tp1_unresolved_drop"
    TP2_UNRESOLVED_BAD = "tp2_unresolved_bad"
    PLUS_1R_BEFORE_MINUS_1R = "plus_1r_before_minus_1r"
    PLUS_1_5R_BEFORE_MINUS_1R = "plus_1_5r_before_minus_1r"


_VALID = {p.value for p in LabelPolicy}


def apply_label_policy(df: pd.DataFrame, policy: "LabelPolicy | str") -> pd.DataFrame:
    val = policy.value if isinstance(policy, LabelPolicy) else policy
    if val not in _VALID:
        raise ValueError(f"unknown label policy: {policy!r}")
    out = df.copy()

    if val == LabelPolicy.TP1_UNRESOLVED_BAD.value:
        out["label_filter"] = (out["outcome_class"] == "TP").astype(int)
    elif val == LabelPolicy.TP1_UNRESOLVED_DROP.value:
        out = out[out["outcome_class"] != "UNRESOLVED"].copy()
        out["label_filter"] = (out["outcome_class"] == "TP").astype(int)
    elif val == LabelPolicy.TP2_UNRESOLVED_BAD.value:
        out["label_filter"] = (out["outcome_class"] == "TP").astype(int)
    elif val == LabelPolicy.PLUS_1R_BEFORE_MINUS_1R.value:
        out["label_filter"] = (out["final_R"] >= 1.0).astype(int)
    elif val == LabelPolicy.PLUS_1_5R_BEFORE_MINUS_1R.value:
        out["label_filter"] = (out["final_R"] >= 1.5).astype(int)

    return out
