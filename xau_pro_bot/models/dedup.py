"""Drop near-identical Path D samples produced by sub-H1 harvest cadence."""

from __future__ import annotations

import pandas as pd


def dedup_near_identical(df: pd.DataFrame,
                         key_cols: tuple[str, ...] = ("entry", "sl", "direction"),
                         tol: float = 0.5) -> pd.DataFrame:
    """Keep the first occurrence of (entry, sl, direction) within `tol` price units.

    Time-ordered input is required; output preserves order and DatetimeIndex.
    """
    if df.empty:
        return df
    df = df.sort_index().copy()
    keep = [True]
    last = df.iloc[0]
    for i in range(1, len(df)):
        row = df.iloc[i]
        same_dir = row.get("direction") == last.get("direction")
        close_entry = abs(float(row["entry"]) - float(last["entry"])) <= tol
        close_sl = abs(float(row["sl"]) - float(last["sl"])) <= tol
        if same_dir and close_entry and close_sl:
            keep.append(False)
        else:
            keep.append(True)
            last = row
    return df[keep]
