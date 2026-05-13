import pandas as pd
from xau_pro_bot.models.dedup import dedup_near_identical


def test_dedup_keeps_first_within_tolerance():
    df = pd.DataFrame({
        "entry":     [1900.00, 1900.05, 1905.00, 1905.04],
        "sl":        [1895.00, 1895.00, 1900.00, 1900.00],
        "direction": ["BUY", "BUY", "BUY", "BUY"],
    }, index=pd.to_datetime(
        ["2024-01-01", "2024-01-01 00:15", "2024-01-01 01:00", "2024-01-01 01:15"],
        format="mixed", utc=True))
    out = dedup_near_identical(df, key_cols=("entry", "sl", "direction"), tol=0.1)
    assert len(out) == 2
    assert out.iloc[0]["entry"] == 1900.00
    assert out.iloc[1]["entry"] == 1905.00


def test_dedup_empty_df():
    df = pd.DataFrame(columns=["entry", "sl", "direction"])
    out = dedup_near_identical(df, key_cols=("entry", "sl", "direction"), tol=0.5)
    assert out.empty


def test_dedup_different_direction_not_merged():
    df = pd.DataFrame({
        "entry":     [1900.00, 1900.05],
        "sl":        [1895.00, 1905.00],
        "direction": ["BUY", "SELL"],
    }, index=pd.to_datetime(["2024-01-01", "2024-01-01 00:15"], format="mixed", utc=True))
    out = dedup_near_identical(df, key_cols=("entry", "sl", "direction"), tol=0.5)
    assert len(out) == 2
