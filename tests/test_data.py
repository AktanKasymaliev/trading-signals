import pandas as pd
import pytest

from xau_pro_bot import data as data_mod


@pytest.fixture(autouse=True)
def reset_cache():
    data_mod._CACHE.clear()
    yield
    data_mod._CACHE.clear()


def _fake_td_response(rows: int = 100) -> dict:
    values = []
    for i in range(rows):
        values.append({
            "datetime": f"2026-01-{(i % 28) + 1:02d} 00:00:00",
            "open": "2000.0",
            "high": "2010.0",
            "low": "1995.0",
            "close": "2005.0",
            "volume": "1000",
        })
    return {"status": "ok", "values": values}


def test_normalize_response_to_df():
    payload = _fake_td_response(50)
    df = data_mod._payload_to_df(payload)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.is_monotonic_increasing
    assert df["Close"].iloc[-1] == pytest.approx(2005.0)


def test_normalize_volume_missing_becomes_nan():
    payload = _fake_td_response(10)
    for row in payload["values"]:
        row.pop("volume")
    df = data_mod._payload_to_df(payload)
    assert df["Volume"].isna().all()


def test_fetch_uses_cache(monkeypatch):
    payload = _fake_td_response(50)
    call_count = {"n": 0}

    def fake_fetch(tf):
        call_count["n"] += 1
        return data_mod._payload_to_df(payload)

    monkeypatch.setattr(data_mod, "_fetch_single_tf", fake_fetch)

    data_mod.fetch_all_timeframes(api_key="dummy")
    data_mod.fetch_all_timeframes(api_key="dummy")
    assert call_count["n"] == 5  # 5 TFs, cached on 2nd call


def test_fetch_retries_on_error(monkeypatch):
    attempts = {"n": 0}

    def flaky(tf):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return data_mod._payload_to_df(_fake_td_response(50))

    monkeypatch.setattr(data_mod, "_fetch_single_tf", flaky)
    monkeypatch.setattr(data_mod.time, "sleep", lambda _: None)

    df = data_mod._fetch_with_retry("M15")
    assert isinstance(df, pd.DataFrame)
    assert attempts["n"] == 3
