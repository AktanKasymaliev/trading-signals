import json
from datetime import datetime

import pytest

from xau_pro_bot.state import State
from xau_pro_bot.signals.engine import MasterSignalEngine
from xau_pro_bot.signals.filters import should_send


@pytest.fixture
def state(tmp_path):
    return State(db_path=str(tmp_path / "p.db"))


def test_pipeline_records_signal(state, all_tfs):
    eng = MasterSignalEngine()
    sig = eng.analyze(all_tfs)
    ok, _ = should_send(sig, state)
    if ok:
        state.record_signal({
            "ts_utc": sig["ts_utc"].isoformat(),
            "direction": sig["direction"],
            "tier": sig["tier"],
            "score": sig["score"],
            "entry": sig["entry"],
            "sl": sig.get("sl") or 0.0,
            "tp1": sig.get("tp1"),
            "tp2": sig.get("tp2"),
            "tp3": sig.get("tp3"),
            "rr": sig.get("rr"),
            "killzone": sig.get("killzone"),
            "reasons_json": json.dumps(sig["reasons"]),
        })
        assert state.last_signal() is not None
