"""Dev-only lifecycle smoke (no Telegram, no live market data).

Inserts a synthetic ACTIVE signal into a temp SQLite, drives the
lifecycle.evaluate_candle pipeline through TP1 → TP2 → SL/timeout,
and prints the formatter output that the bot would send.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xau_pro_bot import formatter
from xau_pro_bot.lifecycle import (
    Candle, TTL_BY_STREAM, evaluate_candle, lifecycle_signal_from_row,
)
from xau_pro_bot.state import State


def _apply(state: State, sig, candle: Candle, now: datetime) -> tuple[str, str]:
    tr = evaluate_candle(sig, candle, now=now,
                          ttl_hours=TTL_BY_STREAM[sig.stream])
    if tr is None:
        return sig.status, "no-op"
    state.update_lifecycle(
        tr.signal_id,
        status=tr.new_status,
        closed=tr.closed,
        final_R=tr.final_R,
        max_favorable_R=tr.max_favorable_R,
        max_adverse_R=tr.max_adverse_R,
        closed_at=tr.closed_at,
    )
    msg = formatter.format_lifecycle_transition(
        signal_id=tr.signal_id,
        direction=sig.direction,
        old_status=tr.old_status,
        new_status=tr.new_status,
        closed=tr.closed,
        final_R=tr.final_R,
        entry=sig.entry,
        price=tr.price,
    )
    return tr.new_status, msg


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dev_lifecycle_"))
    db_path = tmp / "dev.db"
    print(f"[dev-smoke] DB: {db_path}")
    state = State(db_path=str(db_path))

    opened = datetime.now(timezone.utc) - timedelta(hours=1)
    sid = state.record_signal({
        "ts_utc": opened.isoformat(),
        "direction": "BUY",
        "tier": "STRONG",
        "score": 80,
        "entry": 2000.0,
        "sl": 1990.0,
        "tp1": 2010.0,
        "tp2": 2020.0,
        "tp3": 2030.0,
        "rr": 3.0,
        "killzone": "London KZ",
        "reasons_json": "{}",
        "stream": "intraday",
        "ai_action": "PASS",
        "ai_risk_label": "CLEAN_SETUP",
        "ai_model_name": "path_c_lgb",
    })
    print(f"[dev-smoke] inserted synthetic signal id={sid}")

    print("\n--- /active (initial ACTIVE) ---")
    print(formatter.format_active_signals(state.get_active()))

    print("\n--- /history (empty before close) ---")
    print(formatter.format_history(state.recent_closed(10)))

    print("\n--- /stats (initial) ---")
    print(formatter.format_stats(
        state.lifecycle_metrics(days=1),
        state.lifecycle_metrics(days=7),
        len(state.get_active()),
        state.lifecycle_stats_by_risk(),
    ))

    # 1) Candle hits TP1 only.
    row = next(r for r in state.get_active() if r["id"] == sid)
    sig = lifecycle_signal_from_row(row)
    c1 = Candle(high=2012.0, low=2003.0, close=2010.5,
                ts=opened + timedelta(minutes=30))
    status, msg = _apply(state, sig, c1, c1.ts)
    print(f"\n--- candle TP1 → status={status} ---")
    print(msg)

    # 2) Next candle hits TP2 but not TP3.
    row = next(r for r in state.get_active() if r["id"] == sid)
    sig = lifecycle_signal_from_row(row)
    c2 = Candle(high=2022.0, low=2015.0, close=2021.0,
                ts=opened + timedelta(minutes=45))
    status, msg = _apply(state, sig, c2, c2.ts)
    print(f"\n--- candle TP2 → status={status} ---")
    print(msg)

    # 3) Reversal → SL.
    row = next(r for r in state.get_active() if r["id"] == sid)
    sig = lifecycle_signal_from_row(row)
    c3 = Candle(high=2023.0, low=1989.0, close=1992.0,
                ts=opened + timedelta(hours=2))
    status, msg = _apply(state, sig, c3, c3.ts)
    print(f"\n--- candle SL → status={status} ---")
    print(msg)

    print("\n--- /active after close (must be empty) ---")
    print(formatter.format_active_signals(state.get_active()))

    print("\n--- /history after close ---")
    print(formatter.format_history(state.recent_closed(10)))

    print("\n--- /stats after close ---")
    print(formatter.format_stats(
        state.lifecycle_metrics(days=1),
        state.lifecycle_metrics(days=7),
        len(state.get_active()),
        state.lifecycle_stats_by_risk(),
    ))

    # Verify DB row is consistent with SL_HIT.
    row = state._conn.execute(
        "SELECT status, final_R, closed_at, max_favorable_R, max_adverse_R "
        "FROM signals WHERE id=?", (sid,)).fetchone()
    print(f"\n[dev-smoke] final DB row: {dict(row)}")
    state.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
