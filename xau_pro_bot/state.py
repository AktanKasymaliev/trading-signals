"""SQLite persistence for signals, dedup, and rate-limit state."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    direction TEXT NOT NULL,
    tier TEXT NOT NULL,
    score INTEGER NOT NULL,
    entry REAL NOT NULL,
    sl REAL NOT NULL,
    tp1 REAL,
    tp2 REAL,
    tp3 REAL,
    rr REAL,
    killzone TEXT,
    reasons_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc);
CREATE INDEX IF NOT EXISTS idx_signals_dir ON signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_tier ON signals(tier);
"""


class State:
    """Thin wrapper over SQLite for signal persistence and dedup queries."""

    def __init__(self, db_path: str = "./state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def record_signal(self, sig: dict[str, Any]) -> int:
        cols = ("ts_utc", "direction", "tier", "score", "entry", "sl",
                "tp1", "tp2", "tp3", "rr", "killzone", "reasons_json")
        placeholders = ", ".join("?" * len(cols))
        cur = self._conn.execute(
            f"INSERT INTO signals ({', '.join(cols)}) VALUES ({placeholders})",
            tuple(sig.get(c) for c in cols),
        )
        return int(cur.lastrowid or 0)

    def last_signal(self, direction: str | None = None) -> dict[str, Any] | None:
        if direction is None:
            row = self._conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM signals WHERE direction = ? ORDER BY id DESC LIMIT 1",
                (direction,),
            ).fetchone()
        return dict(row) if row else None

    def count_today(self, tier: str | None = None) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        if tier is None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE substr(ts_utc, 1, 10) = ?",
                (today,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM signals "
                "WHERE substr(ts_utc, 1, 10) = ? AND tier = ?",
                (today, tier),
            ).fetchone()
        return int(row["n"])

    def last_weak_ts(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT ts_utc FROM signals WHERE tier = 'WEAK' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["ts_utc"])

    def prune_old(self, days: int = 90) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.execute("DELETE FROM signals WHERE ts_utc < ?", (cutoff,))
        return int(cur.rowcount)
