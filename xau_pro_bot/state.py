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
    reasons_json TEXT,
    stream TEXT NOT NULL DEFAULT 'intraday'
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
        self._migrate()

    def _migrate(self) -> None:
        cols = [r[1] for r in self._conn.execute(
            "PRAGMA table_info(signals)").fetchall()]
        if "stream" not in cols:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN stream TEXT NOT NULL "
                "DEFAULT 'intraday'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_stream ON signals(stream)"
        )

    def close(self) -> None:
        self._conn.close()

    def record_signal(self, sig: dict[str, Any]) -> int:
        cols = ("ts_utc", "direction", "tier", "score", "entry", "sl",
                "tp1", "tp2", "tp3", "rr", "killzone", "reasons_json", "stream")
        placeholders = ", ".join("?" * len(cols))
        values = tuple(
            sig.get(c) if c != "stream" else sig.get("stream", "intraday")
            for c in cols
        )
        cur = self._conn.execute(
            f"INSERT INTO signals ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )
        return int(cur.lastrowid or 0)

    def last_signal(self, direction: str | None = None,
                    stream: str | None = None) -> dict[str, Any] | None:
        where = []
        params: list[Any] = []
        if direction:
            where.append("direction = ?")
            params.append(direction)
        if stream:
            where.append("stream = ?")
            params.append(stream)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        row = self._conn.execute(
            f"SELECT * FROM signals {clause} ORDER BY id DESC LIMIT 1",
            tuple(params),
        ).fetchone()
        return dict(row) if row else None

    def count_today(self, tier: str | None = None,
                    stream: str | None = None) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        where = ["substr(ts_utc, 1, 10) = ?"]
        params: list[Any] = [today]
        if tier:
            where.append("tier = ?")
            params.append(tier)
        if stream:
            where.append("stream = ?")
            params.append(stream)
        row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM signals WHERE {' AND '.join(where)}",
            tuple(params),
        ).fetchone()
        return int(row["n"])

    def last_weak_ts(self, stream: str = "intraday") -> datetime | None:
        row = self._conn.execute(
            "SELECT ts_utc FROM signals WHERE tier = 'WEAK' AND stream = ? "
            "ORDER BY id DESC LIMIT 1",
            (stream,),
        ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["ts_utc"])

    def last_scalp_ts(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT ts_utc FROM signals WHERE stream = 'scalp' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["ts_utc"])

    def prune_old(self, days: int = 90) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.execute("DELETE FROM signals WHERE ts_utc < ?", (cutoff,))
        return int(cur.rowcount)
