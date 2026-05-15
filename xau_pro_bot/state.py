"""SQLite persistence for signals, dedup, rate-limit, and lifecycle state."""

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
    stream TEXT NOT NULL DEFAULT 'intraday',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    closed_at TEXT,
    max_favorable_R REAL NOT NULL DEFAULT 0.0,
    max_adverse_R REAL NOT NULL DEFAULT 0.0,
    final_R REAL,
    ai_action TEXT,
    ai_risk_label TEXT,
    ai_model_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc);
CREATE INDEX IF NOT EXISTS idx_signals_dir ON signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_tier ON signals(tier);
"""

_LIFECYCLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("stream", "TEXT NOT NULL DEFAULT 'intraday'"),
    ("status", "TEXT NOT NULL DEFAULT 'ACTIVE'"),
    ("closed_at", "TEXT"),
    ("max_favorable_R", "REAL NOT NULL DEFAULT 0.0"),
    ("max_adverse_R", "REAL NOT NULL DEFAULT 0.0"),
    ("final_R", "REAL"),
    ("ai_action", "TEXT"),
    ("ai_risk_label", "TEXT"),
    ("ai_model_name", "TEXT"),
)


def _empty_bucket() -> dict[str, Any]:
    return {"total": 0, "wins": 0, "losses": 0, "timeouts": 0,
            "active": 0, "sum_R": 0.0}


def _bump(buckets: dict[str, dict[str, Any]], key: str | None,
          *, active: bool, win: bool, loss: bool, timeout: bool,
          final_R: float | None) -> None:
    if not key:
        return
    b = buckets.setdefault(key, _empty_bucket())
    b["total"] += 1
    if active:
        b["active"] += 1
    if win:
        b["wins"] += 1
    if loss:
        b["losses"] += 1
    if timeout:
        b["timeouts"] += 1
    if final_R is not None:
        b["sum_R"] += float(final_R)


def _build_paper_report(rows: list[dict[str, Any]],
                        days: int) -> dict[str, Any]:
    total = len(rows)
    active = closed = wins = losses = timeouts = 0
    gross_win = gross_loss = 0.0
    sum_R = 0.0
    max_adverse: float | None = None
    by_stream: dict[str, dict[str, Any]] = {}
    by_risk: dict[str, dict[str, Any]] = {}
    by_action: dict[str, dict[str, Any]] = {}
    by_tier: dict[str, dict[str, Any]] = {}

    for r in rows:
        is_closed = r.get("closed_at") is not None
        status = r.get("status") or "ACTIVE"
        final_R = r.get("final_R")
        is_timeout = status == "TIMEOUT"
        is_win = is_closed and not is_timeout and (final_R or 0.0) > 0
        is_loss = is_closed and not is_timeout and (final_R or 0.0) <= 0
        is_active = not is_closed

        if is_active:
            active += 1
        else:
            closed += 1
        if is_win:
            wins += 1
            gross_win += float(final_R)
        if is_loss:
            losses += 1
            gross_loss += abs(float(final_R))
        if is_timeout:
            timeouts += 1
        if is_closed and final_R is not None:
            sum_R += float(final_R)

        mae = r.get("max_adverse_R")
        if mae is not None:
            mae_f = float(mae)
            if max_adverse is None or mae_f > max_adverse:
                max_adverse = mae_f

        for buckets, key in (
            (by_stream, r.get("stream")),
            (by_risk, r.get("ai_risk_label")),
            (by_action, r.get("ai_action")),
            (by_tier, r.get("tier")),
        ):
            _bump(buckets, key, active=is_active, win=is_win, loss=is_loss,
                  timeout=is_timeout,
                  final_R=final_R if is_closed else None)

    decided = wins + losses
    wr = (wins / decided) if decided else 0.0
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = 0.0
    expectancy = (sum_R / closed) if closed else 0.0

    return {
        "period_days": days,
        "total": total,
        "active": active,
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "wr": wr,
        "pf": pf,
        "expectancy": expectancy,
        "total_final_R": sum_R,
        "max_adverse_R": max_adverse,
        "by_stream": by_stream,
        "by_risk": by_risk,
        "by_action": by_action,
        "by_tier": by_tier,
    }


class State:
    """Thin wrapper over SQLite for signal persistence and lifecycle queries."""

    def __init__(self, db_path: str = "./state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        existing = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(signals)").fetchall()}
        for col, decl in _LIFECYCLE_COLUMNS:
            if col not in existing:
                self._conn.execute(
                    f"ALTER TABLE signals ADD COLUMN {col} {decl}"
                )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_stream ON signals(stream)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)"
        )

    def close(self) -> None:
        self._conn.close()

    # ── Inserts ────────────────────────────────────────────────────────
    def record_signal(self, sig: dict[str, Any]) -> int:
        cols = ("ts_utc", "direction", "tier", "score", "entry", "sl",
                "tp1", "tp2", "tp3", "rr", "killzone", "reasons_json",
                "stream", "ai_action", "ai_risk_label", "ai_model_name")
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

    # ── Lookups ────────────────────────────────────────────────────────
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

    # ── Lifecycle ──────────────────────────────────────────────────────
    def get_active(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE status = 'ACTIVE' OR "
            "(status IN ('TP1_HIT', 'TP2_HIT') AND closed_at IS NULL) "
            "ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_lifecycle(
        self,
        signal_id: int,
        *,
        status: str,
        closed: bool = False,
        final_R: float | None = None,
        max_favorable_R: float | None = None,
        max_adverse_R: float | None = None,
        closed_at: datetime | None = None,
    ) -> None:
        sets = ["status = ?"]
        params: list[Any] = [status]
        if max_favorable_R is not None:
            sets.append("max_favorable_R = ?")
            params.append(float(max_favorable_R))
        if max_adverse_R is not None:
            sets.append("max_adverse_R = ?")
            params.append(float(max_adverse_R))
        if closed:
            sets.append("closed_at = ?")
            params.append((closed_at or datetime.now(timezone.utc)).isoformat())
            if final_R is not None:
                sets.append("final_R = ?")
                params.append(float(final_R))
        params.append(signal_id)
        self._conn.execute(
            f"UPDATE signals SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

    def recent_closed(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE closed_at IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def lifecycle_stats_by_risk(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT COALESCE(ai_risk_label, 'UNKNOWN') AS lbl, "
            "SUM(CASE WHEN final_R > 0 THEN 1 ELSE 0 END) AS wins, "
            "SUM(CASE WHEN final_R <= 0 THEN 1 ELSE 0 END) AS losses, "
            "AVG(final_R) AS avg_R "
            "FROM signals WHERE closed_at IS NOT NULL "
            "GROUP BY COALESCE(ai_risk_label, 'UNKNOWN')"
        ).fetchall()
        return {
            r["lbl"]: {
                "wins": int(r["wins"] or 0),
                "losses": int(r["losses"] or 0),
                "avg_R": float(r["avg_R"] or 0.0),
            }
            for r in rows
        }

    def paper_report(self, days: int = 1) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = [
            dict(r) for r in self._conn.execute(
                "SELECT stream, tier, status, closed_at, final_R, "
                "max_adverse_R, ai_action, ai_risk_label "
                "FROM signals WHERE ts_utc >= ?",
                (cutoff,),
            ).fetchall()
        ]
        return _build_paper_report(rows, days)

    def lifecycle_metrics(self, days: int = 7) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT final_R FROM signals WHERE closed_at IS NOT NULL "
            "AND ts_utc >= ?",
            (cutoff,),
        ).fetchall()
        rs = [float(r["final_R"]) for r in rows if r["final_R"] is not None]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r <= 0]
        total = len(rs)
        wr = (len(wins) / total) if total else 0.0
        expectancy = (sum(rs) / total) if total else 0.0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss > 0:
            pf = gross_win / gross_loss
        elif gross_win > 0:
            pf = float("inf")
        else:
            pf = 0.0
        return {
            "wins": len(wins),
            "losses": len(losses),
            "total": total,
            "wr": wr,
            "expectancy": expectancy,
            "pf": pf,
        }
