"""Signal lifecycle tracking: status transitions per candle.

Pure logic. Consumes the latest M15 candle plus the current signal
state, returns a transition record describing the resulting status
(ACTIVE / TP1_HIT / TP2_HIT / TP3_HIT / SL_HIT / TIMEOUT / CANCELLED).

Conservative rule: when a single candle pierces both SL and a TP,
SL wins (worst-case assumption for manual analysis).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


TTL_BY_STREAM: dict[str, float] = {
    "intraday": 24.0,
    "scalp": 8.0,
    "swing": 120.0,
}

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"SL_HIT", "TP3_HIT", "TIMEOUT", "CANCELLED"}
)

_TP_LEVEL_ORDER = ("TP1_HIT", "TP2_HIT", "TP3_HIT")


@dataclass(frozen=True)
class Candle:
    high: float
    low: float
    close: float
    ts: datetime


@dataclass(frozen=True)
class LifecycleSignal:
    id: int
    stream: str
    direction: str
    entry: float
    sl: float
    tp1: float | None
    tp2: float | None
    tp3: float | None
    status: str
    opened_at: datetime
    max_favorable_R: float
    max_adverse_R: float
    ai_action: str | None
    ai_risk_label: str | None
    ai_model_name: str | None


@dataclass(frozen=True)
class Transition:
    signal_id: int
    old_status: str
    new_status: str
    closed: bool
    closed_at: datetime | None
    final_R: float | None
    max_favorable_R: float
    max_adverse_R: float
    price: float


def _risk(sig: LifecycleSignal) -> float:
    risk = abs(sig.entry - sig.sl)
    return risk if risk > 0 else 1e-9


def _excursions(sig: LifecycleSignal, candle: Candle) -> tuple[float, float]:
    risk = _risk(sig)
    if sig.direction == "BUY":
        fav = (candle.high - sig.entry) / risk
        adv = (sig.entry - candle.low) / risk
    else:
        fav = (sig.entry - candle.low) / risk
        adv = (candle.high - sig.entry) / risk
    return max(0.0, fav), max(0.0, adv)


def _sl_hit(sig: LifecycleSignal, candle: Candle) -> bool:
    if sig.direction == "BUY":
        return candle.low <= sig.sl
    return candle.high >= sig.sl


def _tp_levels(sig: LifecycleSignal) -> list[tuple[str, float | None]]:
    return [
        ("TP1_HIT", sig.tp1),
        ("TP2_HIT", sig.tp2),
        ("TP3_HIT", sig.tp3),
    ]


def _tp_hit(sig: LifecycleSignal, candle: Candle, level: float) -> bool:
    if sig.direction == "BUY":
        return candle.high >= level
    return candle.low <= level


def _best_tp_reached(sig: LifecycleSignal, candle: Candle) -> str | None:
    best: str | None = None
    for name, level in _tp_levels(sig):
        if level is None:
            continue
        if _tp_hit(sig, candle, level):
            best = name
    return best


def _status_rank(status: str) -> int:
    order = ("ACTIVE", "TP1_HIT", "TP2_HIT", "TP3_HIT")
    try:
        return order.index(status)
    except ValueError:
        return -1


def _final_r_from_status(sig: LifecycleSignal, status: str,
                         candle: Candle) -> float:
    risk = _risk(sig)
    if status == "SL_HIT":
        return -1.0
    level_for: dict[str, float | None] = {
        "TP1_HIT": sig.tp1,
        "TP2_HIT": sig.tp2,
        "TP3_HIT": sig.tp3,
    }
    target = level_for.get(status)
    if target is not None:
        return abs(target - sig.entry) / risk
    # TIMEOUT/CANCELLED: realised PnL from candle close
    if sig.direction == "BUY":
        return (candle.close - sig.entry) / risk
    return (sig.entry - candle.close) / risk


def evaluate_candle(
    sig: LifecycleSignal,
    candle: Candle,
    *,
    now: datetime,
    ttl_hours: float,
) -> Transition | None:
    """Compute the next lifecycle state for `sig` after observing `candle`.

    Returns None when status would not change AND there is nothing
    material to persist. Returns a Transition otherwise (including the
    case where only max_favorable/adverse moved).
    """
    if sig.status in TERMINAL_STATUSES:
        return None

    fav, adv = _excursions(sig, candle)
    new_fav = max(sig.max_favorable_R, fav)
    new_adv = max(sig.max_adverse_R, adv)

    # SL first (conservative same-candle rule).
    if _sl_hit(sig, candle):
        return Transition(
            signal_id=sig.id,
            old_status=sig.status,
            new_status="SL_HIT",
            closed=True,
            closed_at=candle.ts,
            final_R=-1.0,
            max_favorable_R=new_fav,
            max_adverse_R=new_adv,
            price=sig.sl,
        )

    # Best TP reached on this candle (TP3 > TP2 > TP1).
    best_tp = _best_tp_reached(sig, candle)
    if best_tp is not None and _status_rank(best_tp) > _status_rank(sig.status):
        closed = best_tp == "TP3_HIT"
        return Transition(
            signal_id=sig.id,
            old_status=sig.status,
            new_status=best_tp,
            closed=closed,
            closed_at=candle.ts if closed else None,
            final_R=_final_r_from_status(sig, best_tp, candle) if closed else None,
            max_favorable_R=new_fav,
            max_adverse_R=new_adv,
            price={
                "TP1_HIT": sig.tp1, "TP2_HIT": sig.tp2, "TP3_HIT": sig.tp3,
            }[best_tp] or candle.close,
        )

    # Timeout check.
    age = now - sig.opened_at
    if age >= timedelta(hours=ttl_hours):
        return Transition(
            signal_id=sig.id,
            old_status=sig.status,
            new_status="TIMEOUT",
            closed=True,
            closed_at=now,
            final_R=_final_r_from_status(sig, "TIMEOUT", candle),
            max_favorable_R=new_fav,
            max_adverse_R=new_adv,
            price=candle.close,
        )

    # No status change but excursions may have moved.
    if new_fav != sig.max_favorable_R or new_adv != sig.max_adverse_R:
        return Transition(
            signal_id=sig.id,
            old_status=sig.status,
            new_status=sig.status,
            closed=False,
            closed_at=None,
            final_R=None,
            max_favorable_R=new_fav,
            max_adverse_R=new_adv,
            price=candle.close,
        )
    return None


def lifecycle_signal_from_row(row: dict[str, Any]) -> LifecycleSignal:
    opened_at = row["ts_utc"]
    if isinstance(opened_at, str):
        opened_at = datetime.fromisoformat(opened_at)
    return LifecycleSignal(
        id=int(row["id"]),
        stream=row.get("stream") or "intraday",
        direction=row["direction"],
        entry=float(row["entry"]),
        sl=float(row["sl"]),
        tp1=float(row["tp1"]) if row.get("tp1") is not None else None,
        tp2=float(row["tp2"]) if row.get("tp2") is not None else None,
        tp3=float(row["tp3"]) if row.get("tp3") is not None else None,
        status=row.get("status") or "ACTIVE",
        opened_at=opened_at,
        max_favorable_R=float(row.get("max_favorable_R") or 0.0),
        max_adverse_R=float(row.get("max_adverse_R") or 0.0),
        ai_action=row.get("ai_action"),
        ai_risk_label=row.get("ai_risk_label"),
        ai_model_name=row.get("ai_model_name"),
    )
