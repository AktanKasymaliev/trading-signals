"""Harvest training samples for Path D.

Walks history H1-bar by H1-bar, asks baseline MasterSignalEngine for a
setup, resolves the TP/SL outcome on M15 future bars, and optionally
appends synthetic ATR-based NO_TRADE samples for Mode A2.

The output is a flat DataFrame: one row per (cutoff, sample) with
features + labels + outcome bookkeeping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xau_pro_bot.models.features import build_ai_features
from xau_pro_bot.models.trade_outcome import (
    OutcomeClass,
    resolve_outcome_m15,
)
from xau_pro_bot.signals.engine import MasterSignalEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarvestConfig:
    step_h1: int = 4
    timeout_m15: int = 192
    label_tp_target: str = "tp1"
    include_synthetic: bool = False
    synth_stride: int = 8
    synth_atr_sl: float = 1.5
    synth_rr: float = 2.0
    min_lookback_h1: int = 250


_KILLZONES = ("Asian KZ", "London KZ", "NY AM KZ", "NY PM KZ", "OFF")


def _killzone_onehot(label: str | None) -> dict[str, int]:
    label = label if label in _KILLZONES else "OFF"
    return {f"kz_{k.replace(' ', '_')}": int(label == k) for k in _KILLZONES}


def _atr(series: pd.DataFrame, n: int = 14) -> float:
    high = series["High"]; low = series["Low"]; close = series["Close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(n).mean().iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _baseline_context_features(sig: dict, m15: pd.DataFrame,
                               h1: pd.DataFrame) -> dict:
    bull = float(sig.get("bull_score", 0.0))
    bear = float(sig.get("bear_score", 0.0))
    tier = sig.get("tier", "NO_SIGNAL")
    direction = sig.get("direction", "BUY")
    atr_h1 = _atr(h1.tail(50))
    atr_pct = float((h1["High"] - h1["Low"]).tail(100).rank(pct=True).iloc[-1])
    range_m15 = float(m15["High"].iloc[-1] - m15["Low"].iloc[-1])
    range_vs_atr = range_m15 / atr_h1 if atr_h1 > 0 else 0.0
    ts = m15.index[-1].tz_convert("America/New_York")
    return {
        "bull_score": bull,
        "bear_score": bear,
        "score_gap": abs(bull - bear),
        "final_score": float(sig.get("score", 0.0)),
        "tier_WEAK":   int(tier == "WEAK"),
        "tier_NORMAL": int(tier == "NORMAL"),
        "tier_STRONG": int(tier == "STRONG"),
        "tier_NO_SIGNAL": int(tier == "NO_SIGNAL"),
        "dir_BUY":  int(direction == "BUY"),
        "dir_SELL": int(direction == "SELL"),
        "rr": float(sig.get("rr") or 0.0),
        "hour_ny": float(ts.hour),
        "day_of_week": float(ts.dayofweek),
        "atr_percentile_h1": atr_pct,
        "range_vs_atr_m15": range_vs_atr,
        **_killzone_onehot(sig.get("killzone")),
    }


def _directional_label(direction: str, outcome_class: OutcomeClass) -> int:
    if outcome_class == OutcomeClass.TP:
        return 1 if direction == "BUY" else -1
    return 0


def _filter_label(outcome_class: OutcomeClass,
                  unresolved_policy: str = "bad") -> int:
    if outcome_class == OutcomeClass.TP:
        return 1
    if outcome_class == OutcomeClass.UNRESOLVED:
        return 0 if unresolved_policy == "bad" else 1
    return 0


def harvest_path_d_samples(history: dict[str, pd.DataFrame],
                           cfg: HarvestConfig = HarvestConfig(),
                           ) -> pd.DataFrame:
    h1 = history["H1"]; m15 = history["M15"]
    if len(h1) < cfg.min_lookback_h1:
        return pd.DataFrame()

    engine = MasterSignalEngine(ai_enabled=False)
    rows: list[dict] = []
    step_count = 0

    for i in range(cfg.min_lookback_h1, len(h1) - 1, cfg.step_h1):
        cutoff = h1.index[i]
        slice_data = {tf: df.loc[:cutoff].tail(720) for tf, df in history.items()}
        step_count += 1
        try:
            sig = engine.analyze(slice_data)
        except Exception:
            continue
        if sig is None:
            continue

        m15_future = m15.loc[m15.index > cutoff]
        if len(m15_future) < 10:
            break

        try:
            feats_29, complete = build_ai_features(slice_data)
        except Exception:
            continue
        if not complete:
            continue
        feats_29_row = feats_29.iloc[0].to_dict()

        base_ctx = _baseline_context_features(sig, slice_data["M15"], slice_data["H1"])

        tier = sig.get("tier", "NO_SIGNAL")
        tp = (sig.get("tp1") if cfg.label_tp_target == "tp1"
              else (sig.get("tp2") or sig.get("tp1")))
        if tier in {"WEAK", "NORMAL", "STRONG"} and tp is not None and sig.get("sl") is not None:
            try:
                out = resolve_outcome_m15(
                    entry=float(sig["entry"]), sl=float(sig["sl"]),
                    tp=float(tp), direction=str(sig["direction"]),
                    m15_future=m15_future, timeout_bars=cfg.timeout_m15,
                )
            except ValueError:
                continue
            rows.append({
                **feats_29_row, **base_ctx,
                "is_synthetic": 0,
                "baseline_sample": True,
                "entry": float(sig["entry"]),
                "sl": float(sig["sl"]),
                "tp_used": float(tp),
                "direction": sig["direction"],
                "tier": tier,
                "outcome_class": out.outcome_class.value,
                "final_R": out.final_R,
                "mfe_R": out.mfe_R,
                "mae_R": out.mae_R,
                "bars_to_outcome": out.bars_to_outcome,
                "label_directional": _directional_label(sig["direction"], out.outcome_class),
                "label_filter": _filter_label(out.outcome_class),
                "cutoff": cutoff,
            })
        if cfg.include_synthetic and (step_count % cfg.synth_stride == 0):
            atr = _atr(slice_data["H1"].tail(50))
            if atr <= 0:
                continue
            entry = float(slice_data["M15"]["Close"].iloc[-1])
            for direction in ("BUY", "SELL"):
                if direction == "BUY":
                    sl = entry - cfg.synth_atr_sl * atr
                    tp = entry + cfg.synth_atr_sl * cfg.synth_rr * atr
                else:
                    sl = entry + cfg.synth_atr_sl * atr
                    tp = entry - cfg.synth_atr_sl * cfg.synth_rr * atr
                try:
                    out = resolve_outcome_m15(entry, sl, tp, direction,
                                              m15_future, cfg.timeout_m15)
                except ValueError:
                    continue
                synth_ctx = dict(base_ctx)
                synth_ctx["dir_BUY"]  = int(direction == "BUY")
                synth_ctx["dir_SELL"] = int(direction == "SELL")
                rows.append({
                    **feats_29_row, **synth_ctx,
                    "is_synthetic": 1,
                    "baseline_sample": False,
                    "entry": entry, "sl": sl, "tp_used": tp,
                    "direction": direction,
                    "tier": "NO_SIGNAL",
                    "outcome_class": out.outcome_class.value,
                    "final_R": out.final_R,
                    "mfe_R": out.mfe_R,
                    "mae_R": out.mae_R,
                    "bars_to_outcome": out.bars_to_outcome,
                    "label_directional": _directional_label(direction, out.outcome_class),
                    "label_filter": np.nan,
                    "cutoff": cutoff,
                })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("cutoff").sort_index()
    return df
