"""Pure hybrid-mode policy combining baseline tier, directional model,
and filter model into KEEP/BLOCK. No I/O, no globals — easy to unit-test."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class HybridDecision(str, enum.Enum):
    KEEP = "KEEP"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class HybridThresholds:
    weak: float = 0.70
    normal: float = 0.55
    strong_block: float = 0.80
    directional_conflict: float = 0.65


def decide(tier: str, baseline_dir: str,
           ai_directional: dict | None,
           ai_filter: dict | None,
           thresholds: HybridThresholds) -> HybridDecision:
    if tier == "NO_SIGNAL":
        return HybridDecision.KEEP

    if (ai_directional and
            ai_directional.get("direction") and
            ai_directional["direction"] != baseline_dir and
            float(ai_directional.get("confidence", 0.0)) > thresholds.directional_conflict):
        return HybridDecision.BLOCK

    if tier == "STRONG":
        if ai_filter and ai_filter.get("bad_prob") is not None:
            if float(ai_filter["bad_prob"]) >= thresholds.strong_block:
                return HybridDecision.BLOCK
        return HybridDecision.KEEP

    if tier == "NORMAL":
        if ai_filter and ai_filter.get("good_prob") is not None:
            if float(ai_filter["good_prob"]) < thresholds.normal:
                return HybridDecision.BLOCK
        return HybridDecision.KEEP

    if tier == "WEAK":
        if ai_filter and ai_filter.get("good_prob") is not None:
            if float(ai_filter["good_prob"]) < thresholds.weak:
                return HybridDecision.BLOCK
        return HybridDecision.KEEP

    return HybridDecision.KEEP
