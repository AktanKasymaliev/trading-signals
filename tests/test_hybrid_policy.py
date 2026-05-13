from __future__ import annotations

import pytest

from xau_pro_bot.models.trade_filter_model import FilterDecision
from xau_pro_bot.signals.hybrid_policy import (
    HybridDecision,
    HybridThresholds,
    decide,
)


T = HybridThresholds(weak=0.70, normal=0.55, strong_block=0.80,
                      directional_conflict=0.65)


def _filter(good_prob: float, decision=None):
    return {
        "good_prob": good_prob,
        "bad_prob": 1.0 - good_prob,
        "decision": decision or (FilterDecision.KEEP if good_prob >= 0.5
                                  else FilterDecision.BLOCK),
        "threshold_used": 0.55,
        "error": None,
    }


def test_no_signal_passthrough():
    d = decide(tier="NO_SIGNAL", baseline_dir="BUY",
               ai_directional=None, ai_filter=None, thresholds=T)
    assert d == HybridDecision.KEEP


def test_strong_keep_by_default():
    d = decide("STRONG", "BUY", None, _filter(0.5), T)
    assert d == HybridDecision.KEEP


def test_strong_blocked_only_when_filter_very_confident_bad():
    d = decide("STRONG", "BUY", None, _filter(0.10), T)
    assert d == HybridDecision.BLOCK


def test_normal_requires_filter_approval():
    assert decide("NORMAL", "BUY", None, _filter(0.40), T) == HybridDecision.BLOCK
    assert decide("NORMAL", "BUY", None, _filter(0.60), T) == HybridDecision.KEEP


def test_weak_high_bar():
    assert decide("WEAK", "BUY", None, _filter(0.60), T) == HybridDecision.BLOCK
    assert decide("WEAK", "BUY", None, _filter(0.75), T) == HybridDecision.KEEP


def test_directional_conflict_blocks_normal():
    ai = {"direction": "SELL", "confidence": 0.70}
    assert decide("NORMAL", "BUY", ai, _filter(0.80), T) == HybridDecision.BLOCK


def test_directional_low_confidence_does_not_block():
    ai = {"direction": "SELL", "confidence": 0.50}
    assert decide("NORMAL", "BUY", ai, _filter(0.80), T) == HybridDecision.KEEP


def test_works_without_filter():
    assert decide("NORMAL", "BUY", None, None, T) == HybridDecision.KEEP
    assert decide("WEAK",   "BUY", None, None, T) == HybridDecision.KEEP
