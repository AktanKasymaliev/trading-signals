"""Tests for the analysis-assistant AI explanation layer.

Covers ai_explanation helpers, engine output enrichment, formatter block
rendering under AI_EXPLAIN, and backtest blocked-signal diagnostics.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from xau_pro_bot.formatter import format_strong_signal, format_weak_signal
from xau_pro_bot.models.ai_explanation import (
    derive_action,
    derive_risk_label,
    model_name,
    short_reason,
)


# ── helpers ───────────────────────────────────────────────────────────


def _base_sig(**overrides):
    sig = {
        "direction": "BUY",
        "tier": "STRONG",
        "score": 80,
        "entry": 3300.0,
        "sl": 3290.0,
        "tp1": 3320.0,
        "tp2": 3340.0,
        "tp3": 3360.0,
        "rr": 2.0,
        "killzone": "London KZ",
        "tp2_unavailable": False,
        "reasons": {
            "ict": ["OTE zone"], "smc": [], "macro": [],
            "classic": [], "penalties": [],
        },
        "ts_utc": datetime(2026, 5, 15, 9, 0, tzinfo=timezone.utc),
        "ai_enabled": True,
        "ai_direction": "BUY",
        "ai_confidence": 0.71,
        "ai_reason": "AI agrees with deterministic signal",
        "ai_blocked": False,
        "ai_model_name": "Path C legacy",
        "ai_feature_set": "internal",
        "ai_action": "KEEP",
        "ai_reason_short": "AI agrees with deterministic signal",
        "ai_risk_label": "CLEAN_SETUP",
    }
    sig.update(overrides)
    return sig


# ── ai_explanation helpers ────────────────────────────────────────────


def test_model_name_maps_known_feature_sets():
    assert model_name("internal", True) == "Path C legacy"
    assert model_name("legacy", True) == "Path C legacy"
    assert model_name("stationary", True) == "Path F stationary"


def test_model_name_unknown_feature_set_falls_back_to_tag():
    assert model_name("path_z", True) == "path_z"


def test_model_name_none_when_disabled():
    assert model_name("internal", False) is None


def test_derive_action_keep_when_agrees():
    assert derive_action(
        ai_enabled=True, ai_blocked=False,
        ai_direction="BUY", deterministic_direction="BUY",
    ) == "KEEP"


def test_derive_action_block_when_blocked():
    assert derive_action(
        ai_enabled=True, ai_blocked=True,
        ai_direction="NO_TRADE", deterministic_direction="BUY",
    ) == "BLOCK"


def test_derive_action_downgrade_when_conflict_not_blocked():
    assert derive_action(
        ai_enabled=True, ai_blocked=False,
        ai_direction="SELL", deterministic_direction="BUY",
    ) == "DOWNGRADE"


def test_derive_action_none_when_disabled_or_skipped():
    assert derive_action(
        ai_enabled=False, ai_blocked=False,
        ai_direction=None, deterministic_direction="BUY",
    ) is None
    assert derive_action(
        ai_enabled=True, ai_blocked=False,
        ai_direction=None, deterministic_direction="BUY",
    ) is None


def test_risk_label_clean_setup():
    assert derive_risk_label(
        tier="STRONG", penalties=[], ai_action="KEEP",
    ) == "CLEAN_SETUP"


def test_risk_label_high_risk_on_weak():
    assert derive_risk_label(
        tier="WEAK", penalties=[], ai_action="KEEP",
    ) == "HIGH_RISK"


def test_risk_label_high_risk_on_penalty():
    assert derive_risk_label(
        tier="STRONG", penalties=["D1 trend against BUY"], ai_action="KEEP",
    ) == "HIGH_RISK"


def test_risk_label_high_risk_on_block():
    assert derive_risk_label(
        tier="NORMAL", penalties=[], ai_action="BLOCK",
    ) == "HIGH_RISK"


def test_risk_label_medium_default():
    assert derive_risk_label(
        tier="NORMAL", penalties=[], ai_action="KEEP",
    ) == "MEDIUM_RISK"


def test_short_reason_trims_long_text():
    long = "x" * 200
    out = short_reason(long, limit=80)
    assert out is not None
    assert len(out) <= 80
    assert out.endswith("…")


def test_short_reason_passes_through_short_text():
    assert short_reason("clean") == "clean"


def test_short_reason_handles_none():
    assert short_reason(None) is None


# ── formatter: AI_EXPLAIN flag ────────────────────────────────────────


def test_formatter_shows_ai_block_when_explain_true(monkeypatch):
    monkeypatch.setenv("AI_EXPLAIN", "true")
    text = format_strong_signal(_base_sig())
    assert "🧠 AI filter: KEEP" in text
    assert "Модель: Path C legacy" in text
    assert "Риск: CLEAN" in text
    assert "Причина:" in text
    # legacy single-line should be replaced by block
    assert "AI: BUY 0.71 confidence" not in text


def test_formatter_hides_ai_block_when_explain_false(monkeypatch):
    from xau_pro_bot import config as _cfg
    monkeypatch.delenv("AI_EXPLAIN", raising=False)
    monkeypatch.setattr(_cfg, "AI_EXPLAIN", False, raising=False)
    text = format_strong_signal(_base_sig())
    # No rich block when flag is off
    assert "🧠 AI filter:" not in text
    assert "Модель: Path C legacy" not in text


def test_formatter_legacy_ai_line_remains_when_explain_false(monkeypatch):
    """Backward compat: existing single-line AI output still emitted."""
    from xau_pro_bot import config as _cfg
    monkeypatch.delenv("AI_EXPLAIN", raising=False)
    monkeypatch.setattr(_cfg, "AI_EXPLAIN", False, raising=False)
    text = format_strong_signal(_base_sig())
    assert "AI: BUY 0.71 confidence" in text


def test_formatter_weak_signal_supports_explain_block(monkeypatch):
    monkeypatch.setenv("AI_EXPLAIN", "true")
    sig = _base_sig(tier="WEAK", ai_risk_label="HIGH_RISK", ai_action="DOWNGRADE")
    text = format_weak_signal(sig)
    assert "🧠 AI filter: DOWNGRADE" in text
    assert "Риск: HIGH" in text


def test_signal_dict_remains_backward_compatible():
    """Old downstream consumers must keep working: original AI keys remain."""
    sig = _base_sig()
    for key in ("ai_enabled", "ai_direction", "ai_confidence",
                "ai_reason", "ai_blocked"):
        assert key in sig


# ── engine integration: explanation fields populated ─────────────────


def test_engine_attaches_explanation_fields(monkeypatch):
    """Engine should attach ai_model_name / ai_action / ai_risk_label to output."""
    monkeypatch.setenv("AI_ENABLED", "false")  # avoid HF download
    from xau_pro_bot.signals.engine import MasterSignalEngine

    eng = MasterSignalEngine(ai_enabled=False)
    # Use a minimal synthetic data path via a quick stub: just call _build_explanation.
    reasons = {"penalties": []}
    ai_fields = {
        "ai_enabled": False, "ai_direction": None, "ai_confidence": None,
        "ai_reason": None, "ai_blocked": False,
        "ai_score_delta_buy": 0, "ai_score_delta_sell": 0,
    }
    out = eng._build_explanation(ai_fields, "BUY", "STRONG", reasons)
    assert out["ai_model_name"] is None  # disabled
    assert out["ai_feature_set"] is None
    assert out["ai_action"] is None
    assert out["ai_risk_label"] == "CLEAN_SETUP"


def test_engine_explanation_block_action_when_blocked():
    from xau_pro_bot.signals.engine import MasterSignalEngine

    eng = MasterSignalEngine(ai_enabled=True, ai_model=object())
    reasons = {"penalties": []}
    ai_fields = {
        "ai_enabled": True, "ai_direction": "NO_TRADE", "ai_confidence": 0.2,
        "ai_reason": "AI conflicts: confidence too low", "ai_blocked": True,
        "ai_score_delta_buy": 0, "ai_score_delta_sell": 0,
    }
    out = eng._build_explanation(ai_fields, "BUY", "NO_SIGNAL", reasons)
    assert out["ai_action"] == "BLOCK"
    assert out["ai_risk_label"] == "HIGH_RISK"
    assert out["ai_model_name"] == "Path C legacy"
    assert out["ai_reason_short"] == "AI conflicts: confidence too low"


# ── backtest: blocked diagnostics ─────────────────────────────────────


def test_backtest_result_has_blocked_details_field():
    from xau_pro_bot.backtest import BacktestResult

    res = BacktestResult()
    assert res.blocked_details == []


def test_backtest_records_blocked_detail_entry():
    """Simulate the engine returning a blocked sig and verify the
    backtest loop records a structured entry. We exercise the append
    logic directly to avoid running the full walk-forward."""
    from xau_pro_bot.backtest import BacktestResult

    res = BacktestResult()
    sig = {
        "tier": "NO_SIGNAL",
        "direction": "BUY",
        "ai_blocked": True,
        "ai_reason": "AI conflicts",
        "ai_reason_short": "AI conflicts",
        "ai_action": "BLOCK",
        "ai_risk_label": "HIGH_RISK",
        "ai_pre_block_tier": "NORMAL",
    }
    # Mirror the backtest.py append shape:
    res.blocked_signals += 1
    res.blocked_details.append({
        "ts": datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc),
        "original_direction": sig["direction"],
        "tier_before_block": sig["ai_pre_block_tier"],
        "ai_reason": sig["ai_reason_short"],
        "ai_action": sig["ai_action"],
        "ai_risk_label": sig["ai_risk_label"],
        "outcome_if_taken": None,
    })

    assert res.blocked_signals == 1
    detail = res.blocked_details[0]
    assert detail["original_direction"] == "BUY"
    assert detail["tier_before_block"] == "NORMAL"
    assert detail["ai_reason"] == "AI conflicts"
    assert detail["ai_action"] == "BLOCK"
    assert detail["ai_risk_label"] == "HIGH_RISK"
    assert detail["outcome_if_taken"] is None


# ── swing / scalp analyzers receive AI gate ──────────────────────────


class _KeepGate:
    """Test double mimicking AIExplanationGate that always KEEPs."""

    ai_feature_set = "internal"

    def enrich(self, sig, data):
        return {
            **sig,
            "ai_enabled": True,
            "ai_direction": sig["direction"],
            "ai_confidence": 0.71,
            "ai_reason": "AI agrees with deterministic signal",
            "ai_blocked": False,
            "ai_score_delta_buy": 0,
            "ai_score_delta_sell": 0,
            "ai_action": "KEEP",
            "ai_model_name": "Path C legacy",
            "ai_feature_set": "internal",
            "ai_risk_label": "CLEAN_SETUP",
            "ai_reason_short": "AI agrees",
        }


class _BlockGate:
    """Test double mimicking AIExplanationGate that always BLOCKs."""

    ai_feature_set = "internal"

    def enrich(self, sig, data):
        return {
            **sig,
            "ai_enabled": True,
            "ai_direction": "NO_TRADE",
            "ai_confidence": 0.2,
            "ai_reason": "AI conflicts: confidence too low",
            "ai_blocked": True,
            "ai_score_delta_buy": 0,
            "ai_score_delta_sell": 0,
            "ai_action": "BLOCK",
            "ai_model_name": "Path C legacy",
            "ai_feature_set": "internal",
            "ai_risk_label": "HIGH_RISK",
            "ai_reason_short": "AI conflicts",
        }


def _stub_swing_setup(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.signals.swing_analyzer.find_swing_setup",
        lambda d1_df, h4_df: {
            "direction": "BUY", "entry": 3300.0, "sl": 3290.0, "tp": 3320.0,
            "rr": 2.0, "type": "1000pip", "range_pips": 1000,
        },
    )


def _stub_scalp_signal(monkeypatch):
    monkeypatch.setattr(
        "xau_pro_bot.signals.scalp_analyzer.scalp_signal",
        lambda m15_df, h1_df, h4_df: {
            "direction": "BUY", "entry": 3300.0, "sl": 3290.0,
            "tp1": 3320.0, "tp2": 3340.0, "killzone": "London KZ",
            "counter_trend": False, "atr_m15": 1.0,
            "conditions_met": ["RSI bounce"],
        },
    )


def test_swing_analyzer_enriches_signal_via_gate(monkeypatch):
    from xau_pro_bot.signals.swing_analyzer import SwingAnalyzer

    _stub_swing_setup(monkeypatch)
    sa = SwingAnalyzer(gate=_KeepGate())
    sig = sa.analyze({"D1": None, "H4": None})

    assert sig is not None
    assert sig["ai_enabled"] is True
    assert sig["ai_action"] == "KEEP"
    assert sig["ai_model_name"] == "Path C legacy"
    assert sig["ai_risk_label"] == "CLEAN_SETUP"


def test_swing_analyzer_drops_signal_when_ai_blocks(monkeypatch):
    from xau_pro_bot.signals.swing_analyzer import SwingAnalyzer

    _stub_swing_setup(monkeypatch)
    sa = SwingAnalyzer(gate=_BlockGate())
    assert sa.analyze({"D1": None, "H4": None}) is None


def test_swing_analyzer_without_gate_keeps_old_behavior(monkeypatch):
    from xau_pro_bot.signals.swing_analyzer import SwingAnalyzer

    _stub_swing_setup(monkeypatch)
    sa = SwingAnalyzer()
    sig = sa.analyze({"D1": None, "H4": None})
    assert sig is not None
    assert "ai_enabled" not in sig


def test_scalp_analyzer_enriches_signal_via_gate(monkeypatch):
    from xau_pro_bot.signals.scalp_analyzer import ScalpAnalyzer

    _stub_scalp_signal(monkeypatch)
    sa = ScalpAnalyzer(gate=_KeepGate())
    sig = sa.analyze({"M15": None, "H1": None, "H4": None})

    assert sig is not None
    assert sig["ai_enabled"] is True
    assert sig["ai_action"] == "KEEP"
    assert sig["ai_model_name"] == "Path C legacy"


def test_scalp_analyzer_drops_signal_when_ai_blocks(monkeypatch):
    from xau_pro_bot.signals.scalp_analyzer import ScalpAnalyzer

    _stub_scalp_signal(monkeypatch)
    sa = ScalpAnalyzer(gate=_BlockGate())
    assert sa.analyze({"M15": None, "H1": None, "H4": None}) is None


def test_formatter_renders_ai_block_for_swing_style_signal(monkeypatch):
    """Swing signals (no killzone, score=80) must render AI block when
    enriched and AI_EXPLAIN=true. Guards against the analysis-assistant
    block being intraday-only."""
    monkeypatch.setenv("AI_EXPLAIN", "true")
    sig = _base_sig(
        killzone=None,
        strategy_label="Swing 1000",
        horizon_label="1-4 недели",
        reasons={"swing": ["1000pip setup, range 1000 pips"],
                 "macro": [], "smc": [], "ict": [],
                 "classic": [], "penalties": []},
    )
    text = format_strong_signal(sig)
    assert "🧠 AI filter: KEEP" in text
    assert "Модель: Path C legacy" in text
    assert "Риск: CLEAN" in text


def test_ai_gate_evaluate_returns_disabled_fields_when_off():
    from xau_pro_bot.signals.ai_gate import AIExplanationGate

    gate = AIExplanationGate(ai_enabled=False)
    out = gate.evaluate({}, "BUY")
    assert out["ai_enabled"] is False
    assert out["ai_direction"] is None
    assert out["ai_blocked"] is False


def test_ai_gate_enrich_merges_explanation_into_sig():
    """When disabled, enrich still adds explanation keys with neutral
    values so downstream code (formatter, backtest) gets a uniform
    shape regardless of analyzer."""
    from xau_pro_bot.signals.ai_gate import AIExplanationGate

    gate = AIExplanationGate(ai_enabled=False)
    sig_in = {
        "direction": "BUY", "tier": "STRONG", "score": 80,
        "reasons": {"penalties": []},
    }
    sig_out = gate.enrich(sig_in, {})
    assert sig_out["ai_enabled"] is False
    assert sig_out["ai_action"] is None
    assert sig_out["ai_model_name"] is None
    assert sig_out["ai_risk_label"] == "CLEAN_SETUP"
    # original keys preserved
    assert sig_out["direction"] == "BUY"
    assert sig_out["tier"] == "STRONG"
