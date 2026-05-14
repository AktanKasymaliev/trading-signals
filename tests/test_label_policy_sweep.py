"""Tests for the per-policy label sweep in train_path_d_model."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from xau_pro_bot.models.label_policy import LabelPolicy
from scripts.train_path_d_model import _run_label_policy_sweep


def _fake_metrics() -> dict:
    return {
        "predicts_only_bad": False,
        "confusion_matrix": [[5, 5], [5, 5]],
        "precision_macro": 0.5,
        "recall_macro": 0.5,
        "n_train": 14,
        "n_val": 3,
        "n_test": 3,
        "feature_cols": ["feat_a"],
        "report": "",
    }


def test_sweep_calls_train_filter_once_per_policy(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_train_filter(df: pd.DataFrame, *, policy: str, **kw):
        calls.append(policy)
        return None, _fake_metrics()

    monkeypatch.setattr("scripts.train_path_d_model.train_filter", fake_train_filter)

    minimal_df = pd.DataFrame({"baseline_sample": [True] * 5})
    _run_label_policy_sweep(minimal_df, tmp_path)

    expected = {p.value for p in LabelPolicy}
    assert set(calls) == expected, f"Expected {expected}, got {set(calls)}"


def test_sweep_emits_json_with_entry_per_policy(monkeypatch, tmp_path):
    def fake_train_filter(df: pd.DataFrame, *, policy: str, **kw):
        return None, _fake_metrics()

    monkeypatch.setattr("scripts.train_path_d_model.train_filter", fake_train_filter)

    minimal_df = pd.DataFrame({"baseline_sample": [True] * 5})
    result = _run_label_policy_sweep(minimal_df, tmp_path)

    sweep_file = tmp_path / "path_d_filter_policy_sweep.json"
    assert sweep_file.exists(), "JSON output file not created"

    data = json.loads(sweep_file.read_text())
    expected = {p.value for p in LabelPolicy}
    assert set(data.keys()) == expected


def test_sweep_json_entries_have_required_fields(monkeypatch, tmp_path):
    def fake_train_filter(df: pd.DataFrame, *, policy: str, **kw):
        return None, _fake_metrics()

    monkeypatch.setattr("scripts.train_path_d_model.train_filter", fake_train_filter)

    minimal_df = pd.DataFrame({"baseline_sample": [True] * 5})
    _run_label_policy_sweep(minimal_df, tmp_path)

    data = json.loads((tmp_path / "path_d_filter_policy_sweep.json").read_text())
    required_fields = {
        "n", "class_balance", "good_prob_stats",
        "precision", "recall", "confusion_matrix",
        "predicts_only_bad", "degenerate",
    }
    for policy_val, entry in data.items():
        missing = required_fields - set(entry.keys())
        assert not missing, f"Policy {policy_val!r} missing fields: {missing}"


def test_sweep_degenerate_flag_set_for_all_bad_model(monkeypatch, tmp_path):
    """A model predicting only BAD should be flagged as degenerate."""

    def fake_train_filter(df: pd.DataFrame, *, policy: str, **kw):
        metrics = _fake_metrics()
        metrics["predicts_only_bad"] = True
        metrics["confusion_matrix"] = [[10, 0], [5, 0]]
        return None, metrics

    monkeypatch.setattr("scripts.train_path_d_model.train_filter", fake_train_filter)

    minimal_df = pd.DataFrame({"baseline_sample": [True] * 5})
    _run_label_policy_sweep(minimal_df, tmp_path)

    data = json.loads((tmp_path / "path_d_filter_policy_sweep.json").read_text())
    for policy_val, entry in data.items():
        assert entry["degenerate"] is True, (
            f"Policy {policy_val!r} should be degenerate but degenerate={entry['degenerate']}"
        )


def test_sweep_degenerate_false_for_healthy_model(monkeypatch, tmp_path):
    """A healthy model should not be flagged as degenerate."""

    def fake_train_filter(df: pd.DataFrame, *, policy: str, **kw):
        metrics = _fake_metrics()
        # confusion_matrix with enough GOOD predictions (kept_pct > 5%)
        metrics["confusion_matrix"] = [[90, 10], [20, 20]]
        metrics["predicts_only_bad"] = False
        return None, metrics

    monkeypatch.setattr("scripts.train_path_d_model.train_filter", fake_train_filter)

    minimal_df = pd.DataFrame({"baseline_sample": [True] * 5})
    _run_label_policy_sweep(minimal_df, tmp_path)

    data = json.loads((tmp_path / "path_d_filter_policy_sweep.json").read_text())
    for policy_val, entry in data.items():
        assert entry["degenerate"] is False, (
            f"Policy {policy_val!r} flagged degenerate unexpectedly"
        )


def test_sweep_continues_on_train_filter_exception(monkeypatch, tmp_path):
    """If one policy raises, the sweep should still process all remaining policies."""
    call_count = 0

    def fake_train_filter(df: pd.DataFrame, *, policy: str, **kw):
        nonlocal call_count
        call_count += 1
        if policy == "tp1_unresolved_bad":
            raise RuntimeError("simulated failure")
        return None, _fake_metrics()

    monkeypatch.setattr("scripts.train_path_d_model.train_filter", fake_train_filter)

    minimal_df = pd.DataFrame({"baseline_sample": [True] * 5})
    result = _run_label_policy_sweep(minimal_df, tmp_path)

    # All policies should still appear in the result
    expected = {p.value for p in LabelPolicy}
    assert set(result.keys()) == expected
    # The failed one should have an "error" key
    assert "error" in result["tp1_unresolved_bad"]
    # The rest should have normal fields
    assert "degenerate" in result["tp1_unresolved_drop"]
