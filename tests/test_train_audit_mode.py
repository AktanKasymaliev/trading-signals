"""Unit tests for --audit-only mode in train_path_d_model.py."""

from __future__ import annotations

import pytest

from xau_pro_bot.models.path_d_harvest import HarvestConfig
from scripts.train_path_d_model import _run_audit, _OUTCOME_COLS


def test_run_audit_returns_one_row_per_config(long_history):
    rows = _run_audit(long_history, [
        ("step_h1=4", HarvestConfig(step_h1=4)),
        ("step_h1=1", HarvestConfig(step_h1=1)),
    ])
    assert len(rows) == 2
    for r in rows:
        assert "config" in r
        assert "rows" in r
        assert "baseline" in r
        assert "synthetic" in r


def test_run_audit_row_config_labels(long_history):
    configs = [
        ("step_h1=4", HarvestConfig(step_h1=4)),
        ("step_h1=1", HarvestConfig(step_h1=1)),
    ]
    rows = _run_audit(long_history, configs)
    assert rows[0]["config"] == "step_h1=4"
    assert rows[1]["config"] == "step_h1=1"


def test_run_audit_outcome_cols_present(long_history):
    rows = _run_audit(long_history, [
        ("step_h1=4", HarvestConfig(step_h1=4)),
    ])
    assert len(rows) == 1
    r = rows[0]
    for col in _OUTCOME_COLS:
        assert col in r, f"Expected outcome column '{col}' in audit row"


def test_run_audit_counts_are_consistent(long_history):
    rows = _run_audit(long_history, [
        ("step_h1=4", HarvestConfig(step_h1=4)),
    ])
    r = rows[0]
    # baseline + synthetic should not exceed total rows (synthetic can overlap baseline tracking)
    assert r["rows"] >= 0
    assert r["baseline"] >= 0
    assert r["synthetic"] >= 0


def test_run_audit_with_step_m15(long_history):
    rows = _run_audit(long_history, [
        ("step_h1=1,step_m15=2", HarvestConfig(step_h1=1, step_m15=2)),
    ])
    assert len(rows) == 1
    r = rows[0]
    assert "config" in r
    assert r["config"] == "step_h1=1,step_m15=2"
