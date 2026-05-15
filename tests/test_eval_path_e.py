"""Path E mode runs end-to-end with a stub regressor bundle."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from scripts.eval_path_d import run_all_modes


class _Reg:
    def predict(self, X):
        return np.asarray(X["bull_score"]) - 0.4


def test_path_e_mode_in_results(long_history, tmp_path):
    bundle = tmp_path / "e.joblib"
    joblib.dump({"model": _Reg(), "feature_cols": ["bull_score"]}, bundle)

    payload = run_all_modes(
        long_history,
        path_c_local=None,
        path_d_filter=None,
        path_e=str(bundle),
    )
    res = payload["results"]
    # Sweep recorded with the prescribed thresholds.
    sweep = payload["expected_r_sweep"]
    assert set(sweep.keys()) == {0.00, 0.03, 0.05, 0.10, 0.15}
    # If a chosen threshold exists, L mode and chosen threshold are recorded.
    if payload["chosen_expected_r_threshold"] is not None:
        assert payload["chosen_expected_r_threshold"] in sweep
        assert "L_path_e_expected_r" in res


def test_report_contains_path_e_section(long_history, tmp_path):
    from scripts.eval_path_d import run_all_modes, write_report
    bundle = tmp_path / "e.joblib"
    joblib.dump({"model": _Reg(), "feature_cols": ["bull_score"]}, bundle)
    payload = run_all_modes(long_history, path_c_local=None,
                            path_d_filter=None, path_e=str(bundle))
    out = tmp_path / "report.md"
    write_report(payload, out)
    text = out.read_text()
    assert "Path E (expected_R)" in text
    assert "Chosen threshold" in text
