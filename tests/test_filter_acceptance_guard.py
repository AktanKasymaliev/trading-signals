import pytest
from scripts.train_path_d_model import _acceptance_guard


def test_guard_rejects_all_bad_predictions():
    metrics = {"predicts_only_bad": True, "confusion_matrix": [[100, 0], [40, 0]]}
    with pytest.raises(SystemExit):
        _acceptance_guard(metrics, min_kept_pct=0.05)


def test_guard_rejects_zero_kept():
    metrics = {"predicts_only_bad": False, "confusion_matrix": [[100, 0], [40, 0]]}
    with pytest.raises(SystemExit):
        _acceptance_guard(metrics, min_kept_pct=0.05)


def test_guard_passes_with_some_kept():
    metrics = {"predicts_only_bad": False, "confusion_matrix": [[90, 10], [20, 20]]}
    _acceptance_guard(metrics, min_kept_pct=0.05)  # should not raise
