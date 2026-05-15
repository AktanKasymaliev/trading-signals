from scripts.eval_path_d import THRESHOLDS, pick_best_threshold


def test_thresholds_cover_low_range():
    assert 0.20 in THRESHOLDS
    assert 0.25 in THRESHOLDS
    assert 0.30 in THRESHOLDS
    assert 0.45 in THRESHOLDS
    assert 0.55 in THRESHOLDS
    assert 0.60 in THRESHOLDS
    assert max(THRESHOLDS) <= 0.60


def test_pick_best_threshold_prefers_higher_pf_subject_to_min_kept():
    sweep = {
        0.20: {"pf": 1.50, "kept": 5,  "expectancy": 0.10, "wr": 0.40, "blocked": 100},
        0.30: {"pf": 1.20, "kept": 40, "expectancy": 0.15, "wr": 0.55, "blocked": 60},
        0.40: {"pf": 1.10, "kept": 70, "expectancy": 0.08, "wr": 0.50, "blocked": 30},
    }
    assert pick_best_threshold(sweep, min_kept=25) == 0.30
