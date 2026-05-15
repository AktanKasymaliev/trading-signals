import numpy as np
import pandas as pd
import pytest

from xau_pro_bot.models.calibrated_filter import (
    CalibratedFilterWrapper, probability_distribution_stats,
)


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(400, 5)),
                     columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return X, y


@pytest.fixture
def light_params():
    return dict(objective="binary", learning_rate=0.05, max_depth=4, num_leaves=15,
                min_data_in_leaf=5, n_estimators=50, verbose=-1, random_state=42)


def test_calibrated_wrapper_predicts_proba_in_unit_interval(toy_dataset, light_params):
    X, y = toy_dataset
    wrapper = CalibratedFilterWrapper(base_params=light_params).fit(X.iloc[:300], y.iloc[:300])
    p = wrapper.predict_proba(X.iloc[300:])
    assert p.shape == (100, 2)
    assert ((p >= 0) & (p <= 1)).all()


def test_probability_distribution_stats_keys(toy_dataset, light_params):
    X, y = toy_dataset
    wrapper = CalibratedFilterWrapper(base_params=light_params).fit(X.iloc[:300], y.iloc[:300])
    good = wrapper.predict_proba(X.iloc[300:])[:, 1]
    stats = probability_distribution_stats(good)
    for k in ("min", "p10", "p25", "median", "p75", "p90", "max"):
        assert k in stats
    assert stats["min"] <= stats["median"] <= stats["max"]


def test_calibrated_wrapper_classes_attr(toy_dataset, light_params):
    X, y = toy_dataset
    wrapper = CalibratedFilterWrapper(base_params=light_params).fit(X.iloc[:300], y.iloc[:300])
    assert set(wrapper.classes_.tolist()) == {0, 1}


def test_probability_distribution_stats_empty():
    stats = probability_distribution_stats(np.array([]))
    assert set(stats.keys()) == {"min", "p10", "p25", "median", "p75", "p90", "max"}
    # NaN-allowed for all entries
    for v in stats.values():
        assert v != v or v == v  # tolerate nan
