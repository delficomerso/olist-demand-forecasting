import numpy as np
import pytest

from olist_forecast import cost, decision_costs


def test_cost_counts_shortage_and_idle_capacity():
    y = np.array([10, 10])
    capacity = np.array([8, 13])
    # shortage 2 * cu + idle 3 * co
    assert cost(y, capacity, cu=4, co=1) == 4 * 2 + 1 * 3


def test_cost_is_zero_when_capacity_equals_demand():
    y = np.array([5, 6])
    assert cost(y, y, 4, 1) == 0


def test_decision_costs_returns_three_policies(long):
    out = decision_costs(long, 0, cu=4, co=1)
    assert set(out) == {"test_start", "baseline", "baseline_buffer", "model"}
    assert all(out[k] >= 0 for k in ("baseline", "baseline_buffer", "model"))


@pytest.mark.parametrize("cu", [2, 4, 9])
def test_buffer_helps_when_shortage_is_costlier(long, cu):
    total = {"baseline": 0.0, "baseline_buffer": 0.0}
    for k in range(3):
        out = decision_costs(long, k, cu=cu, co=1)
        for key in total:
            total[key] += out[key]
    # with shortage 2x or more costlier than idle capacity, a calibrated buffer should not hurt
    assert total["baseline_buffer"] <= total["baseline"] * 1.05


@pytest.mark.parametrize("cu, co", [(0, 1), (1, 0), (-1, 1)])
def test_costs_must_be_positive(long, cu, co):
    with pytest.raises(ValueError, match="positive"):
        decision_costs(long, 0, cu=cu, co=co)


@pytest.mark.parametrize("cu", [1, 2, 4, 9])
def test_best_capacity_is_the_critical_ratio_quantile(cu):
    # newsvendor result: with a known demand distribution, reserving the quantile
    # cu / (cu + co) of demand costs less than reserving a bit less or a bit more
    rng = np.random.default_rng(3)
    demand = rng.gamma(shape=5, scale=4, size=200_000)
    best = np.quantile(demand, cu / (cu + 1))
    for other in (best * 0.85, best * 1.15):
        assert cost(demand, best, cu, 1) < cost(demand, other, cu, 1)


def test_relative_widening_with_a_dead_category_is_rejected():
    import pandas as pd

    from olist_forecast import build_features

    days = pd.date_range("2018-01-01", periods=300, freq="D", name="day")
    flat = pd.DataFrame({"a": np.random.default_rng(2).poisson(20, 300), "dead": 0}, index=days)
    with pytest.raises(ValueError, match="positive 28-day mean"):
        decision_costs(build_features(flat), 0, cu=4, co=1, relative=True)
