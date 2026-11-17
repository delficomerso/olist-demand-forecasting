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
