import numpy as np

import pytest

from olist_forecast import build_features, calibrated_bounds, conformal_adjustment, split_period


def test_conformal_rank_uses_finite_sample_correction():
    scores = np.arange(1, 11)  # n = 10
    # ceil(11 * 0.8) / 10 = 0.9 -> quantile 0.9 of 1..10
    assert conformal_adjustment(scores, 0.8) == np.quantile(scores, 0.9)


def test_rank_is_capped_at_one():
    scores = np.array([1.0, 2.0, 3.0])
    assert conformal_adjustment(scores, 0.99) == 3.0


def test_bounds_shape_and_order(long):
    te, lo, hi, adjustment = calibrated_bounds(long, 0)
    assert len(te) == len(lo) == len(hi) == len(adjustment)
    assert (hi >= lo).mean() > 0.95


def test_relative_adjustment_is_proportional_to_level(long):
    te, _, _, adjustment = calibrated_bounds(long, 0, relative=True)
    ratio = adjustment / te["mean_28"].to_numpy()
    assert np.allclose(ratio, ratio[0])


def test_fixed_adjustment_is_one_number(long):
    _, _, _, adjustment = calibrated_bounds(long, 0, relative=False)
    assert np.allclose(adjustment, adjustment[0])


def test_calibrated_band_covers_close_to_target(long):
    covers = []
    for k in range(3):
        te, lo, hi, adjustment = calibrated_bounds(long, k)
        y = te["units"].to_numpy()
        covers.append(((y >= lo - adjustment) & (y <= hi + adjustment)).mean())
    assert 0.6 < np.mean(covers) < 0.95


def test_empty_calibration_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="calibration rows"):
        conformal_adjustment(np.array([]), 0.8)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_level_must_be_between_zero_and_one(level):
    with pytest.raises(ValueError, match="between 0 and 1"):
        conformal_adjustment(np.arange(10), level)


def test_finite_sample_coverage_guarantee():
    # for exchangeable scores, a new score falls below the adjustment with probability
    # ceil((n + 1) * level) / (n + 1), which is at least the nominal level
    rng = np.random.default_rng(1)
    n, level, hits, trials = 50, 0.8, 0, 4000
    for _ in range(trials):
        scores = rng.normal(size=n)
        hits += rng.normal() <= conformal_adjustment(scores, level)
    assert level - 0.02 <= hits / trials <= level + 0.05


def test_bounds_do_not_depend_on_test_actuals(long):
    # fitting and calibration must not read the units of the test period
    changed = long.copy()
    _, _, te = split_period(long, 0)
    changed.loc[te.index, "units"] = 10_000
    original = calibrated_bounds(long, 0)
    perturbed = calibrated_bounds(changed, 0)
    for a, b in zip(original[1:], perturbed[1:]):
        np.testing.assert_allclose(a, b)


def test_relative_widening_rejects_zero_level_rows():
    import pandas as pd

    days = pd.date_range("2018-01-01", periods=300, freq="D", name="day")
    rng = np.random.default_rng(2)
    flat = pd.DataFrame({"a": rng.poisson(20, 300), "dead": 0}, index=days)
    with pytest.raises(ValueError, match="positive 28-day mean"):
        calibrated_bounds(build_features(flat), 0, relative=True)
