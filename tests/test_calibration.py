import numpy as np

from olist_forecast import calibrated_bounds, conformal_adjustment


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
