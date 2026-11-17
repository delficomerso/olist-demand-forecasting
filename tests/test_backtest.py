import pandas as pd

from olist_forecast import split_period, wape


def test_wape():
    actual = pd.Series([10, 10])
    assert wape(actual, pd.Series([8, 13])) == 0.25


def test_periods_do_not_overlap_and_train_precedes_calibration(long):
    tr, cal, te = split_period(long, 0, cal_days=56, test_days=56)
    assert te["day"].nunique() == 56
    assert cal["day"].nunique() == 56
    assert tr["day"].max() < cal["day"].min() <= cal["day"].max() < te["day"].min()
    assert te["day"].max() == long["day"].max()


def test_older_period_ends_before_newer_starts(long):
    _, _, recent = split_period(long, 0)
    _, _, older = split_period(long, 1)
    assert older["day"].max() < recent["day"].min()


def test_excluded_days_are_removed_from_training_only(long):
    exclude = (("2018-03-01", "2018-03-10"),)
    tr, _, te = split_period(long, 2, exclude=exclude)
    window = (tr["day"] >= "2018-03-01") & (tr["day"] <= "2018-03-10")
    assert not window.any()
    tr_all, _, _ = split_period(long, 2, exclude=())
    assert len(tr_all) > len(tr)


def test_wape_without_units_is_nan_not_infinite():
    import numpy as np

    assert np.isnan(wape(pd.Series([0, 0]), pd.Series([1, 1])))


def test_period_outside_the_data_is_rejected(long):
    import pytest

    with pytest.raises(ValueError, match="outside"):
        split_period(long, 50)


def test_period_without_training_rows_is_rejected(long):
    import pytest

    with pytest.raises(ValueError, match="no training rows"):
        split_period(long, 4)


def test_empty_calibration_window_is_rejected(long):
    import pytest

    # a period whose calibration window would fall before the data starts
    tr, _, te = split_period(long, 3)
    assert len(tr) > 0
    with pytest.raises(ValueError):
        split_period(long, 3, cal_days=400)


def test_black_friday_weeks_are_excluded_by_default():
    import numpy as np

    from olist_forecast import build_features
    from olist_forecast.backtest import BLACK_FRIDAY

    assert BLACK_FRIDAY == (("2017-11-24", "2017-12-29"),)
    days = pd.date_range("2017-09-01", periods=400, freq="D", name="day")
    series = pd.DataFrame({"a": np.random.default_rng(0).poisson(20, 400)}, index=days)
    data = build_features(series)
    tr, _, _ = split_period(data, 0)
    in_window = (tr["day"] >= "2017-11-24") & (tr["day"] <= "2017-12-29")
    assert len(tr) > 0 and not in_window.any()
    assert ((data["day"] >= "2017-11-24") & (data["day"] <= "2017-12-29")).any()


def test_calibration_window_with_no_rows_is_rejected(long):
    import pytest

    _, cal, _ = split_period(long, 0, cal_days=56)
    holes = long.drop(cal.index)
    with pytest.raises(ValueError, match="no calibration rows"):
        split_period(holes, 0, cal_days=56)
