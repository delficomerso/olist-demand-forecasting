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
