"""Rolling backtest windows and the WAPE metric."""

import pandas as pd

# Black Friday weeks of 2017, excluded from training (a single event the model cannot learn)
BLACK_FRIDAY = (("2017-11-24", "2017-12-29"),)


def wape(actual, pred):
    """Total absolute error divided by total units."""
    return (actual - pred).abs().sum() / actual.sum()


def split_period(data, k, cal_days=0, test_days=56, exclude=BLACK_FRIDAY):
    """Train, calibration and test rows of backtest period k, where 0 is the most recent.

    Each test period has `test_days` days. Training stops before the calibration window, which
    sits right before the test period, and never includes the days in `exclude`.
    """
    last_day = data["day"].max()
    test_end = last_day - pd.Timedelta(days=test_days * k)
    test_begin = test_end - pd.Timedelta(days=test_days - 1)
    cal_begin = test_begin - pd.Timedelta(days=cal_days)

    excluded = pd.Series(False, index=data.index)
    for first, last in exclude:
        excluded |= (data["day"] >= first) & (data["day"] <= last)

    tr = data[(data["day"] < cal_begin) & ~excluded]
    cal = data[(data["day"] >= cal_begin) & (data["day"] < test_begin)]
    te = data[(data["day"] >= test_begin) & (data["day"] <= test_end)]
    return tr, cal, te
