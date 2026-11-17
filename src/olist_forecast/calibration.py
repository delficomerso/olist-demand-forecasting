"""Conformal calibration of quantile bands (CQR style) on a window before the test period."""

import numpy as np

from .backtest import split_period
from .features import FEATURES
from .models import VERY_SMALL, fit_quantile


def conformal_adjustment(scores, level):
    """Quantile of the calibration scores with the finite-sample rank ceil((n + 1) * level) / n."""
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calibrate without calibration rows")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    return np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n))


def calibrated_bounds(
    data, k, params=VERY_SMALL, cal_days=56, level=0.8, relative=False, features=FEATURES, test_days=56
):
    """Band for backtest period k. Returns the test rows, the 10-90 bounds and the widening of each row.

    With relative=True the widening is proportional to the 28-day mean of each row, so a
    category with more units gets a wider adjustment.
    """
    tr, cal, te = split_period(data, k, cal_days, test_days)
    low, high = fit_quantile(tr, features, 0.1, params), fit_quantile(tr, features, 0.9, params)

    scale_cal = cal["mean_28"].to_numpy() if relative else 1.0
    scale_te = te["mean_28"].to_numpy() if relative else np.ones(len(te))
    if relative and ((scale_cal == 0).any() or (scale_te == 0).any()):
        raise ValueError("relative widening needs a positive 28-day mean in every row")

    y_cal = cal["units"].to_numpy()
    score = np.maximum(low.predict(cal[features]) - y_cal, y_cal - high.predict(cal[features])) / scale_cal
    adjustment = conformal_adjustment(score, level) * scale_te

    return te, low.predict(te[features]), high.predict(te[features]), adjustment
