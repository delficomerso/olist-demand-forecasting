"""Newsvendor capacity decision: reserve the quantile cu / (cu + co) of demand."""

import numpy as np

from .backtest import split_period
from .calibration import conformal_adjustment
from .features import FEATURES
from .models import VERY_SMALL, fit_quantile


def cost(y, capacity, cu, co):
    """Total cost of reserving the given capacity when demand turns out to be y.

    cu is the cost per unit of demand above capacity, co the cost per unit of idle capacity.
    """
    return (cu * np.maximum(y - capacity, 0) + co * np.maximum(capacity - y, 0)).sum()


def decision_costs(data, k, cu, co, params=VERY_SMALL, cal_days=56, relative=False,
                   features=FEATURES, test_days=56):
    """Cost of three reservation policies in backtest period k.

    baseline is the mean of the last four weekly lags, baseline_buffer adds a calibrated buffer
    to it, and model uses the calibrated quantile forecast. With relative=True both calibrated
    policies use a widening proportional to the 28-day mean.
    """
    if cu <= 0 or co <= 0:
        raise ValueError("cu and co must be positive")
    level = cu / (cu + co)
    tr, cal, te = split_period(data, k, cal_days, test_days)
    model = fit_quantile(tr, features, level, params)

    scale_cal = cal["mean_28"].to_numpy() if relative else 1.0
    scale_te = te["mean_28"].to_numpy() if relative else np.ones(len(te))
    if relative and ((scale_cal == 0).any() or (scale_te == 0).any()):
        raise ValueError("relative widening needs a positive 28-day mean in every row")

    y_cal = cal["units"].to_numpy()
    y = te["units"].to_numpy()

    capacity_baseline = te["baseline"].to_numpy()
    buffer = conformal_adjustment((y_cal - cal["baseline"].to_numpy()) / scale_cal, level) * scale_te
    adjustment = conformal_adjustment((y_cal - model.predict(cal[features])) / scale_cal, level) * scale_te
    capacity_model = model.predict(te[features]) + adjustment

    return {
        "test_start": te["day"].min().date(),
        "baseline": cost(y, capacity_baseline, cu, co),
        "baseline_buffer": cost(y, capacity_baseline + buffer, cu, co),
        "model": cost(y, capacity_model, cu, co),
    }
