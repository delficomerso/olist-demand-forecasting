"""Forecast of the next 7 days and recommended capacity, from the series up to the last known day."""

import pandas as pd

from .backtest import BLACK_FRIDAY
from .calibration import band
from .decision import reserve_capacity
from .features import FEATURES, build_forecast_tables
from .models import VERY_SMALL, fit_point


def recommend_capacity(
    series,
    cu,
    co,
    params=VERY_SMALL,
    cal_days=56,
    relative=False,
    features=FEATURES,
    exclude=BLACK_FRIDAY,
):
    """Forecast of the next 7 days per category with a calibrated band and the capacity to reserve.

    The procedure is the one validated in the backtest. The models are trained on the days before
    the last `cal_days` days (never on the excluded days), the widening is calibrated on those
    last `cal_days` days, and the forecast covers the 7 days after the last known day.

    Returns one row per category and day with the median forecast, the calibrated 10-90 band and
    the capacity that minimizes the expected cost when a missing unit costs `cu` and an idle unit
    costs `co`. Values are clipped at zero, since demand and capacity cannot be negative.
    """
    if cu <= 0 or co <= 0:
        raise ValueError("cu and co must be positive")
    history, future = build_forecast_tables(series)

    last_day = history["day"].max()
    cal_begin = last_day - pd.Timedelta(days=cal_days - 1)
    excluded = pd.Series(False, index=history.index)
    for first, last in exclude:
        excluded |= (history["day"] >= first) & (history["day"] <= last)
    train = history[(history["day"] < cal_begin) & ~excluded]
    cal = history[history["day"] >= cal_begin]
    if train.empty or len(cal) < cal_days:
        raise ValueError(f"The series needs more than {cal_days} days after the first usable day")

    level = cu / (cu + co)
    point = fit_point(train, features, params).predict(future[features])
    low, high, adjustment = band(train, cal, future, params, relative=relative, features=features)
    capacity = reserve_capacity(train, cal, future, level, params, relative, features)

    out = pd.DataFrame(
        {
            "category": future["category"].astype(str),
            "day": future["day"],
            "forecast": point,
            "band_low": low - adjustment,
            "band_high": high + adjustment,
            "capacity": capacity,
        }
    )
    value_columns = ["forecast", "band_low", "band_high", "capacity"]
    out[value_columns] = out[value_columns].clip(lower=0)
    return out.sort_values(["category", "day"]).reset_index(drop=True)
