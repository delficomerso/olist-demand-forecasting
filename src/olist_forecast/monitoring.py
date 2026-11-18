"""Replay of retraining and monitoring over past weeks. It is a simulation on past data.

For each week of a window the forecast is made with the model that would have been in use at
that time, and its error, the coverage of its band and the cost of its decision are measured
against what really happened. Three ways of keeping the model up to date are compared, never
retraining it, retraining every four weeks and retraining every week.
"""

import numpy as np
import pandas as pd

from .backtest import BLACK_FRIDAY, wape
from .calibration import band
from .decision import buffered_baseline, cost, reserve_capacity
from .features import FEATURES
from .models import VERY_SMALL, fit_point

BAND_LEVEL = 0.8  # share of days that the 10-90 band is calibrated to contain
STRATEGIES = {"never": None, "every_4_weeks": 4, "weekly": 1}


def default_window(long, periods=3, test_days=56):
    """First and last day of the last `periods` backtest periods, as used in the backtest."""
    last_day = long["day"].max()
    return last_day - pd.Timedelta(days=periods * test_days - 1), last_day


def _windows(long, retrain_day, cal_days, exclude):
    """Training and calibration rows of a model retrained on `retrain_day`.

    The calibration window is the `cal_days` days before the retraining day and the model is
    trained on the days before that window, never on the excluded days.
    """
    cal_begin = retrain_day - pd.Timedelta(days=cal_days)
    excluded = pd.Series(False, index=long.index)
    for first, last in exclude:
        excluded |= (long["day"] >= first) & (long["day"] <= last)
    train = long[(long["day"] < cal_begin) & ~excluded]
    cal = long[(long["day"] >= cal_begin) & (long["day"] < retrain_day)]
    if train.empty:
        raise ValueError(f"No training rows before the calibration window of {retrain_day.date()}")
    if len(cal) < cal_days:
        raise ValueError(f"The calibration window of {retrain_day.date()} is missing days")
    return train, cal


def replay_strategy(
    long,
    cu,
    co,
    retrain_every=None,
    start=None,
    end=None,
    params=VERY_SMALL,
    cal_days=56,
    relative=False,
    features=FEATURES,
    exclude=BLACK_FRIDAY,
    wape_ratio=1.25,
    min_coverage=0.70,
):
    """Week by week results of one way of keeping the model up to date.

    The window is split in weeks that start on `start`. The model is retrained on the first week
    and then every `retrain_every` weeks, or never again when it is None. A model retrained on
    day r is trained on the days before r - cal_days and calibrated on the cal_days days before r,
    the same procedure as the backtest, and it forecasts the days from r up to the next
    retraining, so its age in weeks grows between retrainings.

    An alert is raised in a week when the error of the model is more than `wape_ratio` times its
    error on its own calibration window, or when the coverage of the band is below `min_coverage`.
    Returns one row per week.
    """
    if cu <= 0 or co <= 0:
        raise ValueError("cu and co must be positive")
    if retrain_every is not None and retrain_every < 1:
        raise ValueError("retrain_every must be at least 1 week, or None to never retrain")
    first_day, last_day = default_window(long)
    start = pd.Timestamp(start) if start is not None else first_day
    end = pd.Timestamp(end) if end is not None else last_day
    n_weeks = ((end - start).days + 1) // 7
    if n_weeks < 1:
        raise ValueError("The window needs at least one full week")
    level = cu / (cu + co)
    step = n_weeks if retrain_every is None else retrain_every

    rows = []
    for block_first in range(0, n_weeks, step):
        retrain_day = start + pd.Timedelta(weeks=block_first)
        block_weeks = min(step, n_weeks - block_first)
        block_end = retrain_day + pd.Timedelta(days=7 * block_weeks - 1)
        train, cal = _windows(long, retrain_day, cal_days, exclude)
        target = long[(long["day"] >= retrain_day) & (long["day"] <= block_end)]
        if target.empty:
            raise ValueError(f"No rows to forecast from {retrain_day.date()}")

        point_model = fit_point(train, features, params)
        reference = wape(cal["units"], point_model.predict(cal[features]))
        point = point_model.predict(target[features])
        low, high, adjustment = band(train, cal, target, params, BAND_LEVEL, relative, features)
        capacity = reserve_capacity(train, cal, target, level, params, relative, features)
        capacity_buffer = buffered_baseline(cal, target, level, relative)

        y = target["units"].to_numpy()
        days = target["day"].to_numpy()
        baseline = target["baseline"].to_numpy()
        inside = (y >= low - adjustment) & (y <= high + adjustment)
        for week in range(block_weeks):
            week_start = retrain_day + pd.Timedelta(weeks=week)
            mask = (days >= week_start.to_datetime64()) & (
                days < (week_start + pd.Timedelta(weeks=1)).to_datetime64()
            )
            rows.append(
                {
                    "week_start": week_start,
                    "retrained_on": retrain_day,
                    "model_age_weeks": week,
                    "units": y[mask].sum(),
                    "abs_error_model": np.abs(y[mask] - point[mask]).sum(),
                    "abs_error_baseline": np.abs(y[mask] - baseline[mask]).sum(),
                    "n": int(mask.sum()),
                    "inside": int(inside[mask].sum()),
                    "cost_model": cost(y[mask], capacity[mask], cu, co),
                    "cost_baseline_buffer": cost(y[mask], capacity_buffer[mask], cu, co),
                    "reference_wape": reference,
                }
            )
    out = pd.DataFrame(rows)
    out["wape_model"] = out["abs_error_model"] / out["units"].where(out["units"] > 0)
    out["wape_baseline"] = out["abs_error_baseline"] / out["units"].where(out["units"] > 0)
    out["coverage"] = out["inside"] / out["n"]
    out["alert_error"] = out["wape_model"] > wape_ratio * out["reference_wape"]
    out["alert_coverage"] = out["coverage"] < min_coverage
    out["alert"] = out["alert_error"] | out["alert_coverage"]
    return out


def replay(long, cu, co, strategies=None, **kwargs):
    """Week by week results of several strategies, in one table with a `strategy` column.

    `strategies` maps a name to the number of weeks between retrainings, or None for never.
    The other arguments are those of replay_strategy.
    """
    strategies = STRATEGIES if strategies is None else strategies
    parts = [
        replay_strategy(long, cu, co, retrain_every=every, **kwargs).assign(strategy=name)
        for name, every in strategies.items()
    ]
    return pd.concat(parts, ignore_index=True)


def summarize(results):
    """One row per strategy with totals over the whole window.

    The error and the coverage are computed from the sums of the weeks, so a week with more
    units weighs more, as in the backtest. The saving is that of the model against the baseline
    with a calibrated buffer.
    """
    grouped = results.groupby("strategy", sort=False)
    total = grouped[
        [
            "units",
            "abs_error_model",
            "abs_error_baseline",
            "n",
            "inside",
            "cost_model",
            "cost_baseline_buffer",
        ]
    ].sum()
    return pd.DataFrame(
        {
            "retrainings": grouped["retrained_on"].nunique(),
            "wape_model": total["abs_error_model"] / total["units"],
            "wape_baseline": total["abs_error_baseline"] / total["units"],
            "coverage": total["inside"] / total["n"],
            "cost_model": total["cost_model"],
            "cost_baseline_buffer": total["cost_baseline_buffer"],
            "saving_vs_buffer": 1 - total["cost_model"] / total["cost_baseline_buffer"],
            "alert_weeks": grouped["alert"].sum(),
            "weeks": grouped.size(),
        }
    )
