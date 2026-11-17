"""Feature table: lags and rolling means that are known `horizon` days ahead."""

import pandas as pd

FEATURES = ["category", "lag_7", "lag_14", "lag_21", "lag_28", "mean_7", "mean_28", "dow"]
LAG_COLUMNS = ["lag_7", "lag_14", "lag_21", "lag_28"]
MIN_HISTORY = 34  # the first usable row is the 35th day (28-day mean shifted 7 days)


def build_features(series, horizon=7):
    """Long table (day, category, units, features, baseline) from a day x category series.

    Lags are `horizon`, `horizon + 7`, ... so that only data known at forecast time is used.
    The columns keep the names lag_7 ... lag_28 and are valid for horizon=7, which is the
    horizon studied here. A larger horizon needs a larger first lag, so it raises for now.
    """
    if horizon != 7:
        raise NotImplementedError("Feature names and lags are defined for horizon 7")
    if series.isna().any().any():
        raise ValueError("The series has missing values; fill missing days with 0 first")
    expected = pd.date_range(series.index.min(), series.index.max(), freq="D")
    if len(series) != len(expected) or not series.index.equals(expected):
        raise ValueError("The series must have one row per day with no gaps, because lags are taken by row")
    if len(series) <= MIN_HISTORY:
        raise ValueError(f"The series needs more than {MIN_HISTORY} days to build the features")

    long = (
        series.rename_axis(columns="category")
        .stack()
        .rename("units")
        .reset_index()
        .sort_values(["category", "day"])
        .reset_index(drop=True)
    )
    g = long.groupby("category")["units"]
    for lag in (7, 14, 21, 28):
        long[f"lag_{lag}"] = g.shift(lag)
    long["mean_7"] = g.transform(lambda s: s.shift(horizon).rolling(7).mean())
    long["mean_28"] = g.transform(lambda s: s.shift(horizon).rolling(28).mean())
    long["dow"] = long["day"].dt.day_of_week
    long["category"] = long["category"].astype("category")
    long = long.dropna().reset_index(drop=True)
    long["baseline"] = long[LAG_COLUMNS].mean(axis=1)
    return long


def build_forecast_tables(series, horizon=7, fill=0):
    """Feature tables for the known history and for the next `horizon` days.

    Returns (history, future). History is the same table as build_features(series). Future has
    one row per category and day after the last one, with the same feature columns. Every lag is
    at least `horizon` days old, so the units of the future days are never used: they are filled
    with `fill` only to build the rows and are dropped from the result.
    """
    last_day = series.index.max()
    future_days = pd.date_range(last_day + pd.Timedelta(days=1), periods=horizon, freq="D", name="day")
    placeholder = pd.DataFrame(fill, index=future_days, columns=series.columns)
    long = build_features(pd.concat([series, placeholder]), horizon)
    history = long[long["day"] <= last_day].reset_index(drop=True)
    future = long[long["day"] > last_day].drop(columns=["units"]).reset_index(drop=True)
    return history, future
