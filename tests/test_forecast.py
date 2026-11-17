import numpy as np
import pandas as pd
import pytest

from olist_forecast import (
    build_features,
    build_forecast_tables,
    cost,
    decision_costs,
    recommend_capacity,
    split_period,
)
from olist_forecast.decision import reserve_capacity
from olist_forecast.features import FEATURES


def test_history_is_the_same_table_as_build_features(series):
    history, _ = build_forecast_tables(series)
    pd.testing.assert_frame_equal(history, build_features(series))


def test_future_has_seven_days_per_category_after_the_last_day(series):
    _, future = build_forecast_tables(series)
    assert len(future) == 7 * series.shape[1]
    assert future["day"].min() == series.index.max() + pd.Timedelta(days=1)
    assert future["day"].max() == series.index.max() + pd.Timedelta(days=7)
    assert not future[FEATURES].isna().any().any()
    assert "units" not in future.columns


def test_future_features_use_only_known_days(series):
    _, future = build_forecast_tables(series)
    row = future[(future["category"] == "a") & (future["day"] == series.index.max() + pd.Timedelta(days=7))]
    assert row["lag_7"].iloc[0] == series["a"].iloc[-1]
    assert row["mean_7"].iloc[0] == pytest.approx(series["a"].iloc[-7:].mean())
    assert row["mean_28"].iloc[0] == pytest.approx(series["a"].iloc[-28:].mean())


def test_future_features_do_not_depend_on_the_placeholder_units(series):
    _, with_zeros = build_forecast_tables(series, fill=0)
    _, with_large = build_forecast_tables(series, fill=1_000_000)
    pd.testing.assert_frame_equal(with_zeros, with_large)


def test_future_lags_point_to_known_days_for_every_forecast_day(series):
    _, future = build_forecast_tables(series)
    for step in range(1, 8):
        day = series.index.max() + pd.Timedelta(days=step)
        row = future[(future["category"] == "b") & (future["day"] == day)].iloc[0]
        for lag in (7, 14, 21, 28):
            assert row[f"lag_{lag}"] == series["b"][day - pd.Timedelta(days=lag)]


def test_recommendation_has_one_row_per_category_and_day(series):
    out = recommend_capacity(series, cu=4, co=1)
    assert list(out.columns) == ["category", "day", "forecast", "band_low", "band_high", "capacity"]
    assert len(out) == 7 * series.shape[1]
    assert set(out["category"]) == set(series.columns)


def test_values_are_not_negative_and_the_band_contains_the_forecast(series):
    out = recommend_capacity(series, cu=4, co=1)
    assert (out[["forecast", "band_low", "band_high", "capacity"]] >= 0).all().all()
    assert (out["band_low"] <= out["band_high"]).all()


def test_higher_shortage_cost_reserves_more_capacity(series):
    low = recommend_capacity(series, cu=1, co=1)["capacity"].sum()
    mid = recommend_capacity(series, cu=4, co=1)["capacity"].sum()
    high = recommend_capacity(series, cu=9, co=1)["capacity"].sum()
    assert low < mid < high


def test_only_the_ratio_of_costs_matters(series):
    a = recommend_capacity(series, cu=4, co=1)
    b = recommend_capacity(series, cu=8, co=2)
    pd.testing.assert_frame_equal(a, b)


def test_the_last_known_day_changes_the_forecast(series):
    base = recommend_capacity(series, cu=4, co=1)
    changed = series.copy()
    changed.iloc[-1] = changed.iloc[-1] + 500
    assert not np.allclose(base["forecast"], recommend_capacity(changed, cu=4, co=1)["forecast"])


def test_costs_must_be_positive(series):
    with pytest.raises(ValueError, match="positive"):
        recommend_capacity(series, cu=0, co=1)


def test_series_that_is_too_short_is_rejected(series):
    with pytest.raises(ValueError, match="needs more than"):
        recommend_capacity(series.iloc[:70], cu=4, co=1)


def test_reserved_capacity_is_what_the_backtest_costs_are_computed_with(long):
    # the capacity used in the backtest and the one served by the API come from the same function
    tr, cal, te = split_period(long, 0, cal_days=56)
    capacity = reserve_capacity(tr, cal, te, 0.8)
    expected = decision_costs(long, 0, cu=4, co=1)["model"]
    assert cost(te["units"].to_numpy(), capacity, 4, 1) == pytest.approx(expected)
