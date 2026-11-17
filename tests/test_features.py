import numpy as np
import pytest

from olist_forecast import build_features
from olist_forecast.features import FEATURES


def test_features_only_use_data_known_seven_days_ahead(series):
    long = build_features(series)
    row = long[(long["category"] == "a")].iloc[40]
    day = row["day"]
    history = series["a"]
    assert row["lag_7"] == history[day - np.timedelta64(7, "D")]
    assert row["lag_28"] == history[day - np.timedelta64(28, "D")]
    window = history[day - np.timedelta64(13, "D") : day - np.timedelta64(7, "D")]
    assert row["mean_7"] == pytest.approx(window.mean())


def test_features_have_no_missing_values_and_expected_columns(long):
    assert not long[FEATURES + ["units", "baseline"]].isna().any().any()
    assert set(FEATURES) <= set(long.columns)


def test_baseline_is_mean_of_four_lags(long):
    expected = long[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    assert np.allclose(long["baseline"], expected)


def test_other_horizons_are_rejected(series):
    with pytest.raises(NotImplementedError):
        build_features(series, horizon=14)


def test_lags_do_not_mix_categories():
    import pandas as pd

    days = pd.date_range("2018-01-01", periods=60, freq="D", name="day")
    series = pd.DataFrame({"big": 100, "small": 1}, index=days)
    long = build_features(series)
    for col in ["lag_7", "lag_14", "lag_21", "lag_28", "mean_7", "mean_28", "baseline"]:
        assert (long.loc[long["category"] == "big", col] == 100).all()
        assert (long.loc[long["category"] == "small", col] == 1).all()


def test_gap_in_the_dates_is_rejected(series):
    # lags are taken by row, so a missing day would silently shift every lag after it
    with pytest.raises(ValueError, match="no gaps"):
        build_features(series.drop(series.index[100:103]))


def test_missing_values_are_rejected(series):
    broken = series.copy().astype(float)
    broken.iloc[50, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        build_features(broken)


def test_series_too_short_is_rejected(series):
    with pytest.raises(ValueError, match="more than"):
        build_features(series.iloc[:30])


def test_shortest_valid_series_gives_one_row_per_category(series):
    out = build_features(series.iloc[:35])
    assert len(out) == series.shape[1]
