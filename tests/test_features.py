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
    window = history[day - np.timedelta64(13, "D"): day - np.timedelta64(7, "D")]
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
