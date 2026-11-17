import numpy as np
import pandas as pd
import pytest

from olist_forecast import build_features


@pytest.fixture(scope="session")
def series():
    """Synthetic daily demand: 3 categories, weekly pattern plus noise, 300 days."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2018-01-01", periods=300, freq="D")
    base = {"a": 40, "b": 15, "c": 5}
    data = {
        name: rng.poisson(level * (1 + 0.3 * np.sin(2 * np.pi * days.day_of_week / 7)))
        for name, level in base.items()
    }
    return pd.DataFrame(data, index=pd.Index(days, name="day"))


@pytest.fixture(scope="session")
def long(series):
    return build_features(series)
