import pandas as pd

from olist_forecast.data import _pivot_top_categories


def test_pivot_keeps_top_categories_and_fills_missing_days_with_zero():
    daily = pd.DataFrame({
        "day": ["2018-01-01", "2018-01-03", "2018-01-03", "2018-01-02"],
        "category": ["a", "a", "b", "c"],
        "units": [5, 7, 2, 1],
    })
    out = _pivot_top_categories(daily, "2018-01-01", "2018-01-06", top_n=2,
                                keep_from="2018-01-01", keep_to="2018-01-05")
    assert list(out.columns) == ["a", "b"]
    assert out.loc["2018-01-02", "a"] == 0
    assert out.loc["2018-01-03", "b"] == 2
    assert len(out) == 5
    assert out.index.name == "day"


def test_pivot_trims_to_requested_range():
    daily = pd.DataFrame({"day": ["2018-01-01"], "category": ["a"], "units": [1]})
    out = _pivot_top_categories(daily, "2018-01-01", "2018-01-11", 1, "2018-01-03", "2018-01-08")
    assert out.index.min() == pd.Timestamp("2018-01-03")
    assert out.index.max() == pd.Timestamp("2018-01-08")
