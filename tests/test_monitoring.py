import pandas as pd
import pytest

from olist_forecast import build_features, decision_costs
from olist_forecast.backtest import split_period
from olist_forecast.monitoring import STRATEGIES, default_window, replay, replay_strategy, summarize


@pytest.fixture(scope="module")
def results(long):
    return replay(long, 4, 1)


def test_default_window_covers_three_backtest_periods(long):
    first, last = default_window(long)
    assert last == long["day"].max()
    assert (last - first).days + 1 == 168


def test_every_strategy_covers_the_same_24_weeks(results):
    assert set(results["strategy"]) == set(STRATEGIES)
    for _, part in results.groupby("strategy"):
        assert len(part) == 24
        assert part["week_start"].is_monotonic_increasing
        assert (part["n"] == 7 * 3).all()  # 7 days and 3 categories in the fixture


def test_retraining_schedule_and_model_age(results):
    by = {name: part for name, part in results.groupby("strategy")}
    assert by["never"]["retrained_on"].nunique() == 1
    assert by["never"]["model_age_weeks"].tolist() == list(range(24))
    assert by["every_4_weeks"]["retrained_on"].nunique() == 6
    assert by["every_4_weeks"]["model_age_weeks"].max() == 3
    assert by["weekly"]["retrained_on"].nunique() == 24
    assert (by["weekly"]["model_age_weeks"] == 0).all()


def test_never_retraining_over_one_period_matches_the_backtest(long):
    _, _, test = split_period(long, 0, 56)
    out = replay_strategy(long, 4, 1, None, start=test["day"].min(), end=test["day"].max())
    expected = decision_costs(long, 0, 4, 1)
    assert out["cost_model"].sum() == pytest.approx(expected["model"])
    assert out["cost_baseline_buffer"].sum() == pytest.approx(expected["baseline_buffer"])


def test_weeks_add_up_to_the_totals(long, results):
    out = summarize(results)
    part = results[results["strategy"] == "weekly"]
    assert out.loc["weekly", "wape_model"] == pytest.approx(
        part["abs_error_model"].sum() / part["units"].sum()
    )
    assert out.loc["weekly", "coverage"] == pytest.approx(part["inside"].sum() / part["n"].sum())
    assert out.loc["weekly", "retrainings"] == 24
    assert out.loc["weekly", "weeks"] == 24


def test_alert_thresholds_work_in_both_directions(long):
    always = replay_strategy(long, 4, 1, None, wape_ratio=0.0, min_coverage=0.0)
    never = replay_strategy(long, 4, 1, None, wape_ratio=1e9, min_coverage=0.0)
    assert always["alert_error"].all()
    assert not never["alert"].any()
    low_coverage = replay_strategy(long, 4, 1, None, wape_ratio=1e9, min_coverage=1.1)
    assert low_coverage["alert_coverage"].all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cu": 0, "co": 1},
        {"cu": 4, "co": -1},
        {"cu": 4, "co": 1, "retrain_every": 0},
    ],
)
def test_invalid_arguments_are_rejected(long, kwargs):
    with pytest.raises(ValueError):
        replay_strategy(long, **kwargs)


def test_window_shorter_than_a_week_is_rejected(long):
    start = long["day"].max() - pd.Timedelta(days=3)
    with pytest.raises(ValueError, match="full week"):
        replay_strategy(long, 4, 1, None, start=start)


def test_window_without_training_data_is_rejected(long):
    with pytest.raises(ValueError, match="training"):
        replay_strategy(long, 4, 1, None, start=long["day"].min() + pd.Timedelta(days=30))


def test_works_on_a_series_longer_than_the_fixture(series):
    long = build_features(
        pd.concat([series, series]).set_axis(pd.date_range("2017-01-01", periods=600, freq="D", name="day"))
    )
    out = replay_strategy(long, 4, 1, 4)
    assert len(out) == 24
