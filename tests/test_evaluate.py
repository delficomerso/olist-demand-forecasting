import pytest

from olist_forecast import evaluate_config


def test_metrics_cover_every_period_and_means(long):
    out = evaluate_config(long)
    for k in (1, 2, 3):
        assert f"period_{k}_wape_model" in out
        assert f"period_{k}_cost_baseline_buffer" in out
    assert out["mean_wape_model"] == pytest.approx(
        sum(out[f"period_{k}_wape_model"] for k in (1, 2, 3)) / 3
    )


def test_relative_option_changes_calibrated_coverage_only_through_widening(long):
    fixed, scaled = evaluate_config(long), evaluate_config(long, relative=True)
    assert fixed["mean_wape_model"] == scaled["mean_wape_model"]
    assert fixed["mean_coverage_raw"] == scaled["mean_coverage_raw"]


def test_values_are_plain_floats(long):
    assert all(isinstance(v, float) for v in evaluate_config(long).values())


def test_metrics_match_direct_computation(long):
    from olist_forecast import split_period, wape

    out = evaluate_config(long)
    _, _, te = split_period(long, 0)
    assert out["period_1_wape_baseline"] == pytest.approx(wape(te["units"], te["baseline"]))


def test_number_of_periods_is_respected(long):
    out = evaluate_config(long, n_periods=1)
    assert "period_1_wape_model" in out
    assert "period_2_wape_model" not in out
