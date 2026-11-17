import numpy as np

from olist_forecast import split_period
from olist_forecast.features import FEATURES
from olist_forecast.models import VERY_SMALL, fit_point, fit_quantile


def test_default_parameters_are_the_ones_of_the_notebook():
    assert VERY_SMALL == dict(
        objective="l1",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=4,
        min_child_samples=60,
        random_state=42,
        verbose=-1,
    )


def test_quantile_models_are_ordered_and_roughly_at_their_level(long):
    train, _, _ = split_period(long, 0)
    y = train["units"].to_numpy()
    below = {}
    preds = {}
    for alpha in (0.1, 0.5, 0.9):
        preds[alpha] = fit_quantile(train, FEATURES, alpha).predict(train[FEATURES])
        below[alpha] = (y <= preds[alpha]).mean()
    assert preds[0.1].mean() < preds[0.5].mean() < preds[0.9].mean()
    assert below[0.1] < below[0.5] < below[0.9]
    assert below[0.9] > 0.75 and below[0.1] < 0.3


def test_fitting_does_not_change_the_default_parameters(long):
    before = dict(VERY_SMALL)
    train, _, _ = split_period(long, 0)
    fit_quantile(train, FEATURES, 0.9)
    fit_point(train, FEATURES)
    assert VERY_SMALL == before


def test_point_model_is_deterministic(long):
    train, _, te = split_period(long, 0)
    a = fit_point(train, FEATURES).predict(te[FEATURES])
    b = fit_point(train, FEATURES).predict(te[FEATURES])
    np.testing.assert_array_equal(a, b)
