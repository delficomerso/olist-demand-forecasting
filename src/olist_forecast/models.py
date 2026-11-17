"""LightGBM models. WAPE is minimized by the median, so the point model uses the L1 objective."""

import lightgbm as lgb

VERY_SMALL = dict(
    objective="l1", n_estimators=100, learning_rate=0.05, num_leaves=4,
    min_child_samples=60, random_state=42, verbose=-1,
)


def fit_point(train, features, params=VERY_SMALL):
    """Fit the median (L1) model on a training table."""
    return lgb.LGBMRegressor(**params).fit(train[features], train["units"])


def fit_quantile(train, features, alpha, params=VERY_SMALL):
    """Fit a quantile model at level `alpha`."""
    return lgb.LGBMRegressor(**{**params, "objective": "quantile", "alpha": alpha}).fit(
        train[features], train["units"]
    )
