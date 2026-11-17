"""Evaluate one configuration over the rolling backtest and return flat metrics."""

import numpy as np

from .backtest import split_period, wape
from .calibration import calibrated_bounds
from .decision import decision_costs
from .features import FEATURES
from .models import VERY_SMALL, fit_point


def evaluate_config(
    long,
    params=VERY_SMALL,
    cu=4.0,
    co=1.0,
    relative=False,
    cal_days=56,
    test_days=56,
    n_periods=3,
    features=FEATURES,
):
    """Metrics of each backtest period (period_1 is the most recent) and their means.

    Per period: WAPE of the baseline and of the model, coverage of the 10-90 band before and
    after calibration, and the cost of the three reservation policies.
    """
    metrics = {}
    for k in range(n_periods):
        name = f"period_{k + 1}"
        tr, _, te = split_period(long, k, test_days=test_days)
        model = fit_point(tr, features, params)
        metrics[f"{name}_wape_baseline"] = float(wape(te["units"], te["baseline"]))
        metrics[f"{name}_wape_model"] = float(wape(te["units"], model.predict(te[features])))

        te_band, lo, hi, adjustment = calibrated_bounds(
            long,
            k,
            params,
            cal_days=cal_days,
            relative=relative,
            features=features,
            test_days=test_days,
        )
        y = te_band["units"].to_numpy()
        metrics[f"{name}_coverage_raw"] = float(((y >= lo) & (y <= hi)).mean())
        metrics[f"{name}_coverage_calibrated"] = float(
            ((y >= lo - adjustment) & (y <= hi + adjustment)).mean()
        )

        costs = decision_costs(long, k, cu, co, params, cal_days, relative, features, test_days)
        for policy in ("baseline", "baseline_buffer", "model"):
            metrics[f"{name}_cost_{policy}"] = float(costs[policy])
        metrics[f"{name}_saving_vs_buffer"] = 1 - costs["model"] / costs["baseline_buffer"]

    for suffix in sorted({key.split("_", 2)[2] for key in list(metrics)}):
        values = [metrics[f"period_{k + 1}_{suffix}"] for k in range(n_periods)]
        metrics[f"mean_{suffix}"] = float(np.mean(values))
    return metrics
