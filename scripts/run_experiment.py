"""Run one backtest configuration and log it to MLflow.

Example:
    python scripts/run_experiment.py --name very_small
    python scripts/run_experiment.py --name very_small_scaled --relative
    python scripts/run_experiment.py --name default --n-estimators 300 --num-leaves 31 --min-child-samples 20
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse

import mlflow
import mlflow.lightgbm

from olist_forecast import FEATURES, VERY_SMALL, build_features, evaluate_config, load_series
from olist_forecast.backtest import split_period
from olist_forecast.models import fit_point


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--series", default="data/processed/demand_daily.csv")
    parser.add_argument("--experiment", default="olist-demand-forecast")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--name", required=True, help="Run name shown in MLflow")
    parser.add_argument("--cu", type=float, default=4.0, help="Cost per unit of shortage")
    parser.add_argument("--co", type=float, default=1.0, help="Cost per unit of idle capacity")
    parser.add_argument("--relative", action="store_true", help="Widening proportional to the 28-day mean")
    parser.add_argument("--cal-days", type=int, default=56)
    parser.add_argument("--test-days", type=int, default=56)
    parser.add_argument("--n-estimators", type=int, default=VERY_SMALL["n_estimators"])
    parser.add_argument("--learning-rate", type=float, default=VERY_SMALL["learning_rate"])
    parser.add_argument("--num-leaves", type=int, default=VERY_SMALL["num_leaves"])
    parser.add_argument("--min-child-samples", type=int, default=VERY_SMALL["min_child_samples"])
    return parser.parse_args()


def main():
    args = parse_args()
    params = {
        **VERY_SMALL,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
    }
    long = build_features(load_series(args.series))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.name):
        mlflow.log_params({
            **{f"model_{key}": value for key, value in params.items() if key != "verbose"},
            "cu": args.cu, "co": args.co, "cu_over_co": args.cu / args.co,
            "relative_widening": args.relative, "cal_days": args.cal_days,
            "test_days": args.test_days, "n_features": len(FEATURES),
        })
        mlflow.log_metrics(evaluate_config(
            long, params, args.cu, args.co, args.relative, args.cal_days, args.test_days
        ))
        # Model of the most recent backtest period, trained only on data before its test window
        train, _, _ = split_period(long, 0, test_days=args.test_days)
        mlflow.lightgbm.log_model(fit_point(train, FEATURES, params), name="model")


if __name__ == "__main__":
    main()
