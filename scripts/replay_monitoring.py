"""Replay three ways of keeping the model up to date over the last three backtest periods.

It is a simulation on past data. The model is never retrained, retrained every four weeks or
retrained every week, and for each week the error, the coverage of the band and the cost of the
decision are measured against what really happened.

Example:
    python scripts/replay_monitoring.py
    python scripts/replay_monitoring.py --cu 9 --co 1 --relative
"""

import argparse
from pathlib import Path

from olist_forecast import build_features, load_series
from olist_forecast.monitoring import replay, summarize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--series", default="data/processed/demand_daily.csv")
    parser.add_argument("--output", default="data/processed/monitoring_replay.csv", help="Weekly results")
    parser.add_argument("--cu", type=float, default=4.0, help="Cost per unit of shortage")
    parser.add_argument("--co", type=float, default=1.0, help="Cost per unit of idle capacity")
    parser.add_argument("--relative", action="store_true", help="Widening proportional to the 28-day mean")
    parser.add_argument(
        "--wape-ratio", type=float, default=1.25, help="Error alert, times the calibration error"
    )
    parser.add_argument("--min-coverage", type=float, default=0.70, help="Coverage alert threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    long = build_features(load_series(args.series))
    results = replay(
        long,
        args.cu,
        args.co,
        relative=args.relative,
        wape_ratio=args.wape_ratio,
        min_coverage=args.min_coverage,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(summarize(results).round(3).T.to_string())
    print(f"\nWeekly results saved to {args.output}")


if __name__ == "__main__":
    main()
