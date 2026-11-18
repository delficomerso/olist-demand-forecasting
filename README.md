# Olist demand forecasting and capacity decision

![tests](https://github.com/delficomerso/olist-demand-forecasting/actions/workflows/tests.yml/badge.svg)

## What this project does

A company that sells products has to decide in advance how much capacity to reserve for next week, whether that is staff, trucks or warehouse space. Reserve too little and orders are late or lost. Reserve too much and the extra capacity is paid for and unused.

This project predicts how many items will be ordered each day in each product category over the next seven days, gives a range that the real number should fall in, and turns that into a recommended amount to reserve. It uses public data from Olist, a Brazilian online marketplace with about 100,000 orders, and it is written so that another company can use its own data and its own costs.

## Results

- **The forecast beats a simple rule.** The simple rule is to expect the average of the same weekday over the last four weeks. Its average error is 35% of the units sold. The model brings it to 32%, and it is better in each of the three test periods. A first model with default settings only tied the simple rule, and two changes made the difference. One was leaving out of training the five weeks starting at Black Friday 2017, because that single huge spike distorted the inputs of the following weeks. The other was using a small model, which generalized better than the default one with about 5,000 training rows. Classical statistical methods (Theta and ETS) reach about the same error, 0.32 to 0.33, so the model is not clearly better than them. It is kept because it also produces the range below and can take extra variables.
- **Each forecast comes with a range, and the range holds up.** A single number is never exactly right, so every day also gets a low and a high value, meant to contain the real demand on 8 of every 10 days. On average over the test periods it did on 79.5% of the days, between 70% and 86% depending on the period. This matters because the amount to reserve depends on how much demand can vary, not only on its average.
- **The decision costs less.** When a missing unit costs four times as much as an idle one, a calibrated safety margin on top of the simple rule already saves 24% of the cost. Reserving from the model, with the safety margin scaled to the size of each category, costs a further 5% to 8% less than that buffered rule, for cost ratios from 1 to 9.

The costs are **assumed** values, since Olist does not publish them, so these savings show how the method works and are not a measured result for a real company. The analysis behind each point, with its tables, is in [`notebooks/04_forecast_and_decision.ipynb`](notebooks/04_forecast_and_decision.ipynb).

## How it works

1. **Data.** The nine Olist CSV files are loaded into a PostgreSQL database. SQL scripts convert the text dates to timestamps and define primary and foreign keys.
2. **Demand series.** A SQL query counts the items ordered each day in each of the ten largest categories, leaving out orders that ended as canceled or unavailable.
3. **Forecast.** A LightGBM model predicts the next seven days from the recent past. It is compared with a simple baseline and with classical time series methods, and tested only on dates after the ones it was trained on, as it would be used in practice.
4. **Range.** The model also predicts a low and a high value for each day, and a correction measured on recent weeks makes the range more reliable.
5. **Decision.** If a missing unit costs `Cu` and an idle one costs `Co`, the cheapest amount to reserve is the one that demand stays below on a share `Cu / (Cu + Co)` of the days. With `Cu = 4` and `Co = 1` that share is 80%.
6. **Package and tests.** The code of the notebooks is organized as a Python package in `src/olist_forecast/`, with 98 automated tests written with pytest. Three of them need the real data and check that the package gives the same costs as the notebook, so they run on a machine that has the data and are skipped on GitHub. On every push, GitHub Actions checks the code style, runs the other 95 tests on Python 3.11, 3.12 and 3.14, and builds the Docker image.
7. **Experiment tracking.** `scripts/run_experiment.py` saves each configuration tried, with its parameters and metrics, in MLflow so that runs can be compared.
8. **Service.** A FastAPI service reads the series from the database, checks the inputs and returns the plan for the next week. A `Dockerfile` and a `docker-compose.yml` run the service and the database.

## Try it

You need Python 3.11 or later and Docker with Compose v2. Download the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the nine CSV files in `data/raw/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,tracking,notebooks]"

docker compose up -d --wait db      # PostgreSQL
python src/olist_forecast/load_data.py   # loads the CSV files
docker compose up --build api       # starts the service on port 8000
```

Then ask for a plan, with a missing unit costing 4 and an idle unit costing 1.

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"cu": 4, "co": 1, "categories": ["auto"], "day": "2018-08-22"}'
```

```json
{
  "as_of": "2018-08-21",
  "critical_ratio": 0.8,
  "items": [
    {
      "category": "auto",
      "day": "2018-08-22",
      "forecast": 16.12,
      "band_low": 7.04,
      "band_high": 28.45,
      "capacity": 25.02,
      "capacity_units": 26,
      "actual": null
    }
  ]
}
```

For `auto` on August 22, 2018 the model expects about 16 items, the range goes from 7 to 28, and the recommended reservation is 26 items. The `actual` field is empty because that day is after the last day in the data. Sending `"as_of": "2018-06-26"` instead makes the forecast with only the data up to that date and fills `actual` with what really happened. It is an illustration and not a clean test, because the model settings were chosen looking at these same dates. Interactive documentation of every option is at `http://127.0.0.1:8000/docs`.

The notebooks in `notebooks/` build the demand series and walk through the analysis, and notebook 03 writes the file used by `scripts/run_experiment.py` to log runs in MLflow. The tests run with `pytest`.

## Limitations

- There are one and a half years of data and ten categories, with no prices or promotions.
- The model was evaluated on three test periods of 56 days. Small differences between methods cannot be told apart from chance.
- The costs and the resource being reserved are assumptions, because Olist does not hold stock or ship products, its sellers do.
- The model settings were chosen on the first test period, and the same three periods are used to report the results, so the results are optimistic and should be confirmed on new data.
- The largest category, health_beauty, is under-forecast by about 20% and its range covers only 65% to 69% of the days.
- Black Friday appears once in the data and cannot be forecast from a single example, so it is left out of training. With more years of data, recurring events like this one should be included in the model.

## Repository

```
data/raw/             Olist CSV files, not versioned
data/processed/       Daily demand series, not versioned
notebooks/            Exploration, cleaning, demand series, forecast and decision
sql/                  Queries for exploration, cleaning and keys
scripts/              Logs a backtest configuration to MLflow
src/olist_forecast/   Python package, with the service in api.py
tests/                Automated tests, run on every push by GitHub Actions
Dockerfile            Image of the service
docker-compose.yml    PostgreSQL and the service
```

## Next steps

- Replay the three test periods as a retraining schedule with monitoring of error and coverage. This would be a simulation on past data and not a deployed monitor.
- Lead time from purchase to dispatch per category, to convert arrivals into workload by stage.
- Notebooks that import the package instead of repeating its code.
