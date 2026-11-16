# Olist demand forecasting and capacity decision

> Draft. Sections marked *pending* will be completed as the project advances.

## Objective

Forecast the daily demand of each product category seven days ahead, with its uncertainty, and use the forecast to decide how much capacity to reserve at minimum cost. The data and the cost parameters can be replaced with those of another company.

## Data

The [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) has about 100,000 orders placed on the Olist marketplace between September 2016 and October 2018, with order status, timestamps of each stage, items, products, categories, customers, sellers, payments and reviews.

Olist is a marketplace, it does not hold stock or ship products, the sellers do. The capacity decision is therefore a stylized exercise. The resource being reserved (people, transport, storage, stock) is a parameter of the cost function and not a fixed assumption.

## What is being forecast

The number of items ordered each day in each of the ten largest categories, counted on the purchase date and excluding orders that ended as canceled or unavailable.

- **Items, not orders or revenue.** Each item is a unit of work to fulfill.
- **Purchase date.** It is when the demand becomes known. A business that plans deliveries would count on the delivery date, a one-column change in the query.
- **All statuses except canceled and unavailable.** Orders still in transit at the end of the data are real demand. Using only delivered orders would make the last months look artificially low.
- **Ten categories**, chosen by units sold over the whole window. Smaller categories have many zero days and are an intermittent demand problem, which needs different methods.
- **Daily frequency.** Categories sell between 7 and 19 units per day on average and have no sales on at most 6% of days.
- **Items without a category** (1.4% of items) are dropped. The 1,603 with no category could belong to any category, so the demand of the ten categories may be understated by up to about 2%.

## Cancellations

Canceled and unavailable orders are 1.2% of all orders and are removed before building the series.

- Their monthly share ranges from 0.3% to 3.5%, with the highest value in February 2017. This is small next to a forecast error around 32%, so no separate cancellation model is used.
- Only 75 canceled orders (0.08% of all orders) had already shipped, so almost all cancellations happen before dispatch and orders already in the system can be treated as firm.

A business with a higher or more variable cancellation rate would forecast gross arrivals and apply a cancellation rate. In the package the excluded statuses are a parameter, so the gross series is one argument away.

## Known versus uncertain work

The model forecasts new arrivals only. Orders already in the system are known and can be converted into workload by stage using the observed lead times between purchase, approval, dispatch and delivery.

*Pending.* Measure the lead time from purchase to dispatch per category.

## Pipeline

1. **Load.** The nine CSV files go into PostgreSQL, one table per file (`src/olist_forecast/load_data.py`).
2. **Explore** (`notebooks/01_exploration.ipynb`). Order statuses, date range, orders per month, cancellations, top categories. Sets the modeling window, January 2017 to August 2018.
3. **Clean** (`sql/02_cleaning.sql`, `sql/03_keys.sql`, `notebooks/02_cleaning.ipynb`). Text dates become timestamps. Primary keys on `orders`, `products`, `order_items` and the category translation table, and foreign keys from `order_items` to `orders` and `products`, created without errors so there are no orphan items.
4. **Build the series** (`notebooks/03_demand_series.ipynb`). Items per day and category in SQL, ten largest categories pivoted to one row per day, days without sales filled with zero, trimmed to 2017-01-05 through 2018-08-21 because the first four days of January are empty and the data is incomplete after August 21. Saved to `data/processed/demand_daily.csv`, 594 days by 10 categories.
5. **Forecast and decide** (`notebooks/04_forecast_and_decision.ipynb`). Described below.

## Forecasting

**Horizon.** Seven days is an assumption about how far ahead capacity must be reserved. Every feature uses data from at least seven days earlier, so another horizon changes the lags.

**Evaluation.** Splits are by date, never random. The last 56 days are the test period, and model choices are checked on three consecutive 56-day backtest periods, each predicted with a model trained only on earlier data.

| Period | Training | Test |
|---|---|---|
| 1 | 2017-01-05 to 2018-06-26 | 2018-06-27 to 2018-08-21 |
| 2 | 2017-01-05 to 2018-05-01 | 2018-05-02 to 2018-06-26 |
| 3 | 2017-01-05 to 2018-03-06 | 2018-03-07 to 2018-05-01 |

The configuration was chosen on period 1, so that period is optimistic. Periods 2 and 3 are out-of-sample for that choice. The metric is WAPE, total absolute error divided by total units.

**Baseline.** The average of the same weekday over the last four weeks, with a WAPE of 0.352 on the last 56 days.

**Model.** One LightGBM model for all categories, with the category, the 7-day lag, the 7-day and 28-day rolling means and the day of the week, plus the 14, 21 and 28-day lags. The loss is the absolute error, because WAPE is minimized by the median and not by the mean. A first model with default settings ties the baseline, 0.351 against 0.352. Two changes improve it.

- The weeks from 2017-11-24 to 2017-12-29 are removed from training, because the single 940-unit Black Friday spike enters the lag features of the following weeks.
- A small model (4 leaves, 100 trees) generalizes better than the default with about 5,000 training rows.

| Backtest | Period 1 | Period 2 | Period 3 | Mean |
|---|---|---|---|---|
| Baseline | 0.352 | 0.404 | 0.300 | 0.352 |
| LightGBM, default | 0.342 | 0.378 | 0.290 | 0.337 |
| LightGBM, small | 0.334 | 0.366 | 0.271 | 0.324 |

The small model is better than the baseline in each period. Removing the lags beyond 7 days leaves the error unchanged, 0.324.

**What did not help.** A target relative to the recent level (0.331), a Poisson loss (0.326), blending with the baseline (0.330 and 0.329) and calendar features (0.325).

**Classical benchmarks.** Seasonal naive reaches 0.389, ETS 0.332 and Theta 0.323, fitted per category and refit every week. They see data up to one day before each forecast week, so their average horizon is four days against seven for LightGBM, which favors them. Three methods reaching a similar error suggests the limit of what the history of each category explains. LightGBM is kept because it gives quantiles with the same pipeline and can take external variables when a business has them, which were not tested here.

**Errors.** In the last test period the median forecast is 10% below actual demand overall and 20% below in health_beauty, the largest category. Daily errors correlate with the error of the previous day, 0.2 to 0.6 by category and 0.7 for the daily total, and only weakly with the error of a week earlier. The misses come in runs, consistent with level shifts that the lag features see seven days late.

## Uncertainty

The same model is trained with a quantile loss for the 10th, 50th and 90th percentiles. The raw 10-90 band covers 74% of days on average, between 68% and 79% across periods, against a target of 80%.

The band is widened with a conformal adjustment. A 56-day calibration window placed before each test period measures how far the band fell short, and the band is widened by that amount. Coverage goes from 70% to 79.5% on average, with 82%, 70% and 86% in the three periods. In the second period the calibration window showed almost no shortfall and the coverage did not improve. Coverage is an empirical result and not a guarantee, since consecutive errors are correlated.

A single widening in units serves the categories unevenly, from 65% in health_beauty to 86% in telephony. A widening proportional to the 28-day mean of each category narrows that to between 76% and 84%, except for health_beauty at 69%.

On pinball loss the model beats the calibrated baseline at the 10th and 50th percentiles in all three periods, and on average at the 90th, where the baseline is better in one period.

## Decision

Capacity must be reserved before demand is known. Each unit of unmet demand costs `Cu` and each unit of idle capacity costs `Co`. The reservation that minimizes expected cost is the demand quantile at `Cu / (Cu + Co)`, the newsvendor rule. With `Cu = 4` and `Co = 1` that is the 80th percentile.

| Policy | What it reserves |
|---|---|
| Baseline | The four-week weekday average |
| Baseline with buffer | The same plus a buffer calibrated on the previous 56 days |
| Model | The calibrated model quantile at the critical ratio |

Total cost over the three backtest periods, as a saving against the previous policy.

| Cu / Co | Buffer vs baseline | Model vs buffer |
|---|---|---|
| 1 | -3.5% | 7.5% |
| 2 | 4.1% | 5.7% |
| 4 | 23.9% | 5.4% |
| 9 | 50.1% | 5.6% |

The model uses the widening proportional to the category level and the buffer uses the fixed one, which is the cheaper of the two buffers. With the fixed widening for both, the model saves 7.3%, 5.5%, 4.4% and 3.8%, and at a ratio of 4 the buffer was cheaper than the model in one of the three periods.

The costs are assumed values. They are the parameters a business replaces with its own.

## Main findings

- At a cost ratio of 4 or more, most of the saving comes from the decision rule, reserving the right quantile with a calibrated buffer. At ratios of 1 and 2 the model contributes more than the buffer.
- The model lowers WAPE from 0.352 to 0.324 in each of the three periods, and lowers cost by 5% to 8% against the best buffer. The cost gain is not consistent in every period.
- Three different methods reach a similar error, so the history of each category has a limit that no method tested gets below.
- A single outlier event hurts a lag-based model. Removing the Black Friday weeks from training improves WAPE from 0.351 to 0.342.

## Limitations

- One and a half years of data, ten series, no prices or promotions.
- Three backtest periods of 56 days, no confidence intervals. Differences of about 0.01 in WAPE are not distinguishable from noise.
- Model configuration was chosen on the first period, and the three periods were used both to choose and to report. The ten categories were chosen using the whole window.
- Costs are assumed and the capacity is stylized, so the savings show the method and are not a measured result.
- Each category reserves separately. With capacity shared between categories the total reserve would be lower.
- Health_beauty is under-forecast and under-covered even with the proportional widening.
- Black Friday cannot be forecast, there is a single event in the data.
- Orders in the last months may not have reached their final status, so a small share counted as demand may have been canceled after the data cut.

## Repository

```
data/raw/          Olist CSV files
data/processed/    demand_daily.csv
notebooks/         01 exploration, 02 cleaning, 03 demand series, 04 forecast and decision
sql/               01 exploration queries, 02 cleaning, 03 keys
src/olist_forecast/   load_data.py (package, pending)
api/               pending
tests/             pending
```

## Next steps

*Pending.*

- Package the series construction, feature table, backtest, calibration and decision into `src/olist_forecast/`, with the excluded statuses, horizon, test length and costs as parameters.
- Tests with pytest.
- Experiment tracking with MLflow.
- FastAPI service, Docker image, CI.
- Retraining schedule and monitoring of error and coverage.
- Lead time from purchase to dispatch per category, to convert arrivals into workload by stage.

## Setup

*Pending.* Docker command for PostgreSQL, Python environment, data download and load.
