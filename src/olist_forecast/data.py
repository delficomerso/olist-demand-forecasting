"""Build the daily demand series from the Olist database."""

import pandas as pd
from sqlalchemy import bindparam, text

DEMAND_SQL = """
SELECT
    o.order_purchase_timestamp::date AS day,
    t.product_category_name_english AS category,
    COUNT(*) AS units
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id
JOIN product_category_name_translation t
    ON t.product_category_name = p.product_category_name
WHERE o.order_status NOT IN :excluded
  AND o.order_purchase_timestamp >= :start
  AND o.order_purchase_timestamp < :end
GROUP BY day, category
ORDER BY day, category
"""


def build_demand_series(
    engine,
    excluded_statuses=("canceled", "unavailable"),
    start="2017-01-01",
    end="2018-09-01",
    top_n=10,
    keep_from="2017-01-05",
    keep_to="2018-08-21",
):
    """Daily units per category by purchase date, one column per category.

    Orders with an excluded status are dropped, the `top_n` categories by total units are kept,
    missing days count as zero, and the series is trimmed to [keep_from, keep_to], where the
    marketplace has stable activity. `end` is exclusive.
    """
    query = text(DEMAND_SQL).bindparams(
        bindparam("excluded", expanding=True)
    )
    daily = pd.read_sql(
        query, engine, params={"excluded": list(excluded_statuses), "start": start, "end": end}
    )
    return _pivot_top_categories(daily, start, end, top_n, keep_from, keep_to)


def _pivot_top_categories(daily, start, end, top_n, keep_from, keep_to):
    """Pivot long daily counts into a complete day x category table."""
    top = daily.groupby("category")["units"].sum().nlargest(top_n).index
    series = daily[daily["category"].isin(top)].pivot(index="day", columns="category", values="units")
    full_range = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    series.index = pd.to_datetime(series.index)
    series = series.reindex(full_range).fillna(0).astype(int)
    series.index.name = "day"
    return series.loc[keep_from:keep_to]


def load_series(path):
    """Read a saved demand series (day index, one column per category)."""
    return pd.read_csv(path, index_col="day", parse_dates=True)
