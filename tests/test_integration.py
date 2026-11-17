"""Checks against the real Olist data. They are skipped when the data or the database is missing.

They reproduce what was verified by hand against notebooks 03 and 04, so a change in the code
that moves the published results makes them fail.
"""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from olist_forecast import build_demand_series, build_features, decision_costs, load_series

SERIES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "demand_daily.csv"
DB_URL = "postgresql+psycopg2://olist:olist@localhost:5432/olist"

needs_series = pytest.mark.skipif(
    not SERIES_PATH.exists(), reason="data/processed/demand_daily.csv not found"
)


@needs_series
def test_decision_costs_of_the_latest_period_match_the_notebook():
    long = build_features(load_series(SERIES_PATH))
    out = decision_costs(long, 0, cu=4, co=1)
    assert out["test_start"] == pd.Timestamp("2018-06-27").date()
    assert out["baseline"] == pytest.approx(9105.25, rel=1e-4)
    assert out["baseline_buffer"] == pytest.approx(5977.051, rel=1e-4)
    assert out["model"] == pytest.approx(5764.138, rel=1e-3)


@needs_series
def test_series_has_the_documented_shape():
    series = load_series(SERIES_PATH)
    assert series.shape == (594, 10)
    assert series.index.min() == pd.Timestamp("2017-01-05")
    assert series.index.max() == pd.Timestamp("2018-08-21")


@needs_series
def test_series_built_from_the_database_matches_the_saved_one():
    engine = create_engine(DB_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL database not reachable")
    pd.testing.assert_frame_equal(
        build_demand_series(engine), load_series(SERIES_PATH), check_freq=False, check_names=False
    )
