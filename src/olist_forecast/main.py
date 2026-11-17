"""Entry point of the service: `uvicorn olist_forecast.main:app`.

The database is read from the DATABASE_URL environment variable.
"""

import os

from sqlalchemy import create_engine

from .api import create_app
from .data import build_demand_series

DEFAULT_URL = "postgresql+psycopg2://olist:olist@localhost:5432/olist"


def load_from_database():
    """Build the daily demand series from the Olist tables in PostgreSQL."""
    engine = create_engine(os.environ.get("DATABASE_URL", DEFAULT_URL))
    try:
        return build_demand_series(engine)
    finally:
        engine.dispose()


app = create_app(load_from_database)
