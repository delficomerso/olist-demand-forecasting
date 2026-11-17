"""HTTP service that returns the 7-day forecast and the capacity to reserve."""

import logging
import math
import threading
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .forecast import recommend_capacity

MIN_LEVEL, MAX_LEVEL = 0.01, 0.99
HORIZON = 7
logger = logging.getLogger(__name__)


class RecommendRequest(BaseModel):
    cu: float = Field(gt=0, description="Cost per unit of demand above the reserved capacity")
    co: float = Field(gt=0, description="Cost per unit of reserved capacity that is not used")
    categories: list[str] | None = Field(
        default=None, description="Categories to return. All of them when omitted"
    )
    scaled_widening: bool = Field(
        default=False, description="Widen the band in proportion to the level of each category"
    )
    as_of: date | None = Field(
        default=None,
        description="Last day of data to use. The forecast covers the 7 days after it. "
        "Defaults to the last day in the database",
    )
    day: date | None = Field(
        default=None, description="Return only this day, which must be one of the 7 forecast days"
    )

    @model_validator(mode="after")
    def check_ratio(self):
        level = self.cu / (self.cu + self.co)
        if not MIN_LEVEL <= level <= MAX_LEVEL:
            raise ValueError("cu / (cu + co) must be between 0.01 and 0.99")
        return self


class Item(BaseModel):
    category: str
    day: str
    forecast: float
    band_low: float
    band_high: float
    capacity: float
    capacity_units: int = Field(
        description="Capacity rounded up to whole units, the amount to actually reserve"
    )
    actual: float | None = Field(
        default=None, description="Demand that really happened, when that day is already in the database"
    )


class RecommendResponse(BaseModel):
    as_of: str = Field(description="Last day of data used. The forecast covers the 7 days after it")
    critical_ratio: float = Field(description="Share of demand that the capacity covers, cu / (cu + co)")
    items: list[Item]


def create_app(load_series: Callable[[], pd.DataFrame]) -> FastAPI:
    """Build the app. `load_series` returns the daily series (day index, one column per category).

    The series is read on the first request that needs it, so the service starts even if the
    database has not been loaded yet, and it answers 503 with an explanation until it has.
    """
    app = FastAPI(
        title="Olist demand forecast and capacity",
        description="Seven-day demand forecast per category with a calibrated 10-90 band and the "
        "capacity to reserve for a given shortage and idle cost.",
    )
    state = {"series": None, "results": {}}
    lock = threading.Lock()

    def get_series():
        with lock:
            if state["series"] is None:
                try:
                    series = load_series()
                except Exception as error:
                    logger.exception("The demand series could not be loaded")
                    raise HTTPException(
                        503,
                        "The demand series could not be loaded. Check that the database is running "
                        "and that the Olist tables have been loaded.",
                    ) from error
                if series.empty:
                    raise HTTPException(503, "The demand series is empty. Has the database been loaded?")
                state["series"] = series
            return state["series"]

    @app.get("/health")
    def health():
        return {"status": "ok", "data_loaded": state["series"] is not None}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(request: RecommendRequest):
        full = get_series()
        first_day, last_day = full.index.min().date(), full.index.max().date()
        as_of = request.as_of or last_day
        if not first_day <= as_of <= last_day:
            raise HTTPException(422, f"as_of must be between {first_day} and {last_day}")
        series = full.loc[: pd.Timestamp(as_of)]

        if request.categories:
            unknown = sorted(set(request.categories) - set(series.columns))
            if unknown:
                raise HTTPException(422, f"Unknown categories: {unknown}. Available: {list(series.columns)}")
        forecast_days = [as_of + timedelta(days=i) for i in range(1, HORIZON + 1)]
        if request.day is not None and request.day not in forecast_days:
            raise HTTPException(
                422,
                f"day must be one of the {HORIZON} forecast days: {forecast_days[0]} to {forecast_days[-1]}",
            )

        key = (round(request.cu / (request.cu + request.co), 6), request.scaled_widening, as_of)
        with lock:
            result = state["results"].get(key)
        if result is None:
            try:
                result = recommend_capacity(series, request.cu, request.co, relative=request.scaled_widening)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            with lock:
                state["results"][key] = result

        if request.categories:
            result = result[result["category"].isin(request.categories)]
        if request.day is not None:
            result = result[result["day"] == pd.Timestamp(request.day)]
        items = [
            Item(
                category=row.category,
                day=row.day.date().isoformat(),
                forecast=round(row.forecast, 2),
                band_low=round(row.band_low, 2),
                band_high=round(row.band_high, 2),
                capacity=round(row.capacity, 2),
                capacity_units=math.ceil(round(row.capacity, 6)),
                actual=float(full.at[row.day, row.category]) if row.day <= full.index.max() else None,
            )
            for row in result.itertuples()
        ]
        return RecommendResponse(as_of=as_of.isoformat(), critical_ratio=key[0], items=items)

    return app
