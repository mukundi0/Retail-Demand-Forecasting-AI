"""Local FastAPI service for 28-day M5 forecasts."""
from __future__ import annotations
from contextlib import asynccontextmanager
from datetime import date as Date, timedelta
import math
import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from .forecast_inference import forecast_series, load_model_bundle
except ImportError:  # Supports running with `src` as the working directory too.
    from forecast_inference import forecast_series, load_model_bundle


class StaticMetadata(BaseModel):
    store_id: str = Field(min_length=1)
    cat_id: str = Field(min_length=1)
    dept_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)


class CalendarDay(BaseModel):
    date: Date
    event_day: bool = False
    snap_CA: bool = False
    snap_TX: bool = False
    snap_WI: bool = False


class ForecastRequest(BaseModel):
    item_id: str = Field(min_length=1, description="Item identifier for the response; static metadata is sent separately.")
    series_id: str | None = Field(default=None, description="Optional item-store series identifier for tracking.")
    static_metadata: StaticMetadata
    last_observed_date: Date
    sales_history: Annotated[list[float], Field(min_length=28, max_length=28)]
    recent_prices: Annotated[list[float | None], Field(min_length=28, max_length=28)]
    future_calendar: Annotated[list[CalendarDay], Field(min_length=28, max_length=28)]
    future_prices: Annotated[list[float | None], Field(min_length=28, max_length=28)]

    @field_validator("sales_history")
    @classmethod
    def sales_are_valid(cls, values):
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("sales_history values must be finite and non-negative")
        return values

    @field_validator("recent_prices", "future_prices")
    @classmethod
    def prices_are_valid(cls, values):
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise ValueError("Prices must be finite and non-negative, or null when unavailable")
        return values

    @model_validator(mode="after")
    def calendar_is_next_four_weeks(self):
        expected = [self.last_observed_date + timedelta(days=i) for i in range(1, 29)]
        supplied = [day.date for day in self.future_calendar]
        if supplied != expected:
            raise ValueError("future_calendar must contain the 28 consecutive dates after last_observed_date")
        return self


class ForecastPoint(BaseModel):
    date: Date
    forecast_units: float


class ForecastResponse(BaseModel):
    item_id: str
    series_id: str | None
    model: str
    horizon_days: int
    forecasts: list[ForecastPoint]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("FORECAST_MODEL_DIR", "artifacts/final_model"))
    try:
        app.state.model, app.state.model_metadata = load_model_bundle(model_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Forecast model not found in {model_dir}. Run src/train_final_model.py first "
            "or set FORECAST_MODEL_DIR to the saved model folder."
        ) from exc
    yield


app = FastAPI(
    title="Retail Demand Forecast API",
    description="Returns a recursive 28-day unit-sales forecast for one item-store series.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.model_metadata["model"]}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest, request: Request):
    calendar_rows = []
    for day in payload.future_calendar:
        calendar_rows.append({
            "weekday": day.date.weekday(),
            "month": day.date.month,
            "weekend": day.date.weekday() >= 5,
            "event_day": day.event_day,
            "snap_CA": day.snap_CA,
            "snap_TX": day.snap_TX,
            "snap_WI": day.snap_WI,
        })
    try:
        predictions = forecast_series(
            request.app.state.model,
            request.app.state.model_metadata,
            payload.sales_history,
            payload.static_metadata.model_dump(),
            calendar_rows,
            [np.nan if value is None else value for value in payload.recent_prices],
            [np.nan if value is None else value for value in payload.future_prices],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    points = [
        ForecastPoint(date=day.date, forecast_units=max(0.0, float(units)))
        for day, units in zip(payload.future_calendar, predictions, strict=True)
    ]
    return ForecastResponse(
        item_id=payload.item_id,
        series_id=payload.series_id,
        model=request.app.state.model_metadata["model"],
        horizon_days=len(points),
        forecasts=points,
    )

