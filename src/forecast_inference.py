"""Helpers for producing 28-day recursive forecasts from a saved M5 model."""
from __future__ import annotations
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast_experiments import CODES, DYNAMIC, PRICE_FEATURES, recursive

STATIC = ["store_id", "cat_id", "dept_id", "state_id"]
CALENDAR_COLUMNS = ["weekday", "month", "weekend", "event_day", "snap_CA", "snap_TX", "snap_WI"]


def load_model_bundle(model_dir: Path):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(model_dir / metadata["model_file"])), metadata


def forecast_series(model, metadata, sales_history, static_metadata, future_calendar,
                    recent_prices, future_prices):
    """Return 28 non-negative daily forecasts for one item-store series.

    `sales_history` and `recent_prices` are the latest 28 observed daily values.
    `future_calendar` and `future_prices` each contain the next 28 known days.
    Missing prices are allowed and represented with the model's missing-price feature.
    """
    history = np.asarray(sales_history, dtype=np.float32).reshape(-1)[-28:]
    past_price = np.asarray(recent_prices, dtype=np.float32).reshape(-1)[-28:]
    future_price = np.asarray(future_prices, dtype=np.float32).reshape(-1)
    if len(history) != 28 or len(past_price) != 28 or len(future_price) != 28:
        raise ValueError("Provide exactly 28 recent sales, 28 recent prices, and 28 future prices.")
    cal_future = pd.DataFrame(future_calendar).reset_index(drop=True)
    if len(cal_future) != 28 or any(column not in cal_future for column in CALENDAR_COLUMNS):
        raise ValueError(f"future_calendar must have 28 rows and columns: {CALENDAR_COLUMNS}")
    cal_future = cal_future[CALENDAR_COLUMNS]
    cal_prefix = pd.DataFrame({column: np.zeros(28, dtype=np.int8) for column in CALENDAR_COLUMNS})
    calendar = pd.concat([cal_prefix, cal_future], ignore_index=True)

    codes = {}
    maps = metadata["category_code_maps"]
    for code_column, source_column in zip(CODES, STATIC):
        value = str(static_metadata[source_column])
        try:
            codes[code_column] = np.int32(maps[code_column].index(value))
        except ValueError as exc:
            raise ValueError(f"Unknown {source_column}: {value}") from exc
    codes = pd.DataFrame([codes])

    price_path = np.concatenate([past_price, future_price]).astype(np.float32).reshape(1, -1)
    forecasts = recursive(
        model, history.reshape(1, -1), cutoff=27, cal=calendar,
        codes=codes, prices=price_path,
    )[0]
    return forecasts.astype(float).tolist()

