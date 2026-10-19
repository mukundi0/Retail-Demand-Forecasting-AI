"""Train on the first 70% of M5 dates and evaluate recursively on the final 30%."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from .forecast_experiments import (
        CODES,
        HORIZON,
        SEED,
        TRAIN_DAYS,
        fit,
        load_data,
        make_features,
        volatility,
    )
except ImportError:  # Supports direct script execution from the repository root.
    from forecast_experiments import (
        CODES,
        HORIZON,
        SEED,
        TRAIN_DAYS,
        fit,
        load_data,
        make_features,
        volatility,
    )

TRAIN_FRACTION = 0.70
STATIC = ["store_id", "cat_id", "dept_id", "state_id"]
MODELS = ["seasonal_naive_7d", "four_week_weekday_mean", "lightgbm_recursive_price"]


def model_parameters():
    return {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "num_threads": -1,
    }


def compact_feature_dtypes(features):
    """Keep the large 70%-training feature matrix compact in memory."""
    features["zero_share_28"] = features["zero_share_28"].astype(np.float32, copy=False)
    for column in CODES:
        if features[column].min() >= 0 and features[column].max() <= np.iinfo(np.int8).max:
            features[column] = features[column].astype(np.int8, copy=False)
    return features


def recursive_forecast(model, history, start_day, horizon, calendar, codes, prices):
    """Forecast a long holdout without repeatedly copying the growing history."""
    n_series = history.shape[0]
    stop_day = start_day + horizon
    work = np.empty((n_series, stop_day), dtype=np.float32)
    work[:, :start_day] = history[:, :start_day]
    predictions = np.empty((n_series, horizon), dtype=np.float32)
    for offset, day in enumerate(range(start_day, stop_day)):
        features = compact_feature_dtypes(
            make_features(work, np.asarray([day], dtype=np.int32), calendar, codes, prices)
        )
        values = np.maximum(model.predict(features, num_threads=-1), 0).astype(np.float32)
        work[:, day] = values
        predictions[:, offset] = values
    return predictions


def recursive_baselines(history, start_day, horizon):
    """Generate seasonal-naive and rolling four-week weekday forecasts."""
    n_series = history.shape[0]
    stop_day = start_day + horizon
    naive_work = np.empty((n_series, stop_day), dtype=np.float32)
    weekday_work = np.empty((n_series, stop_day), dtype=np.float32)
    naive_work[:, :start_day] = history[:, :start_day]
    weekday_work[:, :start_day] = history[:, :start_day]
    naive = np.empty((n_series, horizon), dtype=np.float32)
    weekday = np.empty((n_series, horizon), dtype=np.float32)
    for offset, day in enumerate(range(start_day, stop_day)):
        naive[:, offset] = naive_work[:, day - 7]
        naive_work[:, day] = naive[:, offset]
        weeks = weekday_work[:, day - 28 : day].reshape(n_series, 4, 7)
        weekday[:, offset] = weeks.mean(axis=1)[:, day % 7]
        weekday_work[:, day] = weekday[:, offset]
    return naive, weekday


def rolling_holdout_forecasts(model, history, start_day, horizon, calendar, codes, prices):
    """Cover the test period with rolling, non-overlapping 28-day forecast origins."""
    n_series = history.shape[0]
    forecasts = {name: np.empty((n_series, horizon), dtype=np.float32) for name in MODELS}
    for offset in range(0, horizon, HORIZON):
        origin = start_day + offset
        window = min(HORIZON, horizon - offset)
        naive, weekday = recursive_baselines(history, origin, window)
        forecasts["seasonal_naive_7d"][:, offset : offset + window] = naive
        forecasts["four_week_weekday_mean"][:, offset : offset + window] = weekday
        forecasts["lightgbm_recursive_price"][:, offset : offset + window] = recursive_forecast(
            model, history, origin, window, calendar, codes, prices
        )
        print(
            f"Completed holdout forecast {offset // HORIZON + 1}/"
            f"{int(np.ceil(horizon / HORIZON))}: {window} days from day index {origin}",
            flush=True,
        )
    return forecasts


def summarize_metrics(actual, predictions, groups, test_days):
    rows = []
    group_names = [("all", np.ones(len(groups), dtype=bool))]
    group_names.extend((name, groups == name) for name in ("low", "moderate", "high"))
    for model_name, forecast in predictions.items():
        errors = np.abs(actual - forecast)
        for group_name, mask in group_names:
            if not mask.any():
                continue
            group_errors = errors[mask]
            demand = float(actual[mask].sum())
            error_sum = float(group_errors.sum())
            rows.append(
                {
                    "model": model_name,
                    "volatility_group": group_name,
                    "series": int(mask.sum()),
                    "test_days": int(actual.shape[1]),
                    "test_start": test_days[0],
                    "test_end": test_days[-1],
                    "forecast_observations": int(group_errors.size),
                    "actual_demand_sum": demand,
                    "absolute_error_sum": error_sum,
                    "WAPE": error_sum / demand if demand else float("nan"),
                    "MAE": float(group_errors.mean()),
                    "RMSE": float(np.sqrt(np.mean(np.square(actual[mask] - forecast[mask])))),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_comparisons(series_metrics, replicates):
    rng = np.random.default_rng(SEED)
    comparisons = [
        ("seasonal_naive_7d", "lightgbm_recursive_price"),
        ("four_week_weekday_mean", "lightgbm_recursive_price"),
    ]
    rows = []
    for group_name in ("all", "low", "moderate", "high"):
        data = series_metrics if group_name == "all" else series_metrics[series_metrics.volatility_group == group_name]
        if data.empty:
            continue
        denominator = data.actual_demand_sum.to_numpy(dtype=np.float64)
        errors = {
            model: data[f"absolute_error_{model}"].to_numpy(dtype=np.float64)
            for model in MODELS
        }
        for reference, candidate in comparisons:
            bootstrap_values = []
            for offset in range(0, replicates, 100):
                count = min(100, replicates - offset)
                indexes = rng.integers(0, len(data), size=(count, len(data)))
                sampled_demand = denominator[indexes].sum(axis=1)
                valid = sampled_demand > 0
                improvements = (errors[reference][indexes] - errors[candidate][indexes]).sum(axis=1)
                bootstrap_values.append(improvements[valid] / sampled_demand[valid])
            values = np.concatenate(bootstrap_values) if bootstrap_values else np.asarray([])
            total_demand = denominator.sum()
            ref_wape = errors[reference].sum() / total_demand if total_demand else float("nan")
            candidate_wape = errors[candidate].sum() / total_demand if total_demand else float("nan")
            rows.append(
                {
                    "volatility_group": group_name,
                    "reference_model": reference,
                    "candidate_model": candidate,
                    "series": int(len(data)),
                    "bootstrap_replicates": int(len(values)),
                    "reference_WAPE": ref_wape,
                    "candidate_WAPE": candidate_wape,
                    "WAPE_improvement": ref_wape - candidate_wape,
                    "improvement_95CI_low": float(np.quantile(values, 0.025)) if len(values) else float("nan"),
                    "improvement_95CI_high": float(np.quantile(values, 0.975)) if len(values) else float("nan"),
                    "bootstrap_probability_improvement_positive": float((values > 0).mean()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/holdout_70_30"))
    parser.add_argument("--prices-path", type=Path, default=None)
    parser.add_argument("--max-series", type=int, default=0, help="Seeded series sample; use 0 for all series.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--reuse-model", type=Path, default=None, help="Evaluate an existing 70%-trained LightGBM model.")
    args = parser.parse_args()
    if args.max_series < 0 or args.bootstrap_replicates < 1:
        parser.error("--max-series must be zero or positive and --bootstrap-replicates must be positive")
    if args.reuse_model is not None and args.max_series != 0:
        parser.error("--reuse-model requires --max-series 0 to preserve the model's category-code maps")

    meta, y, codes, days, calendar, sales_file, prices, price_file = load_data(
        args.data_dir, args.max_series, args.prices_path
    )
    if prices is None or price_file is None:
        raise ValueError("The price-aware model requires sell_prices.csv.")

    split_day = int(np.floor(y.shape[1] * TRAIN_FRACTION))
    test_horizon = y.shape[1] - split_day
    if split_day < max(TRAIN_DAYS, 52 * 7) + HORIZON or test_horizon < HORIZON:
        raise ValueError("The data must provide at least one training window and one 28-day test window.")

    history = y[:, :split_day]
    cutoff = split_day - 1
    print(
        f"70/30 chronological split: training d_1 through {days[cutoff]} "
        f"({split_day} days); test {days[split_day]} through {days[-1]} "
        f"({test_horizon} days) across {len(meta)} series.",
        flush=True,
    )

    train_days = np.arange(28, split_day, dtype=np.int32)
    if args.reuse_model is not None:
        model = lgb.Booster(model_file=str(args.reuse_model))
        rounds = max(1, int(model.current_iteration()))
        print(f"Reusing existing model with {rounds} boosting rounds.", flush=True)
    else:
        selector = fit(history, cutoff, calendar, codes, prices)
        rounds = max(1, int(selector.best_iteration or selector.current_iteration()))
        train_features = compact_feature_dtypes(make_features(history, train_days, calendar, codes, prices))
        train_labels = history[:, train_days].T.reshape(-1)
        train_data = lgb.Dataset(train_features, label=train_labels, categorical_feature=CODES, free_raw_data=True)
        model = lgb.train(model_parameters(), train_data, num_boost_round=rounds)
        del train_features, train_labels, train_data, selector

    train_count = y.shape[0]
    volatility_groups, zero_week_share = volatility(history)
    rolling_origins = int(np.ceil(test_horizon / HORIZON))
    print(
        f"Using {rounds} boosting rounds; evaluating {test_horizon} future days "
        f"in {rolling_origins} rolling {HORIZON}-day windows.",
        flush=True,
    )
    predictions = rolling_holdout_forecasts(model, y, split_day, test_horizon, calendar, codes, prices)
    actual = y[:, split_day:]
    test_days = days[split_day:]

    metrics = summarize_metrics(actual, predictions, volatility_groups, test_days)
    series_metrics = pd.DataFrame(
        {
            "series_id": meta["id"].to_numpy(),
            "volatility_group": volatility_groups,
            "zero_week_share": zero_week_share,
            "actual_demand_sum": actual.sum(axis=1),
            **{
                f"absolute_error_{name}": np.abs(actual - forecast).sum(axis=1)
                for name, forecast in predictions.items()
            },
        }
    )
    bootstrap = bootstrap_comparisons(series_metrics, args.bootstrap_replicates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.csv").write_text(metrics.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "series_metrics.csv").write_text(series_metrics.to_csv(index=False), encoding="utf-8")
    (args.output_dir / "bootstrap_comparison.csv").write_text(bootstrap.to_csv(index=False), encoding="utf-8")
    model_path = args.output_dir / "lightgbm_price.txt"
    if args.reuse_model is None or args.reuse_model.resolve() != model_path.resolve():
        model.save_model(str(model_path))
    category_maps = {
        code: [str(value) for value in pd.Categorical(meta[source]).categories]
        for code, source in zip(CODES, STATIC)
    }
    model_metadata = {
        "model": "lightgbm_recursive_price",
        "model_file": model_path.name,
        "feature_names": model.feature_name(),
        "categorical_features": CODES,
        "category_code_maps": category_maps,
        "horizon_days": HORIZON,
        "history_days_required": 28,
        "selected_boosting_rounds": rounds,
        "training_fraction": TRAIN_FRACTION,
        "training_days": split_day,
        "test_days": test_horizon,
        "series_evaluated": len(meta),
        "seed": SEED,
        "sales_file_sha256": hashlib.sha256(sales_file.read_bytes()).hexdigest(),
        "sell_prices_file_sha256": hashlib.sha256(price_file.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
    }
    (args.output_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    manifest = {
        "evaluation": "chronological 70/30 holdout",
        "train_fraction": TRAIN_FRACTION,
        "test_fraction": round(1 - TRAIN_FRACTION, 2),
        "forecast_window_days": HORIZON,
        "rolling_forecast_origins": rolling_origins,
        "train_start_day": days[0],
        "train_end_day": days[cutoff],
        "test_start_day": days[split_day],
        "test_end_day": days[-1],
        "training_days": split_day,
        "test_days": test_horizon,
        "training_observations": int(train_count * len(train_days)),
        "series_evaluated": len(meta),
        "selected_boosting_rounds": rounds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": SEED,
        "sales_file_sha256": model_metadata["sales_file_sha256"],
        "sell_prices_file_sha256": model_metadata["sell_prices_file_sha256"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    overall = metrics[metrics.volatility_group == "all"].set_index("model")
    print("\nOverall holdout metrics:")
    print(overall[["WAPE", "MAE", "RMSE"]].to_string())
    print(f"\nSaved holdout results and the 70%-trained model to {args.output_dir}")


if __name__ == "__main__":
    main()
