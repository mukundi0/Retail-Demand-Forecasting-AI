"""Train and save the selected M5 LightGBM model for local/API inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from .forecast_experiments import CODES, HORIZON, SEED, TRAIN_DAYS, fit, load_data, make_features
except ImportError:  # Supports direct script execution from the repository root.
    from forecast_experiments import CODES, HORIZON, SEED, TRAIN_DAYS, fit, load_data, make_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_model"))
    parser.add_argument("--max-series", type=int, default=0, help="Use 0 for the full M5 population.")
    parser.add_argument("--prices-path", type=Path, default=None)
    args = parser.parse_args()
    if args.max_series < 0:
        parser.error("--max-series must be zero or positive")

    meta, y, codes, days, cal, sales_file, prices, price_file = load_data(
        args.data_dir, args.max_series, args.prices_path
    )
    if prices is None:
        raise ValueError("The selected production model uses scheduled prices; provide sell_prices.csv.")

    cutoff = y.shape[1] - 1
    history = y[:, : cutoff + 1]
    # Select the tree count from the final chronological early-stopping window.
    selector = fit(history, cutoff, cal, codes, prices)
    rounds = max(1, int(selector.best_iteration or selector.current_iteration()))

    # Refit on all labeled targets from the most recent training window, using
    # the selected round count. This makes the artifact use all available history.
    first = max(28, cutoff - TRAIN_DAYS + 1)
    target_days = np.arange(first, cutoff + 1, dtype=np.int32)
    features = make_features(history, target_days, cal, codes, prices)
    labels = history[:, target_days].T.reshape(-1)
    train_data = lgb.Dataset(features, label=labels, categorical_feature=CODES, free_raw_data=True)
    params = {
        "objective": "regression_l1", "metric": "mae", "learning_rate": .05,
        "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": .9,
        "verbosity": -1, "seed": SEED, "feature_fraction_seed": SEED,
        "bagging_seed": SEED, "data_random_seed": SEED, "num_threads": -1,
    }
    model = lgb.train(params, train_data, num_boost_round=rounds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "lightgbm_price.txt"
    model.save_model(str(model_path))
    category_maps = {
        column: [str(value) for value in pd.Categorical(meta[source]).categories]
        for column, source in zip(CODES, ["store_id", "cat_id", "dept_id", "state_id"])
    }
    metadata = {
        "model": "lightgbm_recursive_price",
        "model_file": model_path.name,
        "feature_names": model.feature_name(),
        "categorical_features": CODES,
        "category_code_maps": category_maps,
        "horizon_days": HORIZON,
        "history_days_required": 28,
        "selected_boosting_rounds": rounds,
        "selection_validation_days": 28,
        "refit_training_window_days": len(target_days),
        "series_evaluated": len(meta),
        "seed": SEED,
        "sales_file_sha256": hashlib.sha256(sales_file.read_bytes()).hexdigest(),
        "sell_prices_file_sha256": hashlib.sha256(price_file.read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "lightgbm": lgb.__version__,
    }
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {model_path} using {rounds} boosting rounds; refit on {len(target_days)} days across {len(meta)} series.")


if __name__ == "__main__":
    main()

