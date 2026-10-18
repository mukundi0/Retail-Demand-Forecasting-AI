# Retail Demand Forecasting AI

Research phase for **“Evaluating the Effectiveness of AI-Based Demand Forecasting for Retail Businesses Under Volatile Demand Conditions.”**

The original `01_Data_Exploration.ipynb` is retained unchanged. This phase adds reproducible 28-day experiments with chronological validation, demand-volatility groups, traditional baselines, and initial LightGBM training.

## Experiment design

The runner `src/forecast_experiments.py` uses seed 42 and evaluates a deterministic sample of 1,000 item-store series by default. Set `--max-series 0` to include all series. It creates three non-overlapping 28-day validation folds at the end of the observed sales history. Each fold trains only on earlier observations, using a rolling 365-day window of daily examples; the final 28 training days form a chronological early-stopping window.

Models compared:

- **7-day seasonal naive:** repeat the last observed week.
- **Four-week weekday mean:** average the last four observations for each weekday and repeat the resulting pattern.
- **LightGBM recursive:** one-step model using demand lags, rolling summaries, weekday/month, calendar event and SNAP indicators; forecast values are fed back through the 28-day horizon.

Metrics are WAPE and MAE, reported overall and by volatility group. At each fold cutoff, series are assigned to relative terciles of coefficient of variation across positive-demand weekly totals in the preceding 52 complete weeks. The zero-demand-week share is reported separately. Validation-period demand is not used to calculate labels.

## Data setup

Download the public M5 Forecasting data and put these files in `data/raw/`:

- `sales_train_validation.csv`
- `calendar.csv`

The experiment uses daily unit sales and calendar date, weekday, event-day, and SNAP indicators. Prices and evaluation/test labels are not used. Raw data is excluded from version control.

```text
data/
  raw/
    calendar.csv
    sales_train_validation.csv
```

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/forecast_experiments.py
```

For the complete item-store population:

```bash
python src/forecast_experiments.py --max-series 0
```

The 1,000-series default is an initial pipeline comparison, not a final population-level result. Record the sample size and data checksum when comparing experiments.

## Outputs

Runs write local outputs under `artifacts/`:

- `metrics.csv`: metrics for each fold, model, and volatility group, including series counts.
- `run_manifest.json`: fixed seed, sample size, sales-file SHA-256, library versions, and label definition.

WAPE is NaN if a group has no positive actual demand; MAE remains defined.

## Limitations

This is an initial fixed-parameter model, not a tuned result. The recursive model may accumulate error across the 28-day horizon. The current phase uses unit sales and calendar covariates; it does not incorporate prices, promotions beyond calendar indicators, uncertainty intervals, or specialized intermittent-demand methods.

## Project files

- `01_Data_Exploration.ipynb`: original exploration, preserved.
- `src/forecast_experiments.py`: reproducible experiment runner.
- `requirements.txt`: pinned runtime dependencies.
- `.gitignore`: excludes local datasets, environments, caches, and generated artifacts.
