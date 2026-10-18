# Retail Demand Forecasting AI

Research implementation for **“Evaluating the Effectiveness of AI-Based Demand Forecasting for Retail Businesses Under Volatile Demand Conditions.”**

The original exploration notebook is unchanged. The new Phase 2 notebook, runner, and report snapshot make the chronological experiment and its current 5,000-series findings easy to inspect.

## Repository map

- `01_Data_Exploration.ipynb`: original exploratory work, retained unchanged.
- `notebooks/02_Chronological_Experiments.ipynb`: experiment design, findings, and summary tables.
- `src/forecast_experiments.py`: data preparation, rolling validation, forecasting, pooled scoring, and series-cluster bootstrap.
- `reports/m5_5000_pilot/`: aggregate results and run manifest for the corrected 5,000-series pilot.
- `requirements.txt`: pinned Python dependencies.
- `data/raw/`: local M5 input files; not committed.
- `artifacts/`: newly generated outputs, including per-series errors used by bootstrap; not committed.

## Experiment design

The runner uses seed 42 and samples 1,000 item-store series by default. Set `--max-series 5000` for the larger pilot or `--max-series 0` for every series. It evaluates three non-overlapping 28-day chronological folds at the end of observed history. Each fold trains on earlier observations using a rolling 365-day window, with the final 28 days of that training window reserved for chronological early stopping.

The models are:

- **7-day seasonal naive:** repeat the last observed week.
- **Four-week weekday mean:** average the previous four observations for each weekday and repeat that pattern.
- **LightGBM recursive:** one-step forecasts using lags, rolling demand summaries, calendar events, SNAP indicators, and item/store metadata. Each recursive step uses the correct forecast-day index, so lag 1 is the most recent observed or predicted day.
- **Price-aware LightGBM:** the same model plus scheduled sell price, 7-day price change, and price relative to the prior 28-day mean.

M5 price features assume scheduled prices are known for the forecast horizon, as supplied in the M5 price table. If `sell_prices.csv` is absent, the runner still produces the no-price LightGBM comparison. There is no separate promotion field; price changes are used as a proxy.

Volatility groups are recalculated at each fold cutoff as relative terciles of the coefficient of variation across positive-demand weekly totals in the preceding 52 complete weeks. Zero-demand-week share is reported separately.

## Scores and uncertainty

`metrics.csv` contains fold-level WAPE and MAE by model and volatility group. `pooled_metrics.csv` combines absolute errors and actual demand across folds before calculating pooled WAPE.

`bootstrap_comparison.csv` compares each LightGBM model with both traditional baselines, and compares price-aware LightGBM with the no-price model. Each replicate samples series IDs with replacement and keeps all of a selected series’ fold observations together. Positive WAPE improvement means the candidate model has lower error. Intervals are the 2.5th and 97.5th percentiles of 1,000 bootstrap replicates. The positive-improvement share is descriptive, not a p-value. Volatility-group intervals are conditional on each series’ fold-specific group membership and on this sample/model setup.

The runner writes `series_metrics.csv` locally for the cluster bootstrap; generated artifacts are excluded from version control. `run_manifest.json` records sample size, fixed seed, data checksums, dependency versions, and bootstrap count.

## Data and run instructions

Place `sales_train_validation.csv`, `calendar.csv`, and optionally `sell_prices.csv` under `data/raw/`. The script uses unit sales, calendar covariates, and (when available) known scheduled prices. Raw files are not committed.

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/forecast_experiments.py --max-series 5000 --bootstrap-replicates 1000
```

To use a price table at a separate path, pass `--prices-path PATH`. For a full-population run, use `--max-series 0`; this can require substantial memory because features expand across series and days. Increase sample size in stages.

## Corrected 5,000-series pilot

Using the supplied M5 files, seed 42, three folds, and 1,000 bootstrap replicates:

| Model | Pooled WAPE | Pooled MAE |
|---|---:|---:|
| 7-day seasonal naive | 0.8753 | 1.2085 |
| Four-week weekday mean | 0.7764 | 1.0721 |
| LightGBM, no price | 0.6937 | 0.9578 |
| LightGBM, scheduled price features | 0.6845 | 0.9451 |

Against the stronger traditional baseline (four-week weekday mean), price-aware LightGBM improved WAPE by 0.0919 points overall (95% series-cluster bootstrap interval 0.0843–0.0998). Adding price features to LightGBM improved WAPE by 0.0092 points (interval 0.0080–0.0103). In low, moderate, and high volatility groups, price-aware LightGBM improvements over the weekday-mean baseline were 0.0441, 0.1345, and 0.1821 WAPE points, respectively; subgroup intervals are in the notebook and report snapshot.

These are pilot estimates for one seeded 5,000-series sample. They do not establish performance for the full M5 population or other retailers. The bootstrap does not include variation from drawing a new 5,000-series sample or model tuning. M5 future prices are treated as scheduled and known at forecast time.

## Limitations

The model is not tuned. The current phase does not estimate forecast uncertainty, include a standalone promotion variable, or specialize for intermittent demand. A full-population run is the next validation step before making population-level claims.
