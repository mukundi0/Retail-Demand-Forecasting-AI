# Retail Demand Forecasting AI

Research implementation for **“Evaluating the Effectiveness of AI-Based Demand Forecasting for Retail Businesses Under Volatile Demand Conditions.”**

The original exploration notebook remains unchanged. The next notebook documents the chronological experiment, pooled scores, and series-level bootstrap intervals.

## Repository map

- `01_Data_Exploration.ipynb`: original data exploration; retained as the initial research record.
- `src/forecast_experiments.py`: data preparation, three-fold forecasting, pooled scoring, and bootstrap comparison.
- `notebooks/02_Chronological_Experiments.ipynb`: experiment protocol, current 5,000-series results, and an interactive results summary.
- `requirements.txt`: pinned Python dependencies.
- `data/raw/`: local M5 input files; not committed.
- `artifacts/`: generated metrics and run manifest; not committed.

## Experiment design

The runner uses seed 42 and samples 1,000 item-store series by default. Set `--max-series 5000` for the larger pilot or `--max-series 0` to use every series. It evaluates three non-overlapping 28-day chronological folds at the end of observed history. Each fold trains from earlier data using a rolling 365-day window, with the final 28 training days used for chronological early stopping.

The models are:

- **7-day seasonal naive:** repeat the last observed week.
- **Four-week weekday mean:** average the previous four observations for each weekday and repeat that pattern.
- **LightGBM recursive:** one-step model with lag and rolling-demand features, weekday/month, event-day, and SNAP indicators. Predictions feed into later forecast days.

Volatility groups are recalculated at each fold cutoff as relative terciles of coefficient of variation across positive-demand weekly totals during the preceding 52 complete weeks. The zero-demand-week share is reported separately.

## Scores and uncertainty

`metrics.csv` contains WAPE and MAE by fold, model, and volatility group. `pooled_metrics.csv` combines absolute error and actual demand across folds before calculating pooled WAPE, and combines absolute error and observation counts for pooled MAE.

`bootstrap_comparison.csv` reports a paired, series-cluster bootstrap comparison of LightGBM against **both** traditional baselines. Each replicate samples series IDs with replacement and keeps all of a sampled series’ fold observations together. Positive WAPE improvement favors LightGBM. The 95% interval uses the 2.5th and 97.5th percentiles; the bootstrap probability is the share of replicates with positive improvement, not a p-value. Volatility-group results use the series’ fold-specific labels. Intervals are conditional on this sample and model setup; they do not measure variation from drawing a new 5,000-series sample or tuning the models.

`series_metrics.csv` contains the compact per-series fold errors used to reproduce those intervals. `run_manifest.json` records seed, sample size, input checksum, dependency versions, and bootstrap count.

## Data and run instructions

Place the public M5 files `sales_train_validation.csv` and `calendar.csv` under `data/raw/`. The script uses unit sales and calendar features; it does not use prices or test labels.

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/forecast_experiments.py --max-series 5000 --bootstrap-replicates 1000
```

The notebook reads the generated files under `artifacts/`. Full-population runs can use substantial memory because training features expand across series and dates. Increase the sample size in stages before using `--max-series 0`.

## Current 5,000-series pilot

With seed 42 and the supplied sales file (SHA-256 recorded in the run manifest), pooled WAPE was 0.6928 for LightGBM, 0.7764 for the weekday-mean baseline, and 0.8753 for seasonal naive. LightGBM’s improvement over the weekday-mean baseline was 0.0836 WAPE points overall (series-cluster bootstrap 95% interval 0.0758–0.0919). The new notebook reports the subgroup comparisons.

Treat these as pilot estimates. This sample and fixed model configuration do not establish performance for every M5 series or other retailers. The model does not yet use price/promotion features, produce uncertainty forecasts, or specialize for intermittent demand.
