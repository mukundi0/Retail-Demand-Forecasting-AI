# Retail Demand Forecasting AI

Research project: **Evaluating the Effectiveness of AI-Based Demand Forecasting for Retail Businesses Under Volatile Demand Conditions.** The original exploration notebook, `01_Data_Exploration.ipynb`, is preserved unchanged. This phase adds reproducible chronological experiments and initial LightGBM comparisons.

## Repository map

- `01_Data_Exploration.ipynb` — original data exploration (unchanged).
- `notebooks/02_Chronological_Experiments.ipynb` — experiment design, findings, and result tables.
- `src/forecast_experiments.py` — data preparation, validation, baselines, LightGBM, scoring, and bootstrap.
- `reports/m5_5000_pilot/` — committed aggregate report snapshot and run manifest.
- `data/raw/` — local M5 input files; ignored by Git.
- `artifacts/` — generated fold and per-series outputs; ignored by Git.
- `requirements.txt` — pinned Python dependencies.

## Data placement

Put these files in `data/raw/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv` (optional; enables scheduled price features)

The runner also accepts a separate price path with `--prices-path`. Raw data and per-series results are not committed. SHA-256 checksums for the data used in the pilot are recorded in the manifest.

## Reproduce the experiment

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/forecast_experiments.py --max-series 5000 --bootstrap-replicates 1000
```

Use `--max-series 0` to select all available item-store series. The full M5 run is compute-intensive; start at 1,000, then 5,000, before attempting the full population. Outputs are written under `artifacts/`.

## Protocol

The experiment predicts 28 days in each of three rolling chronological folds. A 365-day training window precedes each validation horizon; a chronological 28-day segment of that window is used for early stopping. Series selection is seeded (`42`). Volatility groups are fold-specific terciles of the coefficient of variation of positive-demand weekly totals over the previous 52 complete weeks. The share of zero-demand weeks is reported separately.

Models:

1. 7-day seasonal naive.
2. Four-week weekday mean.
3. Recursive LightGBM using lagged demand, rolling demand summaries, calendar/events, SNAP, and series metadata.
4. Price-aware LightGBM, adding scheduled sell price, 7-day relative price change, and price relative to the preceding 28-day mean. Scheduled prices are assumed known over the forecast horizon; price changes are a proxy, not an explicit promotion flag.

WAPE is pooled absolute error divided by pooled actual demand. MAE is pooled absolute error per forecast observation. Bootstrap resampling samples series IDs with replacement and retains each selected series' fold observations together (1,000 replicates). Intervals describe series-sampling uncertainty conditional on this 5,000-series sample and model setup.

## Corrected 5,000-series pilot

The sample was evaluated on all three folds (420,000 forecast observations per model). Pooled results:

| Model | Pooled WAPE | Pooled MAE |
|---|---:|---:|
| 7-day seasonal naive | 0.8753 | 1.2085 |
| Four-week weekday mean | 0.7765 | 1.0721 |
| LightGBM, no price | 0.6916 | 0.9549 |
| LightGBM, scheduled price features | 0.6842 | 0.9447 |

Compared with the four-week weekday mean, price-aware LightGBM reduced WAPE by 0.0923 points (95% series-cluster bootstrap interval 0.0843–0.1002). The no-price LightGBM improvement was 0.0849 (0.0776–0.0922). Adding scheduled price features reduced WAPE by 0.0074 (0.0062–0.0086). Price-aware WAPE improvements over the weekday mean were 0.0443 (0.0373–0.0521) in low volatility, 0.1352 (0.1234–0.1469) in moderate volatility, and 0.1823 (0.1608–0.2038) in high volatility.

These pilot results support further investigation, not a population-wide conclusion. The full-population run was attempted but did not complete a verified fold within the available compute session. Run it in a provisioned environment before making full-M5 claims. This phase does not tune models, estimate forecast uncertainty, or include an independent promotion feature.
