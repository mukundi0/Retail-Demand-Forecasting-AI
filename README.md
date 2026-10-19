# Retail Demand Forecasting AI

A beginner-friendly research project that predicts daily product demand for retail stores. It uses the public M5 forecasting data and LightGBM, a tree-based machine-learning library.

**Recommended starting point:** read [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md), then open [`notebooks/03_Dataset_to_Model_70_30.ipynb`](notebooks/03_Dataset_to_Model_70_30.ipynb) and run the cells in order. That notebook is the complete chronological 70% train / 30% test workflow.

## What this project does

For each item sold in each store, it uses past sales, calendar information, item/store identifiers, and scheduled prices to forecast future unit sales. The evaluation compares the machine-learning model against simple forecasts based on recent sales. The holdout split is by date: all training dates come before all test dates.

## Project structure

```text
Retail-Demand-Forecasting-AI/
├── README.md                         # Project overview and quick links
├── requirements.txt                  # Python packages used by this project
├── docs/
│   └── PROJECT_GUIDE.md              # Beginner guide, setup, concepts, troubleshooting
├── notebooks/
│   ├── 01_Data_Exploration.ipynb     # Initial dataset exploration
│   ├── 02_Chronological_Experiments.ipynb # Earlier multi-fold research results
│   └── 03_Dataset_to_Model_70_30.ipynb   # Main end-to-end training/evaluation
├── src/
│   ├── forecast_experiments.py        # Data loading, features, shared model helpers
│   ├── forecast_holdout.py            # 70/30 training and rolling evaluation runner
│   ├── train_final_model.py           # Optional model fit on all available dates
│   ├── forecast_inference.py          # Load a model and forecast one item-store series
│   └── api.py                         # Optional local FastAPI service
├── reports/                           # Small, committed summaries of prior experiments
├── data/raw/                          # Downloaded source CSVs; ignored by Git
└── artifacts/                         # Locally generated models/reports; ignored by Git
```

Virtual environments such as `.venv_cp312/`, caches, raw datasets, and generated model artifacts are local working files; they are intentionally excluded from Git.

## Quick start

1. Use 64-bit CPython 3.12. Install it from python.org if it is not already available.
2. Create an environment and install packages:

   ```powershell
   py -3.12 -m venv .venv_cp312
   .\.venv_cp312\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m pip install jupyterlab ipykernel
   python -m ipykernel install --user --name retail-demand --display-name "Python (retail demand)"
   ```

   On macOS/Linux, use `python3 -m venv .venv_cp312` and `source .venv_cp312/bin/activate` instead.

3. Put the M5 files listed below in `data/raw/`.
4. Start Jupyter with `jupyter lab` from the project root, open `notebooks/03_Dataset_to_Model_70_30.ipynb`, select the `Python (retail demand)` kernel, and run all cells from top to bottom. Set `MAX_SERIES = 1000` for a faster pipeline check; leave it at `0` for the full dataset.

The notebook saves the trained 70%-only model and its evaluation to `artifacts/holdout_70_30/`. It is CPU based and the full dataset can require substantial memory and time.

## Required data

Place these CSV files in `data/raw/`:

- `sales_train_validation.csv` — daily unit sales per product-store series.
- `calendar.csv` — dates, events, weekdays, and SNAP program flags.
- `sell_prices.csv` — weekly scheduled prices for each product and store.

These M5 files are not committed because they are large. The data guide explains where to get them, what the columns mean, and how to check the setup.

## Command-line alternative

From the repository root, run a 1,000-series smoke run:

```powershell
python src/forecast_holdout.py --data-dir data/raw --output-dir artifacts/holdout_70_30_smoke --max-series 1000 --bootstrap-replicates 50
```

Then run the full dataset:

```powershell
python src/forecast_holdout.py --data-dir data/raw --output-dir artifacts/holdout_70_30 --max-series 0 --bootstrap-replicates 1000
```

The notebook presents the same stages interactively. Both use the feature and evaluation helpers in `src/forecast_experiments.py` and `src/forecast_holdout.py`.

## Other notebooks and scripts

- `notebooks/01_Data_Exploration.ipynb` introduces the dataset through exploratory charts and tables.
- `notebooks/02_Chronological_Experiments.ipynb` records an earlier experiment with three rolling validation folds. Its figures are research results, not the 70/30 holdout result.
- `src/train_final_model.py` is an optional post-evaluation workflow that fits a model using available data for future inference. Because it uses all available dates, it is not the model to use when reporting a 70/30 test score.
- `src/api.py` serves forecasts locally after a model bundle has been created. See the guide for the API commands and request format.

## Previously recorded multi-fold results

The committed `reports/m5_full_population/` files summarize a separate three-fold experiment over 30,490 series. They compare naive, weekday-average, and LightGBM models; those values should not be mixed with the newer 70/30 holdout metrics. The exact split and data checksums for a holdout run are stored in its local `run_manifest.json`.

## Limitations

These results describe the supplied M5 data and this specific split and setup. They do not prove performance for another retailer. This is a research prototype; it does not tune every hyperparameter or provide prediction intervals for individual forecasts.
