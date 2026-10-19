# Beginner Guide to the Retail Demand Forecasting Project

This guide explains what the project does, where its files belong, how to run it, and how to understand the results. You do not need to know machine learning before starting, but basic Python familiarity will help.

If the technical details feel like too much, start with the shorter [plain-language beginner guide](BEGINNER_GUIDE.md). The end-to-end notebook is also written as a CRISP-DM walkthrough.

## 1. The project in one sentence

We train a program to estimate how many units of each product will sell in each store, using earlier sales, product and store identifiers, calendar details, and scheduled prices.

The project uses the M5 Forecasting competition data. Its source files describe Walmart item-store sales, the calendar, and prices. See the [M5 data download page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) for the original files.

## 2. Important words

| Word | Meaning in this project |
| --- | --- |
| Observation | One measured value, such as units sold for one product in one store on one date. |
| Time series | A sequence of observations over time. Here, each item-store pair is a series. |
| Feature | An input value the model can use, such as sales 7 days ago or the day of the week. |
| Target / label | The value we want the model to predict: daily units sold. |
| Training | Showing the model examples with known answers so it can learn patterns. |
| Validation | A slice of training data used to choose training settings, such as how many trees to build. |
| Test / holdout | Later data kept out of training and used to estimate performance on unseen dates. |
| Forecast horizon | How far ahead a prediction covers. The project forecasts in windows of up to 28 days. |
| Baseline | A simple reference forecast. A complex model should be compared with one. |
| WAPE | Total absolute error divided by total actual demand. Lower is better. |
| MAE | Average absolute error in units sold. Lower is better. |
| RMSE | Square root of average squared error; it penalizes large misses more heavily. |

## 3. Folder map

```text
project root
├── README.md                 Short overview and quick start
├── requirements.txt          Exact Python package versions
├── docs/                     Explanations and setup guidance
├── notebooks/                Interactive analyses, in learning order
├── src/                      Reusable Python code and command-line programs
├── reports/                  Small summaries of earlier completed experiments
├── data/raw/                 Downloaded input CSVs (local, not in Git)
└── artifacts/                Models and outputs made by runs (local, not in Git)
```

### What each source file does

- `src/forecast_experiments.py` loads the CSVs, builds model features, fits a model for a validation window, creates recursive predictions, and contains shared comparison utilities.
- `src/forecast_holdout.py` runs the chronological 70/30 experiment. It selects training rounds, refits the model on the training period, evaluates the test period, and writes results.
- `src/train_final_model.py` builds a separate model for future use from available sales history. It is not the script to use when measuring the 70/30 test score, because it fits on all available dates.
- `src/forecast_inference.py` loads a saved model bundle and forecasts one product-store series for the next 28 days.
- `src/api.py` wraps inference in a local web API. The API is optional; you do not need it to train or evaluate the model.

### Notebook order

1. `notebooks/01_Data_Exploration.ipynb` looks at the raw data and its shape.
2. `notebooks/02_Chronological_Experiments.ipynb` documents an earlier three-fold research experiment.
3. `notebooks/03_Dataset_to_Model_70_30.ipynb` is the main end-to-end notebook for the strict 70/30 holdout process.

## 4. Set up Python

Use standard 64-bit CPython 3.12. The project pins package versions in `requirements.txt`; other Python builds, including some MSYS2 builds and very new Python releases, may not have compatible prebuilt packages.

### Windows PowerShell

Open PowerShell in the project root (the folder containing `README.md`) and run:

```powershell
py -3.12 --version
py -3.12 -m venv .venv_cp312
.\.venv_cp312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name retail-demand --display-name "Python (retail demand)"
python -c "import numpy, pandas, lightgbm; print(numpy.__version__, pandas.__version__, lightgbm.__version__)"
```

If PowerShell says script execution is disabled when activating, you may skip activation and use the environment's Python directly:

```powershell
.\.venv_cp312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv_cp312\Scripts\python.exe -m jupyter lab
```

If `py -3.12` cannot find Python, install 64-bit Python 3.12 from python.org and reopen PowerShell. Avoid mixing interpreters: package installation and notebook execution must use the same environment.

### macOS or Linux

```bash
python3 -m venv .venv_cp312
source .venv_cp312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name retail-demand --display-name 'Python (retail demand)'
```

Launch Jupyter with `jupyter lab` from the project root and select the `Python (retail demand)` kernel. Jupyter itself is not needed for the command-line run.

### Google Colab

Upload or extract the project folder in the Colab session, including `src/`, `requirements.txt`, and `data/raw/`. Change directory into that folder before running the notebook. Colab runtimes are temporary; copy any model or metrics you want to keep to Google Drive. The model uses CPU LightGBM, so a GPU runtime does not speed up this training code.

## 5. Prepare the data

Create `data/raw/` in the project root and place these files inside:

```text
data/raw/
├── sales_train_validation.csv
├── calendar.csv
└── sell_prices.csv
```

Download them from the [M5 Forecasting Accuracy data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). Kaggle may require an account and acceptance of competition terms. The project needs the validation sales file, not `sales_train_evaluation.csv`, for the documented experiment. The latter has extra dates and is not used by this 70/30 pipeline.

What the files contain:

- `sales_train_validation.csv`: one row per product-store series, with descriptive ID columns and daily sales columns named `d_1`, `d_2`, and so on.
- `calendar.csv`: maps each day ID to a real date and includes weekday, month, event, and SNAP program information.
- `sell_prices.csv`: gives a product's scheduled price at a store in a retail week. The loader aligns this weekly information to daily sales dates.

The data files are deliberately ignored by Git because they are large. After downloading, check that all three paths exist before running training. The optional exploration notebook also reads `sales_train_evaluation.csv` and `sample_submission.csv`; those extra files are not needed by the 70/30 workflow.

## 6. Run the main 70/30 notebook

Open `notebooks/03_Dataset_to_Model_70_30.ipynb` in Jupyter, choose the Python kernel from `.venv_cp312`, and use **Run All**. Run it from the project root so relative file paths resolve.

At the top, the notebook sets:

```python
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / 'data' / 'raw'
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'holdout_70_30'
MAX_SERIES = 0
```

`MAX_SERIES = 0` means all product-store series. Set it to `1000` for a smaller, faster check that still exercises the full pipeline. Do not use the sample's score as the final full-population result.

### What the notebook does, cell by cell

1. **Configure paths and packages.** It finds the project root and imports dependencies and shared project functions.
2. **Load data.** It reads sales, aligns calendar rows to sales days, and creates a price matrix. It verifies required files exist.
3. **Split by time.** The earliest 70% of dates become the training partition; the latest 30% are held out. This protects the evaluation from future-data leakage.
4. **Choose the number of boosting rounds.** A validation window at the end of the training partition is used for early stopping. Those dates still belong to training, and no test dates are used for this choice.
5. **Refit the model.** A fresh model is fitted on eligible labeled days in the entire 70% training partition. The earliest 28 days are warm-up history for lag features, so they are not training labels.
6. **Forecast the holdout.** The test period is forecast as consecutive, non-overlapping windows of up to 28 days. Forecasts inside each window are recursive: after day one, the model uses its own predictions as recent sales history.
7. **Compare forecasts.** It measures the LightGBM model alongside two simple baselines: sales from the same weekday one week earlier, and the average for that weekday across the previous four weeks.
8. **Report results.** It calculates pooled metrics overall and by demand-volatility group. It uses a series-level bootstrap to estimate confidence intervals for the WAPE improvements over the baselines.
9. **Save outputs.** It writes the trained model, run details, and evaluation tables under `artifacts/holdout_70_30/`.

### Why use rolling windows inside the test period?

At the first test date, the model only has training-period history. It predicts the next 28 days without seeing their true sales. When the following window starts, those earlier dates are now treated as observed history. This represents a business that refreshes forecasts every four weeks. The within-window predictions never use actuals from the same window.

## 7. Run without Jupyter

From the project root, activate the environment and run the same holdout program:

```powershell
python src/forecast_holdout.py --data-dir data/raw --output-dir artifacts/holdout_70_30 --max-series 0 --bootstrap-replicates 1000
```

Try the smaller run first:

```powershell
python src/forecast_holdout.py --data-dir data/raw --output-dir artifacts/holdout_70_30_smoke --max-series 1000 --bootstrap-replicates 50
```

To recompute evaluation using a saved model, the script supports `--reuse-model`, but the existing model must have been trained for the same full set of category codes. Use the same data population and options that were used to make that artifact.

## 8. Understand the output files

After a successful full run, `artifacts/holdout_70_30/` contains:

| File | Contents |
| --- | --- |
| `lightgbm_price.txt` | Trained LightGBM trees used to make predictions. |
| `model_metadata.json` | Feature names, category mappings, training fraction, package versions, and model details. |
| `metrics.csv` | Pooled WAPE, MAE, and RMSE overall and by volatility group for each model. |
| `series_metrics.csv` | Actual demand and absolute error per product-store series; used for bootstrap estimates. |
| `bootstrap_comparison.csv` | WAPE improvement estimates and 95% confidence intervals versus baselines. |
| `run_manifest.json` | Exact train/test day boundaries, population, data checksums, versions, and run timestamp. |

Lower WAPE, MAE, and RMSE indicate smaller errors. WAPE is `sum(abs(actual - forecast)) / sum(actual)`. A WAPE of 0.72 means the sum of absolute errors is about 72% of total demand; it does not mean that 72% of individual forecasts are wrong. The bootstrap interval describes uncertainty in the aggregate improvement across sampled product-store series; it is not an interval for an individual forecast.

The volatility groups are determined from the training history only. They describe how much positive weekly demand varies for each series, divided into low, moderate, and high groups.

## 9. What the model learns

The model is LightGBM regression with an absolute-error objective. Each training row represents one series on one target date. The feature set includes:

- Lagged sales: units sold 1, 7, 14, and 28 days earlier.
- Recent sales summaries: means, variation, and zero-sales share over the previous 7 or 28 days.
- Calendar values: weekday, month, weekend, event, and SNAP flags.
- Product and store codes: categorical identifiers for store, category, department, and state.
- Price values: current price, recent price changes, price compared with its recent average, and a missing-price flag.

The model predicts one day at a time. For a 28-day forecast, a prediction is added to history and used to construct features for the next day. Predictions are clipped at zero because negative unit sales do not make sense.

## 10. Common errors and fixes

| Error | Likely fix |
| --- | --- |
| `FileNotFoundError` for a CSV | Confirm the file name and place it under `data/raw/`. Run notebook or script from the repository root. |
| `ModuleNotFoundError` for pandas, NumPy, or LightGBM | Activate `.venv_cp312` and install with `python -m pip install -r requirements.txt` using that same Python. |
| pandas or NumPy tries to compile from source on Windows | Check that standard 64-bit CPython 3.12 is selected; avoid MSYS2 Python and very new Python releases for this pinned setup. |
| Notebook import fails for `src` | Set `PROJECT_DIR` to the repository folder and add it to `sys.path` as the notebook setup cell does. |
| Kernel is out of memory | Start with `MAX_SERIES = 1000`, close other applications, or use a higher-memory runtime. The complete feature matrix is large. |
| Model appears to take a long time | Full-population training and forecasting is compute-heavy. The script prints progress at each held-out forecast window. |
| Activation is blocked by PowerShell policy | Use `.\.venv_cp312\Scripts\python.exe` directly instead of activating. |

If a notebook cell fails, fix the first failing cell, then choose **Kernel → Restart Kernel and Run All**. Later cells often depend on variables made by earlier cells.

## 11. Optional local forecast API

The API is an optional demo for serving one series' next 28-day forecast. First make a model bundle. For a holdout-trained model, it can be loaded by pointing the API to `artifacts/holdout_70_30`; for a model refit on all available history, use `artifacts/final_model` instead. Start the service from the repository root:

```powershell
$env:FORECAST_MODEL_DIR = 'artifacts/holdout_70_30'
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try a request or `http://127.0.0.1:8000/health` to check startup. A request needs the series identifiers, 28 recent sales values, 28 recent prices, and the next 28 calendar dates with scheduled prices. Keep the service on `127.0.0.1`; this prototype has no authentication.

## 12. Research boundaries

The 70/30 process is a chronological evaluation over the dates in `sales_train_validation.csv`. It evaluates the final 30% of that file, not dates from `sales_train_evaluation.csv`. Results from the earlier three-fold experiment in `reports/m5_full_population/` are a different experiment and should not be described as the 70/30 holdout results. Performance on M5 does not guarantee performance on a different retailer's data.
