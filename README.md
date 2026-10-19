# Retail Demand Forecasting AI

Research topic: **Evaluating the Effectiveness of AI-Based Demand Forecasting for Retail Businesses Under Volatile Demand Conditions.** The original exploration notebook, `01_Data_Exploration.ipynb`, is preserved unchanged.

## Repository map

- `01_Data_Exploration.ipynb` — original exploratory notebook.
- `notebooks/02_Chronological_Experiments.ipynb` — chronological research protocol and full M5 findings.
- `src/forecast_experiments.py` — three-fold evaluation of baselines and LightGBM models.
- `src/train_final_model.py` — final full-data fit and saved model bundle.
- `src/forecast_inference.py` — reusable 28-day inference helper for a future API.
- `reports/m5_5000_pilot/` — earlier 5,000-series pilot snapshot.
- `reports/m5_full_population/` — complete 30,490-series evaluation snapshot.
- `data/raw/` — local M5 files; ignored by Git.
- `artifacts/` — generated metrics and model bundle; ignored by Git.
- `requirements.txt` — pinned Python dependencies.

## Data files

Place these files under `data/raw/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

The price table supplies scheduled prices for price-aware training and forecasts. Raw data and trained model files are deliberately kept local. The run manifests include the SHA-256 checksums of the sales and price files.

## Run training in Google Colab

Colab is a practical option if Windows selects MSYS2 Python or cannot install the pinned wheels. The current project data files total about 309 MiB. ZIP the repository folder itself so the archive contains a top-level `Retail-Demand-Forecasting-AI/` folder with `src/`, `requirements.txt`, and `data/raw/` inside it. Upload that ZIP to Google Drive so it can be copied into a fresh Colab runtime. Do not include `.venv*`, `.git`, or generated artifacts.

In a new Colab notebook, select a CPU runtime, then run these cells in order. LightGBM in this project is configured for CPU training; a GPU runtime does not accelerate this code.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set `ZIP_PATH` to the uploaded archive's path in Drive, then stage it on the runtime's local disk:

```python
from pathlib import Path
import shutil, zipfile

ZIP_PATH = Path('/content/drive/MyDrive/Retail-Demand-Forecasting-AI.zip')
PROJECT = Path('/content/Retail-Demand-Forecasting-AI')
ARCHIVE = Path('/content/project.zip')
shutil.copy2(ZIP_PATH, ARCHIVE)
with zipfile.ZipFile(ARCHIVE) as bundle:
    bundle.extractall('/content')
print('Project:', PROJECT)
print('Sales data:', (PROJECT / 'data/raw/sales_train_validation.csv').exists())
print('Price data:', (PROJECT / 'data/raw/sell_prices.csv').exists())
```

Install the pinned packages and verify imports:

```python
%cd /content/Retail-Demand-Forecasting-AI
%pip install -r requirements.txt
import numpy, pandas, lightgbm
print(numpy.__version__, pandas.__version__, lightgbm.__version__)
```

First train a 1,000-series sample to confirm the pipeline works, then train all series. The two runs save separate bundles:

```python
!python src/train_final_model.py --data-dir data/raw --output-dir /content/drive/MyDrive/retail-smoke-model --max-series 1000
!python src/train_final_model.py --data-dir data/raw --output-dir /content/drive/MyDrive/retail-final-model --max-series 0
```

The output should include `lightgbm_price.txt` and `model_metadata.json`. To run the API within the Colab runtime, start Uvicorn bound to localhost and check `/health`; the runtime stops when Colab disconnects, so copy the model bundle from Drive before ending the session if you need it locally. Colab resource availability and session limits vary, so use a high-RAM runtime if the full-data fit runs out of memory. See [Colab's resource FAQ](https://research.google.com/colaboratory/faq.html#resource-limits) for current runtime limits.

## Run the chronological evaluation

Use standard 64-bit CPython 3.10 or newer. On Windows, use the Python Launcher (`py`) or a CPython installation from python.org; MSYS2/UCRT Python is not a supported interpreter for these pinned Windows wheels. Python 3.12 is the recommended Windows version.

On Windows, create and activate the environment with:

```powershell
py -3.12 -m venv .venv_cp312
.\.venv_cp312\Scripts\Activate.ps1
```

On macOS/Linux, use:

```bash
python3 -m venv .venv_cp312
source .venv_cp312/bin/activate
```

Then install the dependencies and start the evaluation:

```bash
python -m pip install -r requirements.txt
python src/forecast_experiments.py --data-dir data/raw --output-dir artifacts/full_m5 --max-series 0 --bootstrap-replicates 1000
```

The evaluation uses three chronological 28-day folds, a rolling 365-day training window, and seed 42. It compares 7-day seasonal naive, four-week weekday mean, recursive LightGBM without prices, and price-aware LightGBM. Volatility is grouped at each fold cutoff using terciles of the coefficient of variation of positive-demand weekly totals over the preceding 52 complete weeks. Zero-demand-week share is reported separately.

## Full-population result

The supplied M5 data contains 30,490 item-store series. The completed evaluation covers three folds and 2,561,160 forecast observations per model. Pooled scores:

| Model | Pooled WAPE | Pooled MAE |
|---|---:|---:|
| 7-day seasonal naive | 0.8796 | 1.2141 |
| Four-week weekday mean | 0.7788 | 1.0750 |
| LightGBM, no price features | 0.6972 | 0.9625 |
| LightGBM, scheduled price features | 0.6939 | 0.9579 |

Compared with the weekday-mean baseline, price-aware LightGBM reduced WAPE by 0.0848 points overall (95% series-cluster bootstrap interval 0.0808–0.0888). The improvements were 0.0399 (0.0359–0.0436) in low volatility, 0.1196 (0.1105–0.1271) in moderate volatility, and 0.1734 (0.1652–0.1815) in high volatility. Price features gave a small overall improvement over the no-price LightGBM; in the low-volatility subgroup their WAPE was effectively tied. These are M5 results and should not be generalized automatically to other retailers.

Fold-level, pooled, bootstrap, and run-manifest files are in `reports/m5_full_population/`. Per-series bootstrap inputs are generated locally and are not committed.

## Train the final model bundle

After evaluation, create the deployment-ready price-aware LightGBM model:

```bash
python src/train_final_model.py --data-dir data/raw --output-dir artifacts/final_model --max-series 0
```

This selects the boosting-round count using the latest chronological 28-day early-stopping window, then refits on the most recent 365 days of labeled observations across the selected population. For the supplied full M5 data, the selected count was 300 rounds across all 30,490 series. The command writes `lightgbm_price.txt` and `model_metadata.json` locally; Git ignores these generated model artifacts.

## Forecast interface and next step

`src/forecast_inference.py` provides `load_model_bundle()` and `forecast_series()`. A forecast request needs one series' static item/store metadata, the latest 28 daily sales, the latest 28 prices, and the next 28 calendar rows plus scheduled prices. Calendar rows use `weekday`, `month`, `weekend`, `event_day`, `snap_CA`, `snap_TX`, and `snap_WI`. The helper returns 28 non-negative predictions. These functions are the inference layer for the next step: adding and running a FastAPI service.

## Limits

This phase evaluates one retail dataset and one seeded research setup. It does not tune hyperparameters, estimate forecast uncertainty, or use a standalone promotion field. The model and results are an initial research prototype, not a production service.


## Run the local API

Install the pinned packages, create the full-data model bundle, and start the service from the repository root:

```bash
python -m pip install -r requirements.txt
python src/train_final_model.py --data-dir data/raw --output-dir artifacts/final_model --max-series 0
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The API loads `artifacts/final_model` on startup. Set `FORECAST_MODEL_DIR` if the bundle is stored elsewhere. Check `GET /health`, open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), then use `POST /forecast` to request a prediction. The docs page lets you submit a request interactively.

A forecast request is for one product-store series and includes its item ID, store/category/department/state, the latest 28 daily sales, the latest 28 prices, and the next 28 dates with event/SNAP flags and scheduled prices. The response contains one non-negative unit forecast per day. Missing prices can be sent as JSON `null`. Dates must be the 28 consecutive days immediately after `last_observed_date`.

This is a local research prototype. It has no authentication and should stay bound to `127.0.0.1`; do not expose it to the public internet as-is.
