# Handoff: Retail Demand Forecasting AI Python Setup

## Current status

The project environment setup is blocked by the Python installation currently being used. The latest PowerShell output shows Python 3.15.0. The requirements pin NumPy 2.1.3 and pandas 2.2.3; pip is downloading source archives for these packages and attempts to build pandas, then fails because Visual Studio C++ build tools (`vswhere.exe`) are unavailable.

The same output reports that `venv` could not copy its launcher into `.venv\Scripts\python.exe`, so the existing `.venv` may be incomplete. The subsequent pip command still ran, but it does not establish that the environment is healthy.

Earlier shell diagnostics in this assistant session showed a different interpreter, MSYS2 Python 3.12.11 at `C:\msys64\ucrt64\bin\python.exe`, with no pip. The user's newest output shows Python 3.15.0, so interpreter selection is inconsistent across shells or has since changed. The `py` launcher was previously not found in one shell.

## Project facts

- `README.md` says Python 3.10 or newer.
- `requirements.txt` pins:
  - `numpy==2.1.3`
  - `pandas==2.2.3`
  - `lightgbm==4.5.0`
  - `fastapi==0.146.0`
  - `uvicorn==0.53.0`
- Training fails at `import pandas` because dependencies have not installed successfully.
- `lightgbm` is consequently also missing.

## Recommended next steps

1. Install standard 64-bit CPython 3.12 for Windows from python.org, including the Python launcher. Python 3.12 is preferable here to the currently selected 3.15 because the pinned packages are being fetched as source archives under 3.15.
2. Close and reopen PowerShell after installation. From the repository root, verify the launcher and interpreter:

   ```powershell
   py -0p
   py -3.12 --version
   ```

3. Rebuild the incomplete environment using the explicit interpreter. Only remove `.venv` after confirming the current directory is the repository root:

   ```powershell
   Remove-Item -Recurse -Force .venv
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -c "import sys; print(sys.executable); print(sys.version)"
   ```

4. Install wheels and verify imports:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install --only-binary=:all: --timeout 120 -r requirements.txt
   python -c "import pandas, numpy, lightgbm; print(pandas.__version__, numpy.__version__, lightgbm.__version__)"
   ```

5. With dependencies installed, run final model training from the repository root:

   ```powershell
   python src/train_final_model.py --data-dir data/raw --output-dir artifacts/final_model --max-series 0
   ```

## Setup command corrections

A malformed command appeared in the user's console history (`Set-Exepython --versionope ...`). Run each command separately and confirm its output before continuing. Do not append project paths to `-m venv .venv`; the target should be exactly `.venv` from the repository root.

## Scope and data

The README expects M5 source files under `data/raw/`: `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`. Confirm those files are present before training. Raw data and generated artifacts are locally ignored by Git.
