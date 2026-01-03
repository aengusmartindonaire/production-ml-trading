# Production Machine Learning System for Systematic Trading

![Pipeline](https://img.shields.io/badge/Pipeline-end_to_end-green)
![Backtest](https://img.shields.io/badge/Backtest-walk_forward_decile-blue)
![Data](https://img.shields.io/badge/Data-1999_2025_BBg_factors-lightgrey)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Notebooks](https://img.shields.io/badge/Notebooks-8_notebooks-brightgreen)

A research-grade, config-driven machine learning pipeline for cross-sectional equity factor modeling and a simple long/short trading strategy, built from a single Bloomberg multi-factor panel.

**Author:** Aengus Martin Donaire  
**Contact:** aengusmartindonaire@gmail.com

---

## Project Overview

This project turns a raw multi-year factor panel into a complete ML trading prototype:

- Ingests a Bloomberg-style panel (risk factors, returns, classifications).
- Cleans and standardizes features (clipping, GICS handling, z-scoring).
- Compares linear and tree-based models for cross-sectional stock returns.
- Tunes a tree-based “champion” model (XGBoost) on the factor panel.
- Uses SHAP to explain what the model is actually doing.
- Runs a walk-forward long/short decile backtest and summarizes performance.

The core idea is to keep everything **reproducible and inspectable**:
raw data and assumptions live in `data/` and `configs/`, reusable code in
`src/lrcm/`, and experiments in `notebooks/` and `experiments/`.

### Architecture

**Config-driven research pipeline:**

- **Raw data**  
  - `data/raw/20251109_Blg_Rsk_Factors.par`  
  - Controlled by `configs/data.yml` (paths + core column names).

- **Reusable library (`src/lrcm`)**  
  - `data_loading.py`: load raw/processed panels with a single config.  
  - `preprocessing.py`: clipping, GICS category handling, cross-sectional z-scores.  
  - `features.py`: feature configuration, final numeric feature list.  
  - `models.py`: model constructors (BayesianRidge, RandomForest, XGBoost), CV helpers.  
  - `backtest.py`: walk-forward engine and portfolio construction.  
  - `evaluation.py`: IC, alpha, Sharpe, drawdown metrics + yearly summaries.  
  - `plots.py`: SHAP explainability and simple equity curves.

- **Experiments in notebooks**  
  - Each notebook corresponds to a stage of the pipeline: data inspection,
    feature selection, model bake-off, tuning, explainability, and backtest.

**Data flow:**

1. Raw Bloomberg factor panel is loaded via `configs/data.yml`.
2. Preprocessing and feature selection produce `data/processed/factor_panel.parquet`.
3. Static models (bake-off + tuning) are trained on this processed panel.
4. Tuned XGBoost hyperparameters are saved to `configs/models/xgb_champion.yml`.
5. SHAP analyses explain the champion model on the full panel.
6. Walk-forward backtest uses the same processed panel + champion model config
   and writes metrics/figures into `experiments/`.

---

## Current Status

**End-to-end pipeline implemented and tested on the factor panel ✅**

### Phase 1: Data & Features ✅

- **Repository layout established**  
  - `src/lrcm/` library, `configs/`, `notebooks/`, `data/`, `experiments/`, `paper/`.
- **Config-driven data access**  
  - `configs/data.yml` defines:
    - raw panel path,
    - processed panel path,
    - `Date`, `Ticker`, forward return columns.
- **Preprocessing logic** (`src/lrcm/preprocessing.py`)  
  - Forward return clipping by cross-section (removes extreme outliers).  
  - GICS category handling: rare sectors/industries collapsed into “Other”.  
  - Cross-sectional z-scoring by date.
- **Feature configuration** (`configs/features.yml`)  
  - 10 core Bloomberg risk factors (Size, Value, Momentum, etc.).  
  - Candidate extra features (e.g., Beta:Y-1, Market Cap).  
  - Selected extras list to define the final numeric feature set.
- **Processed panel**  
  - `notebooks/1.3_feature_selection.ipynb` writes a cleaned, feature-ready panel to  
    `data/processed/factor_panel.parquet`.

### Phase 2: Models & Tuning ✅

- **Model archetypes** (`src/lrcm/models.py`)  
  - `BayesianRidge` from `configs/models/linear_bayesridge.yml`.  
  - `RandomForestRegressor` from `configs/models/rf.yml`.  
  - `XGBRegressor` from `configs/models/xgb_champion.yml`.
- **Model bake-off** (`notebooks/2.1_model_bakeoff.ipynb`)  
  - 5-fold CV R² comparison across archetypes on the processed panel.  
  - Results written to `experiments/2.1_bakeoff_summary.csv`.
- **Hyperparameter tuning** (`notebooks/2.2_hyperparam_tuning.ipynb`)  
  - Optuna search over XGBoost hyperparameters.  
  - Trial log saved to `experiments/2.2_tuning_study.csv`.  
  - Best params merged back into `configs/models/xgb_champion.yml`.
- **Champion model evaluation** (`notebooks/2.3_champion_model.ipynb`)  
  - Time-based train/test split.  
  - Static out-of-sample R² reported for the tuned XGBoost model.

### Phase 3: Explainability & Walk-Forward Backtest ✅

- **Explainability** (`notebooks/3.1_explaining_predictions.ipynb`)  
  - SHAP TreeExplainer on the tuned XGBoost.  
  - Global feature importance and dependence plots for key factors.  
  - Local explanations for specific stock–date combinations (force plots).
- **Walk-forward backtest** (`notebooks/3.2_walk_forward_backtest.ipynb`)  
  - Rolling/expanding walk-forward engine in `src/lrcm/backtest.py`.  
  - At each date:
    - Train on past window,
    - Score current universe,
    - Long/short decile portfolios by predicted return.  
  - Per-period metrics (rank IC, long/short alpha, benchmark returns).  
  - Annual summaries written to `experiments/3.2_walkforward_metrics_by_year.csv`.  
  - Equity curve plot of cumulative long/short alpha.

---

## Data Universe

The exact contents of the raw file depend on your Bloomberg export, but the
pipeline assumes:

- **Time span:** Multi-year factor panel (e.g. ~1999–2025).  
- **Universe:** Cross-sectional equity universe (e.g. large/mid/small cap stocks).  
- **Frequency:** Panel keyed by `Date` and `Ticker` (e.g. monthly or quarterly codes).  
- **Core fields:**
  - `Date` (e.g. `199901`, `202301`),
  - `Ticker`,
  - 10 core risk factors (`Sz`, `Prof`, `Vol`, `Trd Act`, `Lev`, `Mom`, `Val`,
    `Gr`, `Dvd Yld`, `Earn Var`),
  - Extra features (e.g. `Beta:Y-1`, `Market Cap`, `Total Return:Y-1`, etc.),
  - Forward return column(s) (`FwdRetOrig` raw, `FwdRet` clipped),
  - GICS classifications (`GICS_Sector_Name`, `GICS_Industry_Name`,
    `GICS_SubInd_Name`).

If your schema differs, you can:

- change column names in `configs/data.yml` and `configs/features.yml`, and  
- adjust the feature lists there without touching the core code.

---

## Project Structure

```text
production-ml-trading/
├── README.md                  # Project description and usage
├── .gitignore                 # Ignore data, caches, etc.
├── environment.yml            # Conda environment definition
├── src/
│   └── lrcm/
│       ├── __init__.py        # Package entry, exports helpers
│       ├── data_loading.py    # Load raw / processed panels via configs
│       ├── preprocessing.py   # Clipping, GICS handling, z-scores
│       ├── features.py        # FeatureConfig + final feature list
│       ├── models.py          # Model constructors + CV helpers
│       ├── backtest.py        # Walk-forward long/short engine
│       ├── evaluation.py      # IC, alpha, Sharpe, drawdowns, yearly summaries
│       └── plots.py           # SHAP plots and equity curve helpers
├── notebooks/
│   ├── 1.1_data_overview.ipynb
│   ├── 1.2_feature_analysis.ipynb
│   ├── 1.3_feature_selection.ipynb
│   ├── 2.1_model_bakeoff.ipynb
│   ├── 2.2_hyperparam_tuning.ipynb
│   ├── 2.3_champion_model.ipynb
│   ├── 3.1_explaining_predictions.ipynb
│   └── 3.2_walk_forward_backtest.ipynb
├── configs/
│   ├── data.yml               # Paths + core column names
│   ├── features.yml           # Risk factors, extra features, clipping, GICS
│   ├── models/
│   │   ├── linear_bayesridge.yml
│   │   ├── rf.yml
│   │   └── xgb_champion.yml   # Tuned XGB hyperparameters
│   └── backtest/
│       └── walk_forward_tv_style.yml  # Backtest window and portfolio params
├── experiments/
│   ├── README.md             
│   └── figures/
│       └── .gitkeep          
├── data/
│   ├── raw/                   # Raw panel file (not tracked)
│   └── processed/             # Processed panel(s) (not tracked)
└── paper/
    ├── main.tex               # LaTeX entry point for the paper
    ├── sections/              # Intro/Methods/Results/Discussion
    └── figures/               # Final plots for the paper
```

---
## Environment

Create the conda environment
```bash

    conda env create -f environment.yml
    conda activate production-ml-trading
```
Then launch Jupyter:
```bash
  jupyter lab
```
---

## How to run the pipeline

From a fresh clone, the recommended order is:

### 1. Inspect and preprocess the data

#### 1.1 Data overview

**Notebook:** `notebooks/1.1_data_overview.ipynb`

What this notebook does:

- Loads the raw factor panel from `data/raw/` using the path and column names in `configs/data.yml`.
- Prints basic info about the dataset:
  - available columns and data types,
  - date coverage and number of periods,
  - number of stocks per date,
  - basic missingness patterns and summary statistics.

Why it matters:

- This is your sanity check that the Bloomberg export is read correctly.
- If column names or dtypes don’t match expectations, you’ll see it here before you start modeling.

---

#### 1.2 Feature analysis

**Notebook:** `notebooks/1.2_feature_analysis.ipynb`

What this notebook does:

- Uses `configs/features.yml` to define:
  - **core risk factors** (the main 10 Bloomberg risk columns),
  - **extra candidate features** (e.g. beta, size proxies, etc.).
- Computes:
  - pairwise correlations between core and extra features,
  - simple summary stats for each feature.
- Saves a correlation summary to:

```text
experiments/1.2_extra_feature_correlations.csv
```

Why it matters:

- You can quickly see which extra features are basically duplicates of existing factors and which might add new informations
- This guides the final feature set used by the models.

#### 1.3 Feature selection and preprocessing

**Notebook:** `notebooks/1.3_feature_selection.ipynb`

What this notebook does:
- Appplies the preprocessing logic defined in `src/lrcm/preprocessing.py`:
  - Clips extreme forward returns cross-sectionally to reduce the influence of outliers.
  - Handles rare GICS categories (e.g. very small industries/sectors) by collapsing them into "Other" bucket.
  - Standardizes numeric features cross-sectionally (z-score within each date)
- Produces a clean, model-ready panel and writes it to:
    ```text
      data/processed/factor_panel.parquet
    ```
(Exact path is controlled by `configs/data.yml`.)

Why it matters:
- From this point on, all models and backtests use this processed panel.
- You get a consistent, documented preprocessing step instead of ad-hoc cleaning inside each notebook.

---

