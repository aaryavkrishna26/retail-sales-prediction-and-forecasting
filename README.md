# Retail Sales Prediction and Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle: Store Sales](https://img.shields.io/badge/Kaggle-Store%20Sales%20Forecasting-20BEFF.svg)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

An end-to-end predictive analytics and time series forecasting system built on the **Kaggle Store Sales - Time Series Forecasting** dataset (Corporación Favorita, Ecuador). This repository features data preprocessing, exploratory data analysis, regularized linear regression for continuous sales forecasting, logistic regression for high-demand surge classification, and statistical time series models evaluated against the official competition metric (RMSLE).

---

## 📌 Problem Overview

Accurate store sales forecasting enables grocery retailers to balance inventory levels, prevent stockouts during promotional surges, and minimize spoilage of perishable goods. This repository tackles two complementary predictive challenges:

1. **Continuous Sales Volume Forecasting (Regression & Time Series):**
   - Predict daily unit sales across 54 stores and 33 product families ($1,782$ series).
   - Incorporates promotional activity, calendar seasonality, payday surges (15th and month-end), holiday schedules, and macroeconomic crude oil price fluctuations.
   - Evaluated using Root Mean Squared Logarithmic Error (**RMSLE**), Root Mean Squared Error (**RMSE**), Mean Absolute Error (**MAE**), and **$R^2$**.

2. **High-Demand Surge Classification (Logistic Regression):**
   - Probabilistic classification of demand spikes (sales exceeding the 80th percentile for a given store-family group).
   - Optimizes buffer stocking and alerts supply chain planners to high-risk inventory depletion days.
   - Evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

---

## 📂 Project Architecture

```text
retail-sales-prediction-and-forecasting/
├── .env.example              # Template for environment configuration
├── .gitignore                # Protects datasets, credentials, caches, & venvs
├── PROJECT_DETAILS.md        # Technical architecture and design details
├── README.md                 # Project documentation and quickstart guide
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Raw Kaggle CSVs (git-ignored)
│   │   ├── README.md         # Data dictionary & download instructions
│   │   └── .gitkeep
│   └── processed/            # Feature-engineered datasets (git-ignored)
│       └── .gitkeep
├── models/                   # Serialized model pipelines (.joblib)
│   └── .gitkeep
├── notebooks/                # Modular Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_linear_regression_modeling.ipynb
│   ├── 04_logistic_regression_classification.ipynb
│   ├── 05_time_series_forecasting.ipynb
│   └── 06_model_evaluation_and_comparison.ipynb
├── reports/                  # Evaluation summaries and reports
│   ├── figures/              # Generated plots and figures
│   │   └── .gitkeep
│   ├── academic_project_report.md
│   ├── eda_summary.md
│   └── model_evaluation_metrics.md
└── src/                      # Reusable Python source modules
    ├── __init__.py           # Package initializer
    ├── config.py             # Paths, random seeds, and cutoff dates
    ├── data_loader.py        # Dataset ingestion & availability checks
    ├── eda.py                # Analytical plotting routines
    ├── evaluate.py           # RMSLE, regression, and classification metrics
    ├── linear_regression.py  # OLS, Ridge, and Lasso regression pipelines
    ├── logistic_regression.py# Binary surge classification pipelines
    ├── preprocess.py         # Cleaning, calendar features, lags, & splits
    └── time_series.py        # Seasonal baseline & SARIMAX forecaster
```

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/aaryavkrishna26/retail-sales-prediction-and-forecasting.git
cd retail-sales-prediction-and-forecasting
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Kaggle Dataset
Download the competition files and place them into `data/raw/`:
```bash
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/
cd data/raw
unzip store-sales-time-series-forecasting.zip
```
Expected files in `data/raw/`:
- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`

---

## 🔬 Analytical Workflow

| Phase | Module / Notebook | Description |
| :--- | :--- | :--- |
| **1. Data Ingestion** | `src/data_loader.py` | Validates raw CSV existence and loads typed DataFrames. |
| **2. Feature Engineering** | `src/preprocess.py` | Interpolates oil prices, extracts calendar/payday features, creates non-leaking lags ($\ge 16$ days), and applies temporal splits. |
| **3. Exploratory Analysis** | `src/eda.py` / `01_eda.ipynb` | Analyzes aggregate sales trends, oil price correlations, and holiday impacts. |
| **4. Linear Regression** | `src/linear_regression.py` | Fits OLS, Ridge ($L_2$), and Lasso ($L_1$) on $\log(1 + \text{sales})$ to directly optimize RMSLE. |
| **5. Logistic Regression** | `src/logistic_regression.py` | Models high-demand surge events with class balancing and threshold calibration. |
| **6. Time Series Analysis** | `src/time_series.py` | Evaluates naive seasonal baselines and SARIMAX models with exogenous features. |
| **7. Model Benchmarking** | `src/evaluate.py` | Computes RMSLE, RMSE, MAE, $R^2$, Accuracy, Precision, Recall, F1, and ROC-AUC. |

---

## 🔒 Security & Data Hygiene
- **Data Protection:** All CSV, Parquet, and raw data files are strictly excluded from version control via `.gitignore`.
- **Credential Safety:** Secrets (`.env`, `kaggle.json`, private keys, API tokens) are ignored.
- **Model Checkpoints:** Binary weights (`.joblib`, `.pkl`) are excluded from Git to maintain a clean repository history.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
