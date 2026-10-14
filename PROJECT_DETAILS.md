# Retail Sales Prediction and Forecasting - Technical Architecture & Project Details

## 1. Project Overview & Scope
This project implements an end-to-end Machine Learning and Time Series Forecasting system on the **Kaggle Store Sales - Time Series Forecasting** dataset (Corporación Favorita, Ecuador). The repository is structured to address both regression (continuous unit sales prediction) and classification (demand surge and stockout prevention) objectives.

---

## 2. Core Methodologies & Modules

### 2.1 Data Ingestion & Validation (`src/data_loader.py`)
- Automated availability checks for all primary Kaggle data tables (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`).
- Explicit memory-efficient data types (`int16`, `int8`, `category`, `float32`).

### 2.2 Preprocessing & Feature Engineering (`src/preprocess.py`)
- **Macroeconomic Oil Signal:** Daily interpolation of missing Brent crude oil prices (`dcoilwtico`).
- **Calendar & Payday Decomposition:** Extraction of day of week, month, year, weekend indicators, and bi-weekly Ecuador payday indicators (15th and month-end).
- **Holiday Integration:** Mapping national, regional, and municipal holidays, accounting for transferred holiday events.
- **Lag & Rolling Window Features:** Computation of non-leaking lag features ($\ge 16$ days, matching the test horizon).
- **Surge Classification Target:** Formation of binary target `is_high_sales` based on percentile demand thresholds per store-family pair.

### 2.3 Exploratory Data Analysis (`src/eda.py`)
- Aggregate sales volume trend visualization across the 2013–2017 timeline.
- Product family ranking by total sales.
- Payday demand variance boxplot analysis.
- Macroeconomic oil price vs. retail sales correlation plotting.

### 2.4 Regression Pipeline (`src/linear_regression.py`)
- Standardized linear regression pipelines: Ordinary Least Squares (OLS), Ridge ($L_2$), and Lasso ($L_1$).
- Training on $\log(1 + y)$ transformed sales volume to directly optimize the Root Mean Squared Logarithmic Error (RMSLE).
- Model artifact serialization with `joblib`.

### 2.5 Classification Pipeline (`src/logistic_regression.py`)
- Probabilistic classification of demand surge days.
- Balanced class weighting and probability threshold tuning.
- Metric reporting: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 2.6 Time Series Forecaster (`src/time_series.py`)
- 7-day seasonal persistence naive baseline.
- AutoRegressive Integrated Moving Average with Exogenous factors (SARIMAX).

### 2.7 Model Evaluation & Benchmarking (`src/evaluate.py`)
- Official competition metric calculation (RMSLE):
  $$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\ln(1 + \hat{y}_i) - \ln(1 + y_i))^2}$$
- Standard regression metrics: RMSE, MAE, $R^2$.
- Standard classification metrics: Confusion matrix, Precision, Recall, F1, ROC-AUC.

---

## 3. Directory Layout
```
retail-sales-prediction-and-forecasting/
├── .env.example              # Environment variables template
├── .gitignore                # Comprehensive Git exclusion rules
├── PROJECT_DETAILS.md        # Technical architecture documentation
├── README.md                 # Project guide, installation, and overview
├── requirements.txt          # Production and experimentation dependencies
├── data/
│   ├── raw/                  # Kaggle Store Sales CSVs (git-ignored)
│   │   ├── README.md         # Download and placement instructions
│   │   └── .gitkeep
│   └── processed/            # Preprocessed datasets (git-ignored)
│       └── .gitkeep
├── models/                   # Serialized model weights (git-ignored)
│   └── .gitkeep
├── notebooks/                # Modular Jupyter exploration notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_linear_regression_modeling.ipynb
│   ├── 04_logistic_regression_classification.ipynb
│   ├── 05_time_series_forecasting.ipynb
│   └── 06_model_evaluation_and_comparison.ipynb
├── reports/                  # Project reports and documentation
│   ├── figures/              # Generated visualizations
│   │   └── .gitkeep
│   ├── academic_project_report.md
│   ├── eda_summary.md
│   └── model_evaluation_metrics.md
└── src/                      # Production Python source modules
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── eda.py
    ├── evaluate.py
    ├── linear_regression.py
    ├── logistic_regression.py
    ├── preprocess.py
    └── time_series.py
```
