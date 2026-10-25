Retail Sales Prediction and Forecasting

Project Team Members

S. No.| Name| Student ID
1| Aaryav Krishna| 2427030754
2| Rishit Khanna| 2427030757

An end-to-end Machine Learning, Statistical Econometrics, and Predictive Analytics project developed for university submission using the Kaggle Store Sales — Time Series Forecasting dataset from Corporación Favorita, Ecuador.

The project explores three modeling approaches:

1. Linear Regression: Predict continuous retail sales.
2. Logistic Regression: Classify high-demand sales surges.
3. Time-Series Forecasting: Forecast future sales using historical patterns.

1. Project Overview

Retail businesses need accurate sales forecasts to maintain inventory, prevent stockouts, reduce overstocking, and improve replenishment planning.

This project develops a predictive analytics framework using historical grocery sales, promotions, calendar information, store metadata, holidays, and oil prices.

The system evaluates regression, classification, and classical time-series models to understand retail demand and predict future sales.

Project domain: Retail Analytics, Supply Chain Analytics, Machine Learning, and Econometrics.

2. Problem Statement

Retail sales vary because of seasonal patterns, promotions, weekends, holidays, and changes in consumer demand.

The project addresses three key problems:

- Predicting daily sales for different stores and product families.
- Identifying unusually high-demand periods that may increase stockout risk.
- Forecasting future sales using historical data and time-series methods.

3. Dataset Description

Dataset: Store Sales — Time Series Forecasting
Source: Kaggle
Publisher: Corporación Favorita, Ecuador

Main Dataset Files

File| Description
"train.csv"| Historical sales data with 3,000,888 rows and 6 columns
"test.csv"| Unlabelled sales records for the forecasting period
"stores.csv"| Store locations, types, and clusters
"oil.csv"| Historical daily oil prices
"holidays_events.csv"| Public holidays and regional events
"transactions.csv"| Historical store transaction counts

The training dataset covers 1 January 2013 to 15 August 2017.

The Kaggle test dataset covers 16 August 2017 to 31 August 2017.

The dataset includes 54 stores and 33 product families, resulting in 1,782 store-product-family time series.

Target Variables

Phase 1 — Linear Regression

- Target: "sales"
- Objective: Predict numerical daily sales values.

Phase 2 — Logistic Regression

- Target: Binary demand surge label.
- Class 1: Sales exceed the historical 80th-percentile threshold for the relevant store and product family.
- Class 0: Sales are at or below that threshold.

Phase 3 — Time-Series Forecasting

- Target: "sales" over time.
- Objective: Forecast future sales at the enterprise level and for selected store-product-family combinations.

4. Technologies and Libraries

- Programming language: Python
- Data processing: Pandas, NumPy, SciPy
- Machine learning: Scikit-learn
- Statistical modeling: Statsmodels
- Visualization: Matplotlib, Seaborn
- Model storage: Joblib
- Development tools: Jupyter Notebook, VS Code, Git, and GitHub

5. Project Workflow

The project follows this workflow:

1. Load and audit the raw dataset.
2. Explore sales distributions and historical trends.
3. Clean data and handle missing values.
4. Create calendar, promotion, lag, and rolling-window features.
5. Split the data chronologically into training and validation periods.
6. Train and compare models across three phases.
7. Evaluate model performance using appropriate metrics.
8. Generate plots, reports, and saved model artifacts.
9. Use trained models to generate sales predictions and forecasts.

6. Data Preprocessing and Feature Engineering

The preprocessing pipeline includes:

- Handling missing oil-price observations.
- Extracting year, month, day, and day-of-week features.
- Creating weekend and payday indicators.
- Mapping holiday and regional-event information.
- Incorporating promotion counts.
- Creating historical lag and rolling-window sales features.
- Applying the transformation "log(1 + sales)" where appropriate.
- Preventing target leakage by using only information available at the forecasting origin.

The lag and rolling features use a minimum 16-day shift to respect the intended multi-step forecasting horizon.

The transaction-count table is excluded from predictive features because future transaction counts are not known in advance at prediction time.

7. Modeling Architecture

Phase 1: Linear Regression — Sales Prediction

Objective: Predict continuous daily sales across 1,782 store-product-family series.

Algorithms evaluated:

- Dummy Regressor
- Seasonal Naive Baselines
- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- Elastic Net

The models use historical sales patterns, store information, promotions, and calendar features.

Training and Validation Periods

- Training: 1 August 2016 to 31 July 2017
- Validation: 1 August 2017 to 15 August 2017
- Training observations: 648,648
- Validation observations: 26,730

Linear Regression Results

Model| RMSLE| RMSE| MAE| R²
OLS| 0.6472| 470.97| 128.56| 0.8567
Ridge| 0.6481| 470.02| 128.31| 0.8573
Lasso| 0.6539| 456.40| 121.99| 0.8655
Elastic Net| 0.6516| 459.00| 123.97| 0.8639
Seasonal Naive (Lag 16–22)| 0.6388| 356.20| 92.17| 0.9180

Key finding: Lasso achieved the highest R² among the evaluated linear regression models. However, the Seasonal Naive (Lag 16–22) baseline performed better on the reported validation metrics.

Phase 2: Logistic Regression — Demand Surge Classification

Objective: Predict whether sales will exceed the historical 80th-percentile threshold for a store and product family.

The target is binary:

- "1": Demand surge
- "0": Normal demand

Algorithms evaluated:

- Dummy Classifier
- Logistic Regression with the default threshold
- Class-balanced Logistic Regression
- Threshold-tuned Logistic Regression

Logistic Regression Results

Model| Accuracy| Precision| Recall| F1-score| ROC-AUC
Dummy Classifier| 0.8330| 0.0000| 0.0000| 0.0000| 0.5000
Default Logistic Regression| 0.8212| 0.4524| 0.3363| 0.3858| 0.7518
Balanced Logistic Regression| 0.6151| 0.2705| 0.7694| 0.4003| 0.7529
Tuned Logistic Regression| 0.7465| 0.3416| 0.5590| 0.4241| 0.7518

The tuned decision threshold was approximately "0.2897".

Key finding: The tuned Logistic Regression model achieved the highest F1-score of 0.4241. The balanced model achieved a recall of 76.94%, making it more suitable when missing a genuine demand surge is costly.

Accuracy alone is insufficient for this task because the dataset contains substantially more normal-demand observations than surge observations.

Phase 3: Time-Series Forecasting

Objective: Forecast future sales using historical trends, weekly seasonality, and relevant external variables.

Algorithms evaluated:

- Seasonal Naive
- Holt-Winters Exponential Smoothing
- SARIMA
- SARIMAX

The models were evaluated at two levels:

1. Enterprise-level aggregated daily sales.
2. Five representative store-product-family series.

Enterprise-Level Forecasting Results

Model| RMSLE| RMSE| MAE| R²
Seasonal Naive| 0.1574| 139,177.32| 100,229.67| -0.5537
Holt-Winters| 0.1295| 115,406.36| 82,557.60| -0.0683
SARIMA| 0.1478| 130,393.45| 91,265.18| -0.3638
SARIMAX| 0.1437| 121,980.13| 90,145.81| -0.1935

Holt-Winters achieved the lowest RMSLE, RMSE, and MAE in the enterprise-level evaluation.

However, all four models had negative R² scores on this short validation window, indicating that they did not outperform the validation-mean benchmark under that metric.

Key finding: Forecasting performance varies between aggregate sales and individual store-product-family series. No single forecasting method is best for every series.

8. Evaluation Metrics

Regression and Forecasting

- RMSLE: Measures the difference between logarithmically transformed actual and predicted sales.
- RMSE: Penalizes large prediction errors.
- MAE: Measures average absolute error in sales units.
- R²: Measures explained variance relative to the validation-mean baseline.

Classification

- Accuracy: Percentage of correct classifications.
- Precision: Percentage of predicted surges that were actual surges.
- Recall: Percentage of actual surges successfully identified.
- F1-score: Harmonic mean of precision and recall.
- ROC-AUC: Measures how well the model distinguishes surge from normal-demand observations across thresholds.
- PR-AUC: Evaluates precision-recall performance, particularly for imbalanced classes.

9. Project Directory Structure

retail-sales-prediction-and-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_linear_model.joblib
│   ├── best_logistic_model.joblib
│   └── best_time_series_model.joblib
│
├── notebooks/
│   ├── 01_data_exploration_and_preprocessing.ipynb
│   ├── 02_linear_regression.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_time_series_forecasting.ipynb
│   └── 05_final_model_comparison.ipynb
│
├── reports/
│   ├── figures/
│   ├── academic_project_report.md
│   ├── eda_summary.md
│   ├── linear_regression_comparison.md
│   ├── logistic_regression_evaluation.md
│   ├── time_series_evaluation.md
│   └── model_evaluation_metrics.md
│
├── src/
│   ├── audit_data.py
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── preprocess.py
│   ├── run_eda.py
│   ├── run_linear_regression.py
│   ├── run_logistic_regression.py
│   ├── run_time_series.py
│   └── time_series.py
│
├── README.md
├── PROJECT_DETAILS.md
├── SUBMISSION_CHECKLIST.md
├── VIVA_PREPARATION.md
├── requirements.txt
├── LICENSE
└── .gitignore

10. Installation and Environment Setup

Step 1: Clone the Repository

git clone https://github.com/aaryavkrishna26/retail-sales-prediction-and-forecasting.git
cd retail-sales-prediction-and-forecasting

Step 2: Create a Virtual Environment

python -m venv .venv

Activate it on Windows PowerShell:

.venv\Scripts\Activate.ps1

Step 3: Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Step 4: Add the Dataset

Download the competition data from Kaggle and place the CSV files inside "data/raw/".

Required files include:

- "train.csv"
- "test.csv"
- "stores.csv"
- "oil.csv"
- "holidays_events.csv"
- "transactions.csv"
- "sample_submission.csv"

11. Running the Project

Run these commands from the repository root.

Data Audit

python src/audit_data.py

Exploratory Data Analysis

python src/run_eda.py

Phase 1: Linear Regression

python src/run_linear_regression.py

Phase 2: Logistic Regression

python src/run_logistic_regression.py

Phase 3: Time-Series Forecasting

python src/run_time_series.py

These scripts are intended to generate evaluation reports, figures, and saved model artifacts. Their outputs depend on the installed dependencies, available data, and current source code.

12. Reports and Visualizations

The "reports/" directory contains evaluation reports and comparison tables.

The "reports/figures/" directory includes visualizations such as:

- Daily sales trends
- Day-of-week seasonality
- Promotion impact
- Payday impact
- Store types and clusters
- Actual versus predicted sales
- Regression residuals
- Logistic Regression confusion matrices
- ROC and precision-recall curves
- Time-series forecasts
- Forecasting model comparisons

The saved models in "models/" are intended to store the trained pipelines and forecasting artifacts.

13. Limitations

1. Linear models may not capture complex nonlinear relationships between promotions, stores, and product families.
2. Some store-product-family series are sparse or intermittent, making time-series forecasting difficult.
3. The main validation uses a single 15-day holdout, so performance may differ in other seasons.
4. Promotion counts do not capture item-level discount depth.
5. The reported classification threshold and evaluation metrics depend on the selected training and validation protocol.
6. Model artifacts should be regenerated whenever the training code, model selection, or preprocessing pipeline changes.

14. Future Improvements

- Evaluate gradient boosting models such as LightGBM, XGBoost, and CatBoost.
- Explore deep-learning forecasting models such as DeepAR, N-BEATS, and Temporal Fusion Transformers.
- Introduce rolling-origin temporal cross-validation.
- Improve demand surge detection using operational stockout costs.
- Implement hierarchical forecasting across product families, stores, and enterprise-level demand.
- Develop a dashboard for interactive sales prediction and forecasting.

15. Academic Integrity and Data Hygiene

- Raw and processed datasets are excluded from version control where configured in ".gitignore".
- Credentials, environment files, and virtual environments should not be committed.
- Evaluation results should be reported using the documented experimental protocol.
- All reported results should be reproducible from the corresponding code and data.
- No model should be described as production-ready without appropriate testing and validation.

16. License and References

License: MIT License

Dataset: Corporación Favorita, Store Sales — Time Series Forecasting, Kaggle.

References:

1. Hyndman, R. J., and Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd edition.
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control, 5th edition.
3. Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical Learning, 2nd edition.

---

Project Repository: "Retail Sales Prediction and Forecasting on GitHub" (https://github.com/aaryavkrishna26/retail-sales-prediction-and-forecasting)
