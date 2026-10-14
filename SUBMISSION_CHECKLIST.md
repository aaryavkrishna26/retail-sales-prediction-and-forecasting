# College Project Submission Checklist

**Project:** Retail Sales Prediction and Forecasting: Multi-Phase Machine Learning and Econometric Time Series Framework for Grocery Demand Analytics  
**Repository Directory:** `C:\retail-sales-prediction-and-forecasting`  
**GitHub URL:** `https://github.com/aaryavkrishna26/retail-sales-prediction-and-forecasting`  
**Submission Status:** Pre-Submission Review & Quality Assurance  

---

## 1. Source Code & Modular Python Pipeline Verification

- [ ] **Package Structure & Initializers:**
  - [ ] `src/__init__.py` exists and is properly formatted.
  - [ ] All source modules import cleanly without unresolved relative path errors.
- [ ] **Data Ingestion & Integrity Auditing:**
  - [ ] `src/data_loader.py`: Ingestion logic with memory-efficient dtype casting (`int8`, `int16`, `float32`, `category`).
  - [ ] `src/audit_data.py`: Schema validation script for checking expected CSV tables in `data/raw/`.
- [ ] **Preprocessing & Leakage Prevention:**
  - [ ] `src/preprocess.py`: Implements oil price interpolation, holiday calendar harmonization, bi-weekly payday flagging, and $\ge 16$-day lag generation.
  - [ ] Zero target leakage audit confirmed: `transactions.csv` is excluded from feature matrices, and lag features strictly respect the 16-day horizon cutoff.
- [ ] **Core Modeling Engines:**
  - [ ] `src/linear_regression.py`: Implements DummyRegressor, SeasonalNaiveRegressor, OLS, Ridge, Lasso, and ElasticNet pipelines with log-space target transformation and non-negative clipping.
  - [ ] `src/logistic_regression.py`: Implements DummyClassifier, Default Logistic Regression, Balanced Logistic Regression, and out-of-sample decision threshold calibration.
  - [ ] `src/time_series.py`: Implements SeasonalNaiveForecaster, HoltWintersForecaster, SARIMAForecaster, and SARIMAXForecaster with exogenous regressor support.
- [ ] **Standardized Metric Evaluator:**
  - [ ] `src/evaluate.py`: Implements RMSLE, RMSE, MAE, $R^2$, Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC.
- [ ] **Automated Phase Execution Runners:**
  - [ ] `src/run_eda.py`: Generates EDA charts and markdown summaries.
  - [ ] `src/run_linear_regression.py`: Runs Phase 1 benchmark comparison and exports CSV/figures/model.
  - [ ] `src/run_logistic_regression.py`: Runs Phase 2 surge classification and exports CSV/figures/model.
  - [ ] `src/run_time_series.py`: Runs Phase 3 macro and micro cohort forecasting and exports CSV/figures/model.

---

## 2. Interactive Jupyter Notebooks Verification

- [ ] **Notebook Sequence & Numbering:**
  - [ ] `notebooks/01_exploratory_data_analysis.ipynb`: Comprehensive EDA, store clustering, payday lift, and macroeconomic oil analysis.
  - [ ] `notebooks/02_preprocessing_and_feature_engineering.ipynb`: Step-by-step feature engineering and chronological train-val splitting.
  - [ ] `notebooks/03_linear_regression_modeling.ipynb`: Regularization path exploration (Ridge, Lasso, Elastic Net) and residual diagnostics.
  - [ ] `notebooks/04_logistic_regression_modeling.ipynb`: Canonical Phase 2 demand surge classification notebook.
  - [ ] `notebooks/04_logistic_regression_classification.ipynb`: Synchronized mirror notebook preserved.
  - [ ] `notebooks/05_time_series_forecasting.ipynb`: Canonical Phase 3 econometric forecasting notebook.
  - [ ] `notebooks/05_time_series_modeling.ipynb`: Synchronized mirror notebook preserved.
  - [ ] `notebooks/06_model_evaluation_and_comparison.ipynb`: Cross-phase benchmark synthesis and visualization.
- [ ] **Notebook Preservation:**
  - [ ] All notebooks preserved without accidental deletion of outputs or duplicate files.

---

## 3. Project Documentation & Specifications

- [ ] **`README.md` Completeness:**
  - [ ] Complete project title, overview, and problem statement.
  - [ ] Dataset source, hierarchy (54 stores, 33 families, 1,782 series), and primary tables.
  - [ ] Technologies, libraries, and version requirements clearly listed.
  - [ ] Full directory tree mapping out the entire repository.
  - [ ] Detailed explanation of the three modeling phases.
  - [ ] Exact quantitative benchmark tables matching documented project runs.
  - [ ] Mathematical definitions and interpretations of all evaluation metrics.
  - [ ] Installation instructions, virtual environment setup, and execution commands.
  - [ ] Explanation of all reports, figures, notebooks, and serialized model files.
  - [ ] Project limitations and future research directions.
  - [ ] License and references cited.
- [ ] **`PROJECT_DETAILS.md` Architecture Document:**
  - [ ] In-depth technical architecture, module descriptions, and data flow.
  - [ ] Mathematical formulations for OLS, Ridge, Lasso, Elastic Net, Logistic Regression, Holt-Winters, and SARIMAX.
  - [ ] Chronological splitting protocol and zero-leakage guarantee.
  - [ ] Supply chain operational trade-offs (holding costs vs. stockout penalties).
- [ ] **Environment & Dependency Files:**
  - [ ] `requirements.txt`: Clean, version-pinned production dependencies.
  - [ ] `.env.example`: Template for environment variables and paths.
  - [ ] `LICENSE`: MIT open-source license file included.

---

## 4. Evaluation Reports, Metrics Tables & Visualizations

- [ ] **Technical Benchmark Reports (`reports/`):**
  - [ ] `reports/academic_project_report.md`: Formal academic summary.
  - [ ] `reports/eda_summary.md`: Quantitative EDA findings and distributions.
  - [ ] `reports/linear_regression_comparison.md`: Detailed Phase 1 regression comparison.
  - [ ] `reports/logistic_regression_evaluation.md`: Detailed Phase 2 classification evaluation.
  - [ ] `reports/time_series_evaluation.md`: Detailed Phase 3 econometric forecasting evaluation.
  - [ ] `reports/model_evaluation_metrics.md`: Master reference documenting all metric tables across all phases.
- [ ] **Quantitative Metric CSV Files:**
  - [ ] `reports/linear_regression_comparison.csv`
  - [ ] `reports/logistic_regression_metrics.csv`
  - [ ] `reports/time_series_metrics.csv`
- [ ] **High-Resolution Diagnostic Visualizations (`reports/figures/`):**
  - [ ] *Exploratory Data Analysis Figures:*
    - [ ] `daily_sales_trend.png`
    - [ ] `top_families.png`
    - [ ] `payday_impact.png`
    - [ ] `day_of_week_seasonality.png`
    - [ ] `store_types_and_clusters.png`
    - [ ] `oil_sales_relationship.png`
    - [ ] `promotion_sales_impact.png`
  - [ ] *Phase 1 Linear Regression Figures:*
    - [ ] `linear_regression_actual_vs_pred.png`
    - [ ] `linear_regression_residuals.png`
    - [ ] `linear_regression_model_comparison.png`
    - [ ] `linear_regression_coefficients.png`
  - [ ] *Phase 2 Logistic Regression Figures:*
    - [ ] `logistic_regression_confusion_matrices.png`
    - [ ] `logistic_regression_roc_pr_curves.png`
    - [ ] `logistic_regression_coefficients.png`
  - [ ] *Phase 3 Time Series Figures:*
    - [ ] `time_series_aggregated_forecast.png`
    - [ ] `time_series_cohort_forecasts.png`
    - [ ] `time_series_model_comparison.png`
    - [ ] `time_series_residuals.png`

---

## 5. Serialized Model Artifacts (`models/`)

- [ ] **Fitted Production Pipelines:**
  - [ ] `models/best_linear_model.joblib` (~6 KB): Serialized regularized Ridge regression pipeline.
  - [ ] `models/best_logistic_model.joblib` (~40 KB): Serialized Logistic Regression model and surge threshold dictionary.
  - [ ] `models/best_time_series_model.joblib` (~18 MB): Serialized SARIMAX state-space model artifact.
  - [ ] Model files are properly referenced in documentation and reproducible via script runners.

---

## 6. Viva Voce Examination & Defense Preparation

- [ ] **`VIVA_PREPARATION.md` Document Review:**
  - [ ] Section A: Project fundamentals, problem statement, and supply chain context.
  - [ ] Section B: Linear regression theory, regularization ($L_1/L_2$), non-negativity clipping, and Phase 1 results.
  - [ ] Section C: Logistic regression, surge thresholds, class imbalance, and Phase 2 trade-offs.
  - [ ] Section D: Time series decomposition, Holt-Winters, SARIMA/SARIMAX orders, and Phase 3 results.
  - [ ] Section E: Data leakage, chronological vs. k-fold splitting, and negative $R^2$ mathematical justification.
  - [ ] Section F: Codebase walkthrough and artifact mapping.
  - [ ] Section G: 45 categorized viva questions with concise model answers studied and understood.

---

## 7. Version Control, Hygiene & Security Audit

- [ ] **Git Exclusion Rules (`.gitignore`):**
  - [ ] Raw datasets excluded: `data/raw/*.csv`, `data/raw/*.zip`.
  - [ ] Processed temporary files excluded: `data/processed/*`.
  - [ ] Virtual environments excluded: `.venv/`, `venv/`, `env/`.
  - [ ] Python cache files excluded: `__pycache__/`, `*.pyc`.
  - [ ] Sensitive secrets excluded: `.env`, `kaggle.json`, private keys, API tokens.
  - [ ] OS-specific artifacts excluded: `.DS_Store`, `Thumbs.db`.
- [ ] **Pre-Commit Repository State:**
  - [ ] No extraneous temporary test scripts or scratch dumps in project root.
  - [ ] All filenames use standard, clean naming conventions.
  - [ ] Changes prepared and staged without unapproved commits or pushes.

---

## 8. College Submission Deliverables Review

- [ ] **Digital Submission Folder Structure:**
  - [ ] Confirm all project files, notebooks, reports, figures, and serialized models are intact.
  - [ ] Verify that no raw data files are accidentally included in the submission archive (provide download script/instructions instead).
- [ ] **Turnitin / Plagiarism Report (Institutional Requirement):**
  - [ ] If required by university submission guidelines, generate the official Turnitin originality report through your university portal.
  - [ ] Attach the official PDF originality report alongside the submitted project bundle.
  - [ ] Ensure similarity indices on standard code headers and libraries are accounted for according to departmental guidelines.
- [ ] **Final Student Declaration:**
  - [ ] Verify that author details, student ID, and date are updated in `reports/academic_project_report.md`.
  - [ ] Confirm that all experimental metrics represent genuine, non-fabricated project results.
