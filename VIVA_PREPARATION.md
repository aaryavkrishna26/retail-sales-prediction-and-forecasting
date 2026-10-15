# Comprehensive Viva Voce Examination & Defense Preparation Guide

**Project Title:** Retail Sales Prediction and Forecasting: Multi-Phase Machine Learning and Econometric Time Series Framework for Grocery Demand Analytics  
**Dataset:** Corporación Favorita Grocery Store Sales (Kaggle Time Series Forecasting)  
**System Scope:** Linear Regression (Continuous Prediction), Logistic Regression (Surge Classification), and Statistical Econometrics (Time Series Forecasting)  

---

## 📑 Table of Contents
- [Section A: Project Fundamentals & Supply Chain Context](#section-a-project-fundamentals--supply-chain-context)
- [Section B: Phase 1 — Linear Regression & Regularization](#section-b-phase-1--linear-regression--regularization)
- [Section C: Phase 2 — Logistic Regression & Demand Surge Classification](#section-c-phase-2--logistic-regression--demand-surge-classification)
- [Section D: Phase 3 — Econometric & Statistical Time Series Forecasting](#section-d-phase-3--econometric--statistical-time-series-forecasting)
- [Section E: Temporal Integrity, Data Leakage & Evaluation Theory](#section-e-temporal-integrity-data-leakage--evaluation-theory)
- [Section F: Architectural Code Walkthrough & Artifact Reference](#section-f-architectural-code-walkthrough--artifact-reference)
- [Section G: 45 Comprehensive Viva Questions & Model Answers](#section-g-45-comprehensive-viva-questions--model-answers)

---

## Section A: Project Fundamentals & Supply Chain Context

### 1. Project Objective & Problem Statement
- **Objective:** Design, implement, and benchmark an end-to-end predictive analytics framework capable of forecasting continuous grocery sales volumes, classifying high-demand surge events, and projecting multi-step out-of-sample demand using econometric time-series models.
- **Problem Statement:** Large supermarket chains manage thousands of distinct products across geographically dispersed stores. Retail grocery demand fluctuates due to weekly customer shopping habits, bimonthly wage payments (payday effects), scheduled promotional campaigns, public holidays, and national macroeconomic shocks (e.g., crude oil price swings). 
- **The Core Trade-off:**
  - *Under-forecasting (Stockout):* Leads to empty shelves, lost direct revenue, and damaged customer retention.
  - *Over-forecasting (Overstock):* Leads to excessive working capital allocation, warehouse congestion, and high inventory spoilage/waste in perishable food categories (Produce, Dairy, Meat).

### 2. Why Retail Sales Prediction is Economically Vital
- **Working Capital Optimization:** Enables lean inventory replenishment and lowers safety stock carrying costs.
- **Perishable Spoilage Reduction:** Fresh food categories have strict expiration timelines; accurate day-of-week forecasts directly reduce food waste.
- **Logistical & Labor Efficiency:** Store managers schedule cashier, stocking, and warehouse staff according to anticipated footfall and transaction volume.

### 3. Dataset Architecture & Primary Columns
The project utilizes the **Corporación Favorita** dataset (Kaggle Store Sales Competition):
- **Spatial Scope:** 54 retail stores located across 17 cities and 16 provinces in Ecuador.
- **Product Hierarchy:** 33 unique product families (`GROCERY I`, `BEVERAGES`, `PRODUCE`, `CLEANING`, `DAIRY`, down to `BOOKS`).
- **Matrix Dimension:** $54 \text{ stores} \times 33 \text{ families} = 1,782$ distinct daily time series.
- **Primary Data Tables:**
  1. `train.csv`: Historical daily records (2013-01-01 to 2017-08-15; 3,000,888 observations) with columns `id`, `date`, `store_nbr`, `family`, `sales` (target), and `onpromotion`.
  2. `test.csv`: 16-day operational forecast horizon (2017-08-16 to 2017-08-31; 28,512 rows) with columns `id`, `date`, `store_nbr`, `family`, `onpromotion`.
  3. `stores.csv`: Metadata on `store_nbr`, `city`, `state`, `type` (A through E), and `cluster` (1 to 17).
  4. `oil.csv`: Daily Brent crude oil price (`dcoilwtico`).
  5. `holidays_events.csv`: Calendar holidays, event types (Local, Regional, National), and transfer flags.
  6. `transactions.csv`: Daily customer transaction counts per store. Strictly excluded from predictive features to prevent data leakage during multi-step forecasting.

### 4. Data Preprocessing & Feature Engineering
- **Oil Price Interpolation:** Weekend closures and market holidays cause 43 missing dates in oil prices. These are imputed via forward-fill followed by backward-fill.
- **Calendar & Payday Decomposition:**
  - `is_weekend`: Captures the +39.35% weekend demand lift.
  - `is_payday`: Binary flag indicating the 15th and the final calendar day of each month, capturing the bimonthly Ecuadorian wage cycle (+1.41% average demand lift).
  - `dayofweek`, `month`, `year`: Cyclic calendar coordinates.
- **Promotion Representations:** `onpromotion` (item count), `has_promotion` (binary indicator), and `log_promotion` ($\log(1 + \text{onpromotion})$).
- **Zero-Leakage Lag Features ($\ge 16$ Days):** In production, models must forecast 16 days forward without observing intermediate sales. Hence, all lag features (`sales_lag_16`, `sales_lag_21`, `sales_lag_28`, `sales_lag_35`) and rolling windows (`sales_rolling_mean_7`, `14`, `30`, `60`, and `sales_rolling_std_30`) use a minimum 16-day shift (`shift(16)`).
- **Dynamic Trailing 365-Day Baselines:**
  - `hist_sf_mean_log`: 365-day rolling average log sales for each store-family pair shifted by 16 days.
  - `hist_sf_dow_mean_log`: 52-week rolling average log sales for each store-family pair on that specific day of the week shifted by 16 days.
  - *Impact:* Incorporating these two dynamic features elevated linear regression validation $R^2$ from **0.0113** to **0.8573** (Ridge) and slashed RMSE by 62.4%.

### 5. Training and Validation Strategy
- **Why Random K-Fold Fails:** Randomly partitioning time-series records violates temporal causality by leaking future data into past predictions.
- **Strict Chronological Holdout:**
  - **Option B Training Window (1-Year Rolling):** `2016-08-01` to `2017-07-31` (648,648 rows across all 1,782 series).
  - **Validation Holdout Window:** `2017-08-01` to `2017-08-15` (15 days; 26,730 observations, identical across all models).
  - **Operational Test Horizon:** `2017-08-16` to `2017-08-31` (16 days; 28,512 observations).

---

## Section B: Phase 1 — Linear Regression & Regularization

### 1. What Linear Regression Is
Linear regression models the relationship between a dependent continuous target variable ($y$) and one or more independent explanatory features ($X$) by fitting a linear hyperplane that minimizes the discrepancy between actual and predicted values.

### 2. Regression Equation & Terminology
$$z_i = \beta_0 + \sum_{j=1}^p \beta_j X_{ij} + \varepsilon_i$$
- $z_i$: The transformed dependent variable ($\log(1 + \text{sales}_i)$).
- $\beta_0$: The intercept (baseline expected log sales when all predictors are zero).
- $\beta_j$: Feature regression coefficients (partial derivatives representing the change in expected log sales per unit increase in feature $j$, holding all other features constant).
- $X_{ij}$: Value of predictor $j$ for sample $i$ (numerical features standardized to mean 0, variance 1; categorical features one-hot encoded).
- $\varepsilon_i$: Residual error term assumed to be independently and identically distributed with mean zero.

### 3. Features & Target Transformation
- **Target Transformation:** $z = \log(1 + y)$. Linear models optimize Mean Squared Error in log space, which directly minimizes the Root Mean Squared Logarithmic Error (RMSLE) competition metric.
- **Physical Non-Negativity Mapping:** Linear models can mathematically produce negative outputs. To maintain physical reality (sales cannot be negative), predictions are mapped back via:
  $$\hat{y} = \max(\exp(\hat{z}) - 1, 0)$$

### 4. Regularization Paradigms (OLS, Ridge, Lasso, Elastic Net)
1. **Ordinary Least Squares (OLS):** Unregularized loss minimization. Prone to coefficient inflation and severe collinearity instability when rolling features are correlated.
2. **Ridge Regression ($L_2$ Regularization):** Adds penalty $\frac{\alpha}{2} \sum \beta_j^2$. Shrinks correlated coefficients smoothly toward zero without eliminating them. Best hyperparameter: $\alpha = 100.0$.
3. **Lasso Regression ($L_1$ Regularization):** Adds penalty $\alpha \sum |\beta_j|$. Forces coefficients of uninformative or redundant features strictly to zero, performing automatic feature selection. Best hyperparameter: $\alpha = 0.001$.
4. **Elastic Net ($L_1 + L_2$ Regularization):** Combines $L_1$ and $L_2$ penalties via mixing parameter $\ell_1$. Hyperparameters: $\alpha = 0.001, \ell_1 = 0.5$.

### 5. Evaluation Metrics Explained
- **RMSLE (Root Mean Squared Logarithmic Error):** Measures percentage errors symmetrically; prevents massive stores (Hypermarket Quito) from completely dominating small neighborhood stores.
- **RMSE (Root Mean Squared Error):** Heavily penalizes large unit outliers due to quadratic squaring.
- **MAE (Mean Absolute Error):** Average magnitude of absolute unit errors; directly interpretable by warehouse managers.
- **$R^2$ (Coefficient of Determination):** Proportion of target variance explained relative to the validation mean.

### 6. Phase 1 Documented Results ($N = 26,730$ validation observations)
| Model | RMSLE | RMSE | MAE | $R^2$ | Negative Preds (%) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DummyRegressor (Mean)** | 2.5245 | 1,317.35 | 458.25 | -0.1209 | 0.00% | Empirical Mean Baseline |
| **Seasonal Naive (Lag 16-22)** | 0.6388 | **356.20** | **92.17** | **0.9180** | 0.00% | Horizon-Consistent Benchmark |
| **Seasonal Naive (Lag 1-7)** | **0.6124** | 357.25 | 98.70 | 0.9176 | 0.00% | Immediate Persistence (1-day origin) |
| **Linear Regression (OLS)** | **0.6472** | 470.97 | 128.56 | 0.8567 | 1.65% | Unregularized Linear Pipeline |
| **Ridge Regression ($\alpha=100.0$)** | 0.6481 | 470.02 | 128.31 | 0.8573 | 1.55% | **Selected Best Linear Pipeline** |
| **Lasso Regression ($\alpha=0.001$)** | 0.6539 | **456.40** | **121.99** | **0.8655** | 0.86% | Sparse Feature-Selected Pipeline |
| **Elastic Net ($\alpha=0.001, \ell_1=0.5$)** | 0.6516 | 459.00 | 123.97 | 0.8639 | 1.07% | Balanced Regularization Pipeline |

### 7. Why Seasonal Naive Can Outperform Global Linear Models
Seasonal Naive achieves $R^2 = 0.9180$ and RMSE = 356.20 compared to Ridge ($R^2 = 0.8573$, RMSE = 470.02). This occurs because:
1. Ecuadorian grocery shopping displays intense day-of-week routine persistence (customers visit the same store on Saturdays and buy similar grocery baskets).
2. Global linear regression fits a pooled model across all 1,782 series simultaneously. While one-hot dummy variables capture categorical shifts, a single global hyperplane cannot fit micro-localized idiosyncratic store variances as perfectly as localized persistence for routine staple categories.

---

## Section C: Phase 2 — Logistic Regression & Demand Surge Classification

### 1. Why Classification is Formulated for Demand Surges
Continuous point predictions often fail to provide clear operational trigger points for inventory replenishment. Warehouse planners need a binary alerting mechanism to answer: *"Will this store-family experience an extreme demand spike tomorrow requiring emergency safety stock allocation?"*

### 2. Definition of Demand Surge & Zero-Leakage Threshold
- **Binary Surge Target:**
  $$Y_{s,f,t} = \begin{cases} 1 & \text{if } \text{sales}_{s,f,t} > \tau_{s,f} \\ 0 & \text{if } \text{sales}_{s,f,t} \le \tau_{s,f} \end{cases}$$
- **Threshold Cutoff $\tau_{s,f}$:** Derived as the historical **80th-percentile sales volume** for store $s$ and product family $f$.
- **Strict Leakage Prevention:** Thresholds are computed **strictly on historical training data** (`2016-08-01` to `2017-07-31`). No validation sales are ever referenced.
- **Sparse Series Fallback:** If $\tau_{s,f} \le 0$ (e.g., in rarely purchased categories like Books), the threshold falls back to the product family training 80th percentile; if still $\le 0$, it falls back to the global training 80th percentile with a physical floor of $1.0$ unit.

### 3. Logistic Function & Probability Threshold
The model predicts the probability of a demand surge using the standard sigmoid function:
$$P(Y = 1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_j X_j)}}$$
A binary decision is made by comparing $P(Y=1 \mid X)$ against a probability decision threshold $\tau_{\text{prob}}$:
$$\hat{Y} = \mathbb{I}(P(Y = 1 \mid X) \ge \tau_{\text{prob}})$$

### 4. Classification Metrics Explained
- **Accuracy:** Proportion of all correct predictions. Highly misleading in imbalanced retail data.
- **Precision:** $\frac{TP}{TP + FP}$. Measures the percentage of predicted surges that were actual surges (low precision = false alarms, overstocking costs).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$. Measures the percentage of actual demand surges caught (low recall = missed surges, retail stockouts).
- **F1-Score:** Harmonic mean of precision and recall: $2 \cdot \frac{P \cdot R}{P + R}$.
- **ROC-AUC:** Area under the ROC curve plotting True Positive Rate vs. False Positive Rate across all decision thresholds.
- **PR-AUC:** Area under the Precision-Recall curve. Represents the gold standard metric for imbalanced event detection.

### 5. Class Imbalance & The Deception of Accuracy
- **Class Breakdown:** Class 0 (Normal): 82.89% training / 83.30% validation; Class 1 (Surge): 17.11% training / 16.70% validation.
- **The Accuracy Trap:** The `DummyClassifier` achieves **83.30% Accuracy** simply by always predicting Class 0. However, its **Recall is exactly 0.00%**, missing all 4,463 demand surges on validation days (a 100% stockout failure rate).

### 6. Phase 2 Documented Results ($N = 26,730$, 4,463 true surges)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Decision Threshold | True Neg | False Pos | False Neg | True Pos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | **0.8330** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1670 | 0.5000 | 22,267 | 0 | 4,463 | 0 |
| **Logistic Regression (Default)** | 0.8212 | **0.4524** | 0.3363 | 0.3858 | 0.7518 | **0.4180** | 0.5000 | 20,450 | 1,817 | 2,962 | 1,501 |
| **Logistic Regression (Balanced)** | 0.6151 | 0.2705 | **0.7694** | 0.4003 | **0.7529** | 0.4159 | 0.5000 | 13,008 | 9,259 | 1,029 | **3,434** |
| **Logistic Regression (Tuned $\tau=0.29$)** | 0.7465 | 0.3416 | 0.5590 | **0.4241** | 0.7518 | **0.4180** | **0.2897** | 17,459 | 4,808 | 1,968 | 2,495 |

### 7. Strategic Supply Chain Interpretation
1. **Balanced Class Weighting:** Recommended when customer retention is paramount and stockout costs vastly exceed holding costs. It captures **76.94% of surges** (3,434 detected surges, cutting stockouts by >3x).
2. **Tuned Threshold ($\tau = 0.2897$):** Recommended when holding costs and product spoilage are high (e.g., fresh produce, dairy). Yields the highest F1-score (**0.4241**) and captures 55.90% of surges while generating roughly half the false alarms of balanced weighting (4,808 vs. 9,259 FP).

---

## Section D: Phase 3 — Econometric & Statistical Time Series Forecasting

### 1. What Time Series Data Is
A time series is a sequential collection of observations recorded at regular chronological intervals ($t_1, t_2, \dots, t_T$). Unlike cross-sectional tabular data, consecutive observations exhibit temporal autocorrelation, meaning past values carry predictive information about future values.

### 2. Time Series Decomposition (Trend, Seasonality, Residuals)
Classical additive decomposition represents retail series as:
$$y_t = T_t + S_t + I_t$$
- **Trend ($T_t$):** Long-term multi-month trajectory (e.g., annual growth in supermarket retail turnover).
- **Seasonality ($S_t$):** Repeating calendar cycles of fixed, known duration (in this dataset, strong weekly intra-week cycles where $s=7$).
- **Residuals / Irregular ($I_t$):** Stationary stochastic noise remaining after removing trend and seasonal patterns.

### 3. Statistical Forecasters Implemented
1. **Seasonal Naive:** Parameter-free benchmark predicting that the forecast at horizon $h$ equals the sales observed exactly one full seasonal cycle prior ($s=7$ days).
2. **Holt-Winters Exponential Smoothing (Triple Additive):** Continuously updates smoothed estimates of level ($\ell_t$), local trend slope ($b_t$), and weekly seasonal factors ($s_t$) via smoothing parameters $\alpha, \beta, \gamma \in (0, 1)$.
3. **SARIMA (Seasonal Autoregressive Integrated Moving Average):**
   $$\Phi_P(B^s) \phi_p(B) (1-B)^d (1-B^s)^D \log(1+y_t) = \Theta_Q(B^s) \theta_q(B) \varepsilon_t$$
   Parameterized as $(1, 1, 1) \times (1, 1, 0)_7$. First differencing $(d=1)$ stabilizes non-stationary drift, and seasonal differencing $(D=1, s=7)$ removes weekly cyclical dependence.
4. **SARIMAX (With Exogenous Drivers):** State-space regression combining SARIMA error dynamics with forward-known calendar and economic variables: scheduled promotional item counts (`onpromotion`), bimonthly paydays (`is_payday`), public holidays (`is_holiday`), and daily oil prices (`dcoilwtico`).

### 4. Meaning of SARIMA Parameters at a Conceptual Level
- $p=1$ (Non-seasonal AR order): Forecast depends directly on the immediate previous day's error-adjusted sales.
- $d=1$ (Non-seasonal differencing): Differences adjacent daily values ($y_t - y_{t-1}$) to eliminate linear trend and induce mean stationarity.
- $q=1$ (Non-seasonal MA order): Incorporates the shock/residual from the immediate previous day to smooth transitory shocks.
- $P=1, D=1, Q=0$ with $s=7$ (Seasonal components): Performs seasonal differencing ($y_t - y_{t-7}$) and models weekly autoregressive memory.

### 5. Multi-Step Forecast Horizon & Chronological Validation
- **Forecast Horizon ($H=15$ days):** Simulates the Kaggle competition operational test set (Aug 16–31, 2017).
- **State-Space Multi-Step Projection:** Future predictions for $t+1, \dots, t+15$ are generated recursively using the Kalman filter without observing ground truth future sales.

### 6. Phase 3 Documented Empirical Results

#### Part A: Enterprise Aggregated Daily Retail Sales (Macro Demand, $N=15$ days)
| Model | RMSLE | RMSE | MAE | $R^2$ | Fit Time (s) | Core Technical Insight |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seasonal Naive** | 0.1574 | 139,177.32 | 100,229.67 | -0.5537 | 0.010 | Weekly 7-day cyclical persistence baseline |
| **Holt-Winters** | **0.1295** | **115,406.36** | **82,557.60** | **-0.0683** | 0.633 | Dynamic additive level, trend, and weekly seasonality |
| **SARIMA (Univariate)** | 0.1478 | 130,393.45 | 91,265.18 | -0.3638 | 0.927 | Univariate $(1,1,1) \times (1,1,0)_7$ on $\log(1+y)$ target |
| **SARIMAX (With Exogenous)** | **0.1437** | **121,980.13** | **90,145.81** | **-0.1935** | 2.929 | State-Space model incorporating promotions, paydays, holidays, oil |

#### Part B: Representative Store-Family Cohort Benchmark ($N=15$ days)
| Series Name | Model | RMSLE | RMSE | MAE | $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Store 44 - GROCERY I | **Holt-Winters** | **0.1877** | **2,065.61** | **1,487.48** | **-0.4889** |
| Store 44 - GROCERY I | **SARIMAX** | 0.2708 | 2,116.27 | 1,920.24 | -0.5629 |
| Store 44 - BEVERAGES | **Holt-Winters** | **0.2285** | **2,566.73** | **2,063.24** | **-0.9713** |
| Store 44 - BEVERAGES | **SARIMAX** | 0.2737 | 3,151.11 | 2,405.38 | -1.9711 |
| Store 45 - PRODUCE | **Holt-Winters** | **0.1023** | **742.60** | **569.28** | **0.7165** |
| Store 45 - PRODUCE | **SARIMAX** | 0.1270 | 821.89 | 633.90 | 0.6527 |
| Store 47 - CLEANING | **Holt-Winters** | **0.2334** | **582.88** | 486.04 | **-0.6710** |
| Store 47 - CLEANING | **SARIMAX** | 0.2985 | 674.80 | 565.86 | -1.2397 |
| Store 1 - DELI | **SARIMAX** | **0.1937** | **20.86** | **17.94** | **0.6420** |
| Store 1 - DELI | **Holt-Winters** | 0.3235 | 36.44 | 24.62 | -0.0925 |

#### Cohort Mean Summary
| Model | Mean RMSLE | Mean RMSE | Mean MAE | Mean $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Holt-Winters** | **0.2151** | **1,198.85** | **926.13** | **-0.3014** |
| **SARIMAX** | 0.2327 | 1,356.99 | 1,108.67 | -0.4958 |
| **Seasonal Naive** | 0.2344 | 1,451.58 | 1,074.47 | -0.7458 |
| **SARIMA** | 0.2356 | 1,455.65 | 1,072.15 | -0.7670 |

---

## Section E: Temporal Integrity, Data Leakage & Evaluation Theory

### 1. What Data Leakage Means
Data leakage occurs when information from outside the training dataset (specifically future target information or concurrent operational variables) is inadvertently incorporated into the model training pipeline. Leakage produces unrealistically optimistic validation metrics that collapse upon real-world deployment.

### 2. Why Chronological Splitting Matters
In standard machine learning, datasets are shuffled randomly into $k$-folds. For temporal data, random splitting allows a model predicting sales on Tuesday to train on sales from Wednesday. In actual retail supply chain operations, future sales are strictly unknown. Chronological holdouts enforce the monotonic progression of time.

### 3. Why Future Sales Must Never Be Used as Prediction Features
If a feature uses sales from $t-1$ through $t-15$, it cannot be computed on day 16 of an operational forecast horizon without substituting intermediate predictions. Doing so accumulates recursive compounding error. Constraining all lag and rolling window features to $\ge 16$ days guarantees that all features are known at the forecast origin date.

### 4. Why Training Results and Validation Results Differ
- Training data covers a broad 1-year historical range with diverse seasonal events.
- Validation data covers a specific 15-day window in August 2017 with idiosyncratic promotional intensities and payday timings.
- In-sample training errors fit local noise; out-of-sample holdout errors evaluate true generalization.

### 5. Why Negative $R^2$ is Legitimate in Out-of-Sample Holdout Evaluation
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y}_{\text{val}})^2}$$
- On training data fit with an intercept, $R^2 \in [0, 1]$.
- On out-of-sample holdout data, if the model's Mean Squared Error is larger than the variance of the holdout period around its own holdout mean, the numerator exceeds the denominator, causing $R^2 < 0$.
- A negative $R^2$ simply indicates that a horizontal line representing the validation period's actual mean would have achieved lower mean squared error than the dynamic multi-step model predictions. It does **not** indicate code bugs or mathematical errors.

### 6. Why the Best Model Depends on Evaluation Metrics & Business Context
- If minimizing percentage error across all stores is the goal: **Ridge / Lasso (RMSLE 0.648 / 0.653)** or **Holt-Winters** are optimal.
- If minimizing gross warehouse volume deviation: **Seasonal Naive (RMSE 356.20)** is optimal.
- If maximizing stockout prevention: **Logistic Regression (Balanced, 76.94% Recall)** is optimal.
- If balancing stockouts against spoilage costs: **Logistic Regression (Tuned $\tau=0.29$, F1 0.4241)** is optimal.

---

## Section F: Architectural Code Walkthrough & Artifact Reference

### 1. Key Source Modules (`src/`)
- `config.py`: Central configuration declaring directory paths (`DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`), global random seeds (`RANDOM_SEED = 42`), cutoffs, and target names (`sales`, `is_high_sales`).
- `data_loader.py`: Ingestion module verifying table presence and casting columns to memory-efficient dtypes (`int8`, `int16`, `category`, `float32`).
- `preprocess.py`: Core feature engineering pipeline executing oil interpolation, holiday mapping, payday flagging, $\ge 16$-day lag generation, trailing 365-day baseline aggregation, and chronological partitioning.
- `evaluate.py`: Standardized metric computation functions for continuous regression (RMSLE, RMSE, MAE, $R^2$) and binary classification (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix).
- `linear_regression.py`: Model builders for Dummy, Seasonal Naive, OLS, Ridge, Lasso, and Elastic Net pipelines using `ColumnTransformer` (StandardScaler + OneHotEncoder).
- `logistic_regression.py`: Binary classification pipeline builders, cost-sensitive balanced weighting, and out-of-sample threshold calibration routines.
- `time_series.py`: Statistical forecaster wrappers for Seasonal Naive, Holt-Winters Exponential Smoothing, univariate SARIMA, and multivariate SARIMAX.
- `run_linear_regression.py`, `run_logistic_regression.py`, `run_time_series.py`, `run_eda.py`: Standalone execution runners that automate training, holdout evaluation, CSV table export, diagnostic plot generation, and artifact serialization.

### 2. Modular Jupyter Notebooks (`notebooks/`)
- `01_data_exploration_and_preprocessing.ipynb`: Interactive data inspection, store cluster boxplots, payday lift distributions, calendar engineering, 16-day lag generation, and parquet export.
- `02_linear_regression.ipynb`: Continuous sales regression comparing OLS, Ridge, Lasso, Elastic Net, and DecisionTreeRegressor with residual diagnostics and model serialization.
- `03_logistic_regression.ipynb`: Surge classification analysis, precision-recall curve inspection, threshold tuning ($\tau=0.29$), and confusion matrix diagnostics.
- `04_time_series_forecasting.ipynb`: Multi-algorithm forecasting (Naive, MA, Holt-Winters, SARIMA, SARIMAX) across enterprise and cohort levels.
- `05_final_model_comparison.ipynb`: Master cross-task synthesis unifying continuous regression, surge classification, and econometric forecasters with operational deployment roadmaps.

### 3. Serialized Model Artifacts (`models/`)
- `best_linear_model.joblib` (~6 KB): Serialized `scikit-learn` Ridge pipeline ($\alpha=100.0$) including transformer pipelines and fitted weights.
- `best_logistic_model.joblib` (~40 KB): Serialized Logistic Regression pipeline and calibrated surge threshold dictionary.
- `best_time_series_model.joblib` (~18 MB): Serialized SARIMAX state-space model artifact and forecast parameters.

---

## Section G: 45 Comprehensive Viva Questions & Model Answers

### Category 1: Project Fundamentals & Supply Chain Context (Q1 – Q8)

#### Q1: What is the primary objective of your retail sales prediction project?
**Answer:** The objective is to design, implement, and benchmark an end-to-end predictive analytics framework for grocery demand. It tackles three distinct tasks: continuous sales volume estimation using regularized linear regression, demand surge event detection using logistic regression, and multi-step sales forecasting using statistical econometric models (Holt-Winters, SARIMAX).

#### Q2: What dataset did you use and what is its operational scope?
**Answer:** We used the Kaggle Store Sales — Time Series Forecasting dataset from Corporación Favorita, a major Ecuadorian supermarket chain. It covers 54 retail stores across 17 cities and 33 product families, creating a high-dimensional panel of 1,782 distinct daily time series spanning January 2013 to August 2017.

#### Q3: Why is demand forecasting critical in the retail grocery sector?
**Answer:** Grocery supply chains face thin operating margins, high SKU diversity, and perishable shelf-life constraints. Accurate forecasting minimizes stockouts (lost revenue and dissatisfied customers) while simultaneously preventing overstocking (excess working capital tie-up and perishable goods spoilage).

#### Q4: What are the main external drivers of retail demand identified in your EDA?
**Answer:** Our exploratory analysis identified four major drivers:
1. Intra-week seasonality (+39.35% weekend demand lift).
2. Bimonthly payday cycles on the 15th and month-end (+1.41% average lift).
3. Public and transferred holidays.
4. Macroeconomic fluctuations reflected in Brent crude oil prices, which dictate purchasing power in Ecuador's dollarized economy.

#### Q5: What is the 2016 Ecuador earthquake shock documented in your EDA?
**Answer:** On April 16, 2016, a magnitude 7.8 earthquake struck Manabí and Esmeraldas provinces. Our EDA shows a dramatic spike in immediate demand for staple goods (Food, Water, Cleaning supplies) driven by emergency relief operations, followed by prolonged regional demand shifts.

#### Q6: Why did you exclude transactions.csv from your predictive features?
**Answer:** Customer transactions occur concurrently with sales on day $t$. At the forecast origin (day $t$), future transactions for $t+1$ to $t+16$ are unknown. Using same-day transactions would constitute catastrophic target leakage during operational deployment.

#### Q7: How did you handle missing values in the daily oil price series?
**Answer:** Daily oil prices had 43 missing weekend and market holiday values. We applied forward-fill (carrying Friday's closing price through the weekend) followed by backward-fill, ensuring a continuous daily economic signal without looking forward into future prices.

#### Q8: What product category accounts for the largest sales volume?
**Answer:** `GROCERY I` is the largest category, accounting for 343,462,720 units (32.0% of network sales). The top five categories (`GROCERY I`, `BEVERAGES`, `PRODUCE`, `CLEANING`, `DAIRY`) collectively comprise over 75% of total retail turnover.

---

### Category 2: Linear Regression & Regularization (Q9 – Q16)

#### Q9: What is Ordinary Least Squares (OLS) regression?
**Answer:** OLS is an unregularized linear regression method that estimates unknown parameters $\beta$ by minimizing the sum of squared differences between observed values and linear predictions ($\sum (y_i - \hat{y}_i)^2$).

#### Q10: Why did you transform the sales target using $\log(1 + \text{sales})$?
**Answer:** Sales volumes exhibit severe positive right-skewness and variance that scales with volume. Transforming the target via $z = \log(1 + y)$ normalizes the residuals, stabilizes variance, and ensures that optimizing Mean Squared Error in log space directly minimizes the competition Root Mean Squared Logarithmic Error (RMSLE).

#### Q11: How do you guarantee that your linear model never predicts negative sales?
**Answer:** Linear models can mathematically output negative values. We enforce physical non-negativity by applying an explicit non-linear back-transformation: $\hat{y} = \max(\exp(\hat{z}) - 1, 0)$.

#### Q12: What is Ridge Regression and how does it prevent overfitting?
**Answer:** Ridge regression adds an $L_2$ penalty proportional to the sum of squared coefficients ($\frac{\alpha}{2} \sum \beta_j^2$) to the OLS loss function. This shrinks collinear feature weights smoothly toward zero, reducing model variance and improving generalization on correlated features.

#### Q13: What is Lasso Regression and what unique property does it have?
**Answer:** Lasso regression adds an $L_1$ penalty proportional to the absolute value of coefficients ($\alpha \sum |\beta_j|$). Because the $L_1$ ball has sharp corners on the coordinate axes, Lasso forces uninformative or redundant feature weights strictly to zero, performing automated feature selection.

#### Q14: What is Elastic Net regression?
**Answer:** Elastic Net combines both $L_1$ (Lasso) and $L_2$ (Ridge) penalties via a mixing parameter $\ell_1$. It is especially useful when multiple features are highly correlated; Lasso typically picks only one at random, whereas Elastic Net retains the group structure while inducing sparsity.

#### Q15: What were the actual performance metrics of your best linear model?
**Answer:** On the 26,730 validation holdout observations, our selected **Ridge Regression ($\alpha=100.0$)** achieved an **RMSLE of 0.6481**, **RMSE of 470.02**, **MAE of 128.31**, and **$R^2$ of 0.8573**. Lasso achieved the lowest RMSE of 456.40 and MAE of 121.99 ($R^2 = 0.8655$).

#### Q16: Why did the addition of trailing 365-day rolling baselines dramatically improve your linear model?
**Answer:** Static target encodings create a severe train-validation distribution mismatch. By introducing dynamic 365-day rolling mean log-sales (`hist_sf_mean_log`) and 52-week day-of-week baselines (`hist_sf_dow_mean_log`) shifted by 16 days, the models gained access to store-family scale and seasonal profiles with zero leakage, elevating $R^2$ from 0.0113 to 0.8573.

---

### Category 3: Logistic Regression & Surge Classification (Q17 – Q24)

#### Q17: Why formulate demand surge prediction as a classification problem?
**Answer:** Continuous point forecasts often fail to provide clear operational thresholds for inventory managers. Framing surge detection as a binary classification problem provides explicit probabilistic triggers for allocating emergency safety stock and preventing stockouts.

#### Q18: How is a demand surge defined in your project?
**Answer:** A demand surge is defined as a binary indicator where observed daily sales for a specific store and product family exceed its historical 80th-percentile sales cutoff ($\text{sales}_{s,f,t} > \tau_{s,f}$).

#### Q19: How did you ensure zero data leakage when computing surge thresholds?
**Answer:** All 80th-percentile thresholds $\tau_{s,f}$ were computed **strictly on historical training data** (`2016-08-01` to `2017-07-31`). No validation or test records were ever accessed during threshold derivation.

#### Q20: What is the sigmoid function used in Logistic Regression?
**Answer:** The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps any real-valued linear score $z \in (-\infty, \infty)$ into a bounded probability $P \in (0, 1)$, representing the likelihood that an instance belongs to the positive surge class.

#### Q21: What is the class imbalance ratio in your dataset?
**Answer:** Class 0 (Normal Demand) represents **82.89%** of training records (537,640 rows) and **83.30%** of validation records (22,267 rows). Class 1 (Demand Surge) represents **17.11%** of training records (111,008 rows) and **16.70%** of validation records (4,463 rows).

#### Q22: Why is Accuracy a misleading metric in demand surge classification?
**Answer:** The `DummyClassifier` achieves **83.30% Accuracy** by simply predicting the majority class (Normal Demand) for every observation. However, its Recall is **0.00%**, missing all 4,463 surges. In retail operations, this represents a 100% stockout failure rate on surge days.

#### Q23: What is the operational difference between Default and Balanced Logistic Regression?
**Answer:** 
- *Default Logistic Regression ($\tau=0.5$):* Highly conservative; achieves 45.24% Precision and low false alarms (1,817 FP), but misses 2,962 surges (33.63% Recall).
- *Balanced Logistic Regression:* Weights minority surge instances inversely proportional to class frequencies, achieving **76.94% Recall** (capturing 3,434 surges) to minimize stockouts, at the cost of 9,259 false alarms.

#### Q24: How did you calibrate the decision threshold $\tau = 0.2897$?
**Answer:** We performed an out-of-sample temporal calibration within the training data. A model trained on Aug 2016–May 2017 (539,946 rows) was evaluated on a held-out June–July 2017 split (108,702 rows) to identify the threshold maximizing F1-score ($\tau = 0.2897$). This tuned threshold achieved the highest validation **F1-Score of 0.4241** and captured 55.90% of surges with only 4,808 false alarms.

---

### Category 4: Econometric & Time Series Forecasting (Q25 – Q32)

#### Q25: How does time series forecasting fundamentally differ from regression?
**Answer:** Tabular regression pools records into a shared feature space and treats observations as independent conditional on features. Time series forecasting explicitly models sequential autocorrelation, autoregressive lags, moving average shocks, and deterministic or stochastic seasonality along the temporal dimension.

#### Q26: What is the Seasonal Naive forecasting method?
**Answer:** Seasonal Naive is a persistent benchmark that forecasts future demand at horizon $h$ to be exactly equal to the demand observed in the same seasonal period from the prior cycle: $\hat{y}_{T+h} = y_{T+h - 7 \cdot k}$.

#### Q27: How does Holt-Winters Exponential Smoothing work?
**Answer:** Triple additive Holt-Winters dynamically updates three recursive components: level ($\ell_t$), local trend slope ($b_t$), and weekly seasonal indices ($s_t$) using smoothing parameters $\alpha, \beta, \gamma \in (0, 1)$. The $h$-step forecast is calculated as $\hat{y}_{T+h} = \ell_T + h b_T + s_{T+h-m(k+1)}$.

#### Q28: What do the terms SARIMA and SARIMAX stand for?
**Answer:** 
- *SARIMA:* Seasonal Autoregressive Integrated Moving Average.
- *SARIMAX:* SARIMA with eXogenous regressors (incorporating external explanatory covariates such as promotions, paydays, holidays, and oil prices).

#### Q29: What SARIMA order was implemented and what do the parameters mean?
**Answer:** We implemented $(1, 1, 1) \times (1, 1, 0)_7$ on log-transformed sales:
- $p=1, P=1$: Non-seasonal and seasonal autoregressive terms modeling daily and weekly memory.
- $d=1, D=1$: Non-seasonal differencing (removes trend) and seasonal differencing (removes weekly intra-week seasonality).
- $q=1, Q=0$: Non-seasonal moving average capturing one-period lagged forecast errors.

#### Q30: What were the results of SARIMAX on enterprise aggregated daily retail sales?
**Answer:** On national enterprise daily sales ($N=15$ validation days), SARIMAX achieved an **RMSLE of 0.1437**, **RMSE of 121,980.13**, and **MAE of 90,145.81**, outperforming the Seasonal Naive baseline (RMSE 139,177.32) by reducing error by **12.4%** (-17,197 units) due to exogenous promotional and payday signals.

#### Q31: Which model performed best on the representative store-family cohort?
**Answer:** **Holt-Winters Exponential Smoothing** achieved the lowest overall cohort mean error with a **Mean RMSLE of 0.2151**, **Mean RMSE of 1,198.85**, and **Mean MAE of 926.13**, demonstrating superior capability in dynamically tracking local baseline trends across diverse retail categories.

#### Q32: Why did you evaluate a 5-series representative cohort rather than fitting 1,782 SARIMAX models?
**Answer:** Fitting 1,782 individual SARIMAX state-space models via numerical maximum likelihood takes several hours of computational runtime and frequently suffers from numerical non-convergence on sparse, intermittent zero-sales series (e.g., Books). The 5-series cohort captures the spectrum of hypermarket staples, high-promotion beverages, fresh perishables, stable cleaning goods, and neighborhood delis.

---

### Category 5: Validation Strategy, Data Leakage & Evaluation Metrics (Q33 – Q39)

#### Q33: What is the 16-day operational constraint in your feature engineering?
**Answer:** The Kaggle competition test set spans 16 days (August 16–31, 2017). In actual supermarket deployment, forecasts for the next 16 days must be generated simultaneously without intermediate sales data. Constraining all lag and rolling features to $\ge 16$ days ensures zero data leakage across the entire operational forecast horizon.

#### Q34: What is the mathematical definition of RMSLE and why is it preferred over RMSE?
**Answer:** 
$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(\ln(1 + \hat{y}_i) - \ln(1 + y_i)\right)^2}$$
RMSLE measures relative percentage error. RMSE heavily penalizes errors on high-volume items (e.g., error of 100 on Grocery I is treated the same as error of 100 on Books, even though Books sells only 5 units). RMSLE penalizes proportional errors equally across all product scales.

#### Q35: Why did your models exhibit a negative $R^2$ on certain out-of-sample holdout sets?
**Answer:** $R^2$ is defined as $1 - \frac{\text{MSE}}{\text{Var}(y_{\text{val}})}$. On out-of-sample holdout evaluation, $R^2$ is not constrained to $[0, 1]$. If a model's mean squared prediction error exceeds the sample variance of the holdout period around its own 15-day mean, $R^2$ becomes negative. This is common in short validation windows exhibiting localized level shifts.

#### Q36: Why does Seasonal Naive achieve higher $R^2$ (0.918) than Ridge Regression (0.857)?
**Answer:** Seasonal Naive uses the exact localized store-family sales from the prior cycle. In Ecuador, consumer grocery shopping is heavily habitual by day-of-week. Ridge regression fits a single pooled linear hyperplane across all 1,782 series simultaneously, introducing minor cross-series smoothing errors.

#### Q37: What is the difference between ROC-AUC and PR-AUC?
**Answer:** 
- *ROC-AUC:* Plots True Positive Rate vs. False Positive Rate across all thresholds. Can present an overly optimistic picture when the negative class is dominant because False Positive Rate has a very large denominator ($TN + FP$).
- *PR-AUC:* Plots Precision vs. Recall. Evaluates true surge detection fidelity without being inflated by true negatives, making it the superior metric for imbalanced problems.

#### Q38: What were the ROC-AUC and PR-AUC scores of your Logistic Regression models?
**Answer:** Both Logistic Regression models achieved a **ROC-AUC of ~0.752** and a **PR-AUC of ~0.418**, outperforming the random/no-skill baseline PR-AUC of **0.1670** by over **+150%**.

#### Q39: What is the difference between horizon-consistent Seasonal Naive and 1-day origin Seasonal Naive?
**Answer:** 
- *Horizon-Consistent (Lag 16-22):* Uses the last completed week prior to the 16-day cutoff, respecting the exact same information boundary as the machine learning models ($R^2 = 0.9180$, RMSE = 356.20).
- *1-Day Origin (Lag 1-7):* Uses the immediate 7 days prior to validation ($R^2 = 0.9176$, RMSLE = 0.6124).

---

### Category 6: Implementation, Codebase Architecture & Viva Defense (Q40 – Q45)

#### Q40: How is your project structured for modularity and maintainability?
**Answer:** The repository separates concerns into dedicated directories:
- `src/`: Modular Python engines for data loading, preprocessing, regression, classification, time-series, and metric evaluation.
- `notebooks/`: Numbered Jupyter notebooks documenting each step from EDA to final evaluation.
- `reports/`: Quantitative benchmark CSVs and markdown analysis reports.
- `reports/figures/`: High-resolution diagnostic plots.
- `models/`: Serialized production artifacts (`.joblib`).

#### Q41: Which model artifacts are serialized in your models/ directory?
**Answer:** We serialize three production artifacts:
1. `best_linear_model.joblib` (~6 KB): Fitted `scikit-learn` Ridge pipeline ($\alpha=100.0$) with preprocessors.
2. `best_logistic_model.joblib` (~40 KB): Fitted Logistic Regression pipeline and calibrated surge threshold dictionary.
3. `best_time_series_model.joblib` (~18 MB): Serialized SARIMAX state-space model artifact.

#### Q42: How does your preprocessing pipeline handle categorical variables?
**Answer:** Categorical variables (`family`, `type`, `cluster`, `dayofweek`) are preprocessed using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` within a `ColumnTransformer`, while numeric features are imputed with median values and scaled using `StandardScaler()`.

#### Q43: What are the primary limitations of your current linear regression implementation?
**Answer:** Linear models assume additive linear relationships between features and log-sales. They cannot naturally capture complex higher-order interactions (such as store cluster $\times$ promotional discount depth) without manual interaction terms. Future work should incorporate tree-based gradient boosting models (LightGBM/XGBoost).

#### Q44: What are the key supply chain recommendations you would present to retail executives?
**Answer:** 
1. Use **Holt-Winters Exponential Smoothing** or **SARIMAX** for macro-level warehouse replenishment and cash-flow forecasting.
2. Deploy **Balanced Logistic Regression** for center-store high-margin items to slash stockout risk by >3x.
3. Deploy **Tuned-Threshold Logistic Regression ($\tau=0.29$)** for perishable fresh food (Produce, Dairy) to balance stockout alerts while avoiding severe spoilage costs.

#### Q45: Did you retrain any models or alter experimental results during project preparation?
**Answer:** No. In strict accordance with academic integrity and project guidelines, all models, evaluation metrics, diagnostic figures, and benchmark tables were preserved exactly as previously computed and documented in existing reports; no models were retrained.
