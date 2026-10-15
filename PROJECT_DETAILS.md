# Retail Sales Prediction and Forecasting — Technical Architecture & Project Details

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Store%20Sales-20BEFF.svg)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## 1. Project Overview & Scope

This project implements an end-to-end Machine Learning, Statistical Econometric, and Predictive Analytics system on the **Kaggle Store Sales — Time Series Forecasting** dataset (Corporación Favorita, Ecuador). The repository addresses both continuous sales volume estimation (regression and time-series forecasting) and operational stockout risk mitigation (binary demand surge classification).

The technical architecture is designed around three fundamental engineering principles:
1. **Strict Temporal Integrity (Zero Target Leakage):** Enforcing a $\ge 16$-day lag horizon on all auto-regressive and rolling window features, mirroring real-world multi-step supply chain deployment and the Kaggle 16-day operational test set.
2. **Multi-Paradigm Modeling:** Comparing regularized cross-sectional linear machine learning pipelines against statistical time-series methods (Holt-Winters, SARIMA, SARIMAX) and binary classification models (Logistic Regression with threshold tuning).
3. **Physical & Operational Validity:** Guaranteeing non-negative sales volume predictions via log-space target optimization and explicit back-transformation clipping, while evaluating classification through cost-sensitive trade-offs (holding costs vs. stockout penalties).

---

## 2. Dataset Architecture & External Signals

### 2.1 Raw Data Sources
The primary dataset originates from Corporación Favorita, one of Ecuador's largest supermarket chains, covering operations from January 1, 2013 to August 15, 2017:

| Dataset File | Records | Attributes | Primary Key / Keys | Description & Role |
| :--- | :--- | :--- | :--- | :--- |
| **`train.csv`** | 3,000,888 | 6 | `id` (date, store_nbr, family) | Daily unit sales per store and product family. Spans 1,684 unique dates. Target variable is `sales`. |
| **`test.csv`** | 28,512 | 5 | `id` (date, store_nbr, family) | Operational forecast horizon spanning August 16 to August 31, 2017 (16 days $\times 1,782$ series). |
| **`stores.csv`** | 54 | 5 | `store_nbr` | Store metadata covering 17 cities, 16 provinces, 5 store types (A through E), and 17 operational clusters. |
| **`oil.csv`** | 1,218 | 2 | `date` | Daily Brent crude oil index (`dcoilwtico`). Acts as a macroeconomic proxy for Ecuadorian consumer purchasing power. |
| **`holidays_events.csv`** | 350 | 6 | `date`, `locale_name` | National, regional, and municipal holidays and events, including transferred holidays and the 2016 earthquake. |
| **`transactions.csv`** | 83,488 | 3 | `date`, `store_nbr` | Daily customer transaction counts. Excluded from predictive features to prevent target leakage during multi-step forecasting. |

### 2.2 Network & Product Hierarchy
- **Store Network:** 54 stores categorized into 5 types (`A`: 9 flagship hypermarkets; `B`: 8 medium supermarkets; `C`: 15 neighborhood markets; `D`: 18 standard supermarkets; `E`: 4 small format stores).
- **Product Hierarchy:** 33 distinct product families ranging from high-frequency groceries (`GROCERY I`, `BEVERAGES`, `PRODUCE`, `CLEANING`, `DAIRY`) representing >75% of sales, to specialty low-volume categories (`BOOKS`, `BABY CARE`).
- **Total Time Series:** $54 \times 33 = 1,782$ distinct store-family time series.

---

## 3. Preprocessing & Feature Engineering Pipeline

### 3.1 Data Cleaning & Imputation (`src/preprocess.py`)
1. **Oil Price Imputation:** Daily oil prices contain 43 missing weekend/holiday values. Missing records are imputed using forward-fill followed by backward-fill, ensuring a continuous daily economic signal.
2. **Holiday Harmonization:** National holidays take precedence over regional and local events. Transferred holidays (`transferred == True`) are deactivated on their calendar date and activated on their celebrated date.
3. **Data Type Optimization:** To optimize RAM usage across 3 million records, data types are cast to minimal memory footprints:
   - `store_nbr`: `int8`
   - `cluster`: `int8`
   - `onpromotion`: `int16`
   - `sales`: `float32`
   - `type`, `family`, `city`, `state`: `category`

### 3.2 Calendar & Payday Feature Engineering
- **Intra-Week Seasonality:** `dayofweek` (0–6), `is_weekend` (Saturday/Sunday). In Ecuador, grocery sales exhibit a +39.35% weekend demand lift.
- **Bimonthly Paydays:** Public and private sector employees in Ecuador receive salary payments twice a month: on the 15th and the final calendar day of the month.
  $$\text{is\_payday} = \begin{cases} 1 & \text{if } \text{day} = 15 \text{ or } \text{day} = \text{days\_in\_month} \\ 0 & \text{otherwise} \end{cases}$$
- **Promotional Signals:**
  - `onpromotion`: Count of active promotions for the store-family pair.
  - `has_promotion`: Binary indicator ($\mathbb{I}(\text{onpromotion} > 0)$).
  - `log_promotion`: Logarithmic transform $\log(1 + \text{onpromotion})$ to compress promo spikes.

### 3.3 Zero-Leakage Lag & Rolling Window Formulations
Operational deployment requires forecasting 16 days ahead without intermediate actual sales. Any feature using $t-1$ through $t-15$ leaks future sales. All lag and rolling window features are strictly constrained to $\ge 16$ days:

$$\text{sales\_lag\_k}(t) = y(t - k), \quad \text{for } k \in \{16, 21, 28, 35\}$$

$$\text{sales\_rolling\_mean\_W}(t) = \frac{1}{W} \sum_{i=16}^{15+W} y(t - i), \quad \text{for } W \in \{7, 14, 30, 60\}$$

$$\text{sales\_rolling\_std\_30}(t) = \sqrt{\frac{1}{29} \sum_{i=16}^{45} \left(y(t - i) - \mu_{\text{rolling}}\right)^2}$$

### 3.4 Dynamic Trailing 365-Day Historical Rolling Baselines
To eliminate the massive train-validation distribution shift present in static target encodings, dynamic rolling baselines are computed:
- `hist_sf_mean_log`: 365-day trailing average log sales for each store-family pair, shifted by 16 days.
- `hist_sf_dow_mean_log`: 52-week trailing average log sales for each store-family pair on that specific day of the week, shifted by 16 days.
Incorporating these two features elevated linear model validation $R^2$ from **0.0113** to **0.8573** (Ridge) and **0.8655** (Lasso) while eliminating target leakage.

### 3.5 Mathematical Target Transformation & Non-Negativity
Sales volume exhibits severe right-skewness and heteroscedastic variance. Models are trained on log-transformed targets:
$$z = \log(1 + y)$$
Predicted log-sales $\hat{z}$ are converted back to real physical sales units via:
$$\hat{y} = \max\left(\exp(\hat{z}) - 1, 0\right)$$
This mathematical formulation guarantees:
1. Optimization under Mean Squared Error on $z$ directly minimizes competition **RMSLE** on $y$.
2. Physical impossibility of negative sales predictions ($\hat{y} \ge 0$).

---

## 4. Modeling Methodologies & Algorithmic Formulations

### 4.1 Phase 1: Linear Regression Framework (`src/linear_regression.py`)
All models are evaluated on the 1-year rolling temporal window (`2016-08-01` to `2017-07-31`: 648,648 rows) and tested on the 15-day holdout (`2017-08-01` to `2017-08-15`: 26,730 rows).

1. **Ordinary Least Squares (OLS):**
   $$\min_{\beta} \frac{1}{2N} \sum_{i=1}^N \left(z_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij}\right)^2$$
   Unregularized baseline fitting linear weights via normal equations / QR decomposition.

2. **Ridge Regression ($L_2$ Regularization):**
   $$\min_{\beta} \frac{1}{2N} \sum_{i=1}^N \left(z_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij}\right)^2 + \frac{\alpha}{2} \sum_{j=1}^p \beta_j^2$$
   Shrinks collinear feature coefficients (e.g., between `sales_rolling_mean_7` and `sales_rolling_mean_14`), preventing overfitting. Optimal hyperparameter: $\alpha = 100.0$.

3. **Lasso Regression ($L_1$ Regularization):**
   $$\min_{\beta} \frac{1}{2N} \sum_{i=1}^N \left(z_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij}\right)^2 + \alpha \sum_{j=1}^p |\beta_j|$$
   Forces redundant feature weights exactly to zero, performing automated feature selection. Optimal hyperparameter: $\alpha = 0.001$.

4. **Elastic Net ($L_1 + L_2$ Regularization):**
   $$\min_{\beta} \frac{1}{2N} \sum_{i=1}^N \left(z_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij}\right)^2 + \alpha \cdot \ell_1 \sum_{j=1}^p |\beta_j| + \frac{\alpha(1 - \ell_1)}{2} \sum_{j=1}^p \beta_j^2$$
   Combines the sparsity of Lasso with the grouping effect of Ridge. Hyperparameters: $\alpha = 0.001, \ell_1 = 0.5$.

5. **Seasonal Naive Benchmark (Horizon-Consistent):**
   $$\hat{y}_{s,f,t} = y_{s,f, t - 16 - (t \bmod 7)}$$
   Persistence baseline looking back to the exact same day of the week from the last completed week prior to the 16-day forecast horizon cutoff.

---

### 4.2 Phase 2: Logistic Regression Demand Surge Classification (`src/logistic_regression.py`)
Predicts whether a store-family combination will experience a demand surge on observation date $t$.

#### Target Definition & Leak-Free Thresholds
$$\text{Surge Target } Y_{s,f,t} = \begin{cases} 1 & \text{if } y_{s,f,t} > \tau_{s,f} \\ 0 & \text{if } y_{s,f,t} \le \tau_{s,f} \end{cases}$$
- **Threshold Cutoff $\tau_{s,f}$:** Derived **strictly on historical training data** (`2016-08-01` to `2017-07-31`) as the 80th-percentile sales volume for store $s$ and family $f$.
- **Sparse Series Fallback:** If $\tau_{s,f} \le 0$, the threshold falls back to the product family 80th percentile across all stores; if still $\le 0$, it falls back to the global training 80th percentile with a hard physical minimum of 1.0 unit.

#### Mathematical Model & Probability Calibration
$$P(Y_i = 1 \mid X_i) = \sigma(\beta^T X_i) = \frac{1}{1 + e^{-(\beta_0 + \sum_{j=1}^p \beta_j X_{ij})}}$$
Binary prediction with decision threshold $\tau_{\text{prob}}$:
$$\hat{Y}_i = \begin{cases} 1 & \text{if } P(Y_i = 1 \mid X_i) \ge \tau_{\text{prob}} \\ 0 & \text{if } P(Y_i = 1 \mid X_i) < \tau_{\text{prob}} \end{cases}$$

#### Evaluated Configurations
1. **DummyClassifier:** Most frequent majority class baseline ($\hat{Y} = 0$).
2. **Default Logistic Regression:** Unweighted loss ($C=1.0$, $\tau_{\text{prob}} = 0.50$).
3. **Balanced Logistic Regression:** Cost-sensitive inverse class frequency loss:
   $$w_c = \frac{N}{2 \cdot N_c}$$
   Weights minority surge instances higher ($w_1 \approx 2.92, w_0 \approx 0.60$), aggressively penalizing missed surges.
4. **Tuned Threshold Logistic Regression:** Optimal threshold $\tau_{\text{prob}} = 0.2897$ calibrated on an out-of-sample temporal split within training data (trained on Aug 2016–May 2017: 539,946 rows; calibrated on June–July 2017: 108,702 rows) to maximize F1-score without leakage into the validation holdout.

---

### 4.3 Phase 3: Time Series Econometric Forecasting (`src/time_series.py`)
Direct multi-step out-of-sample recursive and state-space sales forecasting without cross-sectional feature pooling.

#### Forecasting Scope & Hierarchical Design
1. **Enterprise Aggregated Demand (Macro Demand):**
   $$Y_t = \sum_{s=1}^{54} \sum_{f=1}^{33} y_{s,f,t}$$
   National daily sales volume ($N=365$ training days, $15$ holdout validation days).
2. **Representative Store-Family Cohort (Micro Demand):**
   - Store 44 - GROCERY I (High-volume staple hypermarket)
   - Store 44 - BEVERAGES (High promotional volatility)
   - Store 45 - PRODUCE (High-turnover perishable fresh food)
   - Store 47 - CLEANING (Stable non-perishable essentials)
   - Store 1 - DELI (Low-volume neighborhood downtown supermarket)
3. **Cross-Phase Global Anchor:** Full matrix ($1,782$ series, $N=26,730$ validation observations) evaluated under Seasonal Naive.

#### Statistical Forecasters Implemented
1. **Seasonal Naive Forecaster:**
   $$\hat{y}_{T+h} = y_{T+h - 7 \cdot k}$$
   Cyclical 7-day persistence baseline.

2. **Holt-Winters Exponential Smoothing (Triple Additive):**
   $$\begin{aligned}
   \ell_t &= \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1}) \\
   b_t &= \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1} \\
   s_t &= \gamma (y_t - \ell_t) + (1 - \gamma) s_{t-m} \\
   \hat{y}_{T+h} &= \ell_T + h b_T + s_{T+h-m(k+1)}
   \end{aligned}$$
   Dynamically estimates smoothed level ($\ell$), local trend ($b$), and weekly seasonal factors ($s$) with seasonal period $m=7$.

3. **SARIMA (Univariate):**
   $$\Phi_P(B^7) \phi_p(B) (1 - B)^d (1 - B^7)^D \log(1 + y_t) = \Theta_Q(B^7) \theta_q(B) \varepsilon_t$$
   Parameterized as $(1, 1, 1) \times (1, 1, 0)_7$. First differencing $(d=1)$ achieves stationarity, and seasonal differencing $(D=1, s=7)$ removes intra-week cyclical autocorrelation.

4. **SARIMAX (With Exogenous Drivers):**
   $$\begin{aligned}
   z_t &= X_t \beta + u_t \\
   \Phi_P(B^7) \phi_p(B) (1 - B)^d (1 - B^7)^D u_t &= \Theta_Q(B^7) \theta_q(B) \varepsilon_t
   \end{aligned}$$
   Combines seasonal autoregressive moving-average error structures with forward-known calendar and promotional signals ($X_t$ includes `onpromotion`, `is_payday`, `is_holiday`, and `dcoilwtico`).

---

## 5. Validation Strategy & Evaluation Methodology

### 5.1 Strict Chronological Splitting vs. Random K-Fold
In retail time-series forecasting, standard random $k$-fold cross-validation or random train-test splitting introduces catastrophic future-to-past data leakage:
- Random splitting allows a model to train on sales from day $t+1$ and predict sales for day $t$.
- In reality, time moves forward monotonically; models must forecast into an unseen future origin.
- **Protocol:**
  - **Training Window:** `2016-08-01` to `2017-07-31` (365 days; 648,648 rows across 1,782 series).
  - **Validation Holdout:** `2017-08-01` to `2017-08-15` (15 days; 26,730 rows across 1,782 series).
  - **Kaggle Operational Test Horizon:** `2017-08-16` to `2017-08-31` (16 days; 28,512 rows across 1,782 series).

### 5.2 Mathematical Formulation of Evaluation Metrics

#### Continuous Regression & Time Series Metrics
1. **Root Mean Squared Logarithmic Error (RMSLE):**
   $$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(\ln(1 + \hat{y}_i) - \ln(1 + y_i)\right)^2}$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
3. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
4. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y}_{\text{val}})^2}$$

#### Binary Classification Metrics
1. **Confusion Matrix:** True Negatives ($TN$), False Positives ($FP$), False Negatives ($FN$), True Positives ($TP$).
2. **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$.
3. **Precision:** $\frac{TP}{TP + FP}$ (Fraction of surge predictions that were actual surges).
4. **Recall:** $\frac{TP}{TP + FN}$ (Fraction of actual surges caught).
5. **F1-Score:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
6. **ROC-AUC & PR-AUC:** Area under ROC and Precision-Recall curves.

---

## 6. Comprehensive Empirical Results

All experimental benchmarks were executed on the identical 15-day validation holdout (`2017-08-01` to `2017-08-15`).

### 6.1 Phase 1 Linear Regression Benchmark
*Full dataset matrix: 54 stores $\times$ 33 families ($N = 26,730$ validation observations):*

| Model | RMSLE | RMSE | MAE | $R^2$ | Fit Time (s) | Negative Preds (%) | Pred Min | Pred Max | Constraint / Model Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DummyRegressor (Mean)** | 2.5245 | 1,317.35 | 458.25 | -0.1209 | 0.14 | 0.00% | 32.52 | 32.52 | Empirical Mean Baseline |
| **Seasonal Naive (Lag 16-22)** | 0.6388 | **356.20** | **92.17** | **0.9180** | 0.39 | 0.00% | 0.00 | 18,221.00 | Horizon-Consistent ($\ge 16$-day lag) |
| **Seasonal Naive (Lag 1-7)** | **0.6124** | 357.25 | 98.70 | 0.9176 | 0.30 | 0.00% | 0.00 | 18,340.00 | 1-Day Origin Persistence |
| **Linear Regression (OLS)** | **0.6472** | 470.97 | 128.56 | 0.8567 | 11.82 | 1.65% | 0.00 | 20,238.80 | Unregularized Linear ($\ge 16$-day lag) |
| **Ridge Regression ($\alpha=100.0$)** | 0.6481 | 470.02 | 128.31 | 0.8573 | 8.85 | 1.55% | 0.00 | 20,268.12 | **Selected Best Linear Pipeline** |
| **Lasso Regression ($\alpha=0.001$)** | 0.6539 | **456.40** | **121.99** | **0.8655** | 66.43 | 0.86% | 0.00 | 23,346.19 | Sparse Linear Pipeline |
| **Elastic Net ($\alpha=0.001, \ell_1=0.5$)** | 0.6516 | 459.00 | 123.97 | 0.8639 | 158.85 | 1.07% | 0.00 | 21,665.38 | Balanced Regularization Pipeline |

### 6.2 Phase 2 Logistic Regression Surge Classification Benchmark
*Evaluated on holdout ($N=26,730$, 4,463 true surges, 16.70% prevalence):*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold | True Neg | False Pos | False Neg | True Pos | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | **0.8330** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1670 | 0.5000 | 22,267 | 0 | 4,463 | 0 | 4.34 |
| **Logistic Regression (Default)** | 0.8212 | **0.4524** | 0.3363 | 0.3858 | 0.7518 | **0.4180** | 0.5000 | 20,450 | 1,817 | 2,962 | 1,501 | 20.40 |
| **Logistic Regression (Balanced)** | 0.6151 | 0.2705 | **0.7694** | 0.4003 | **0.7529** | 0.4159 | 0.5000 | 13,008 | 9,259 | 1,029 | **3,434** | 17.13 |
| **Logistic Regression (Tuned $\tau=0.29$)** | 0.7465 | 0.3416 | 0.5590 | **0.4241** | 0.7518 | **0.4180** | **0.2897** | 17,459 | 4,808 | 1,968 | 2,495 | 20.40 |

### 6.3 Phase 3 Time Series Econometric Benchmark

#### Part A: Enterprise Aggregated Daily Retail Sales ($N=15$ holdout days)
| Model | RMSLE | RMSE | MAE | $R^2$ | Fit Time (s) | Core Technical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seasonal Naive** | 0.1574 | 139,177.32 | 100,229.67 | -0.5537 | 0.010 | Weekly 7-day cyclical persistence baseline |
| **Holt-Winters** | **0.1295** | **115,406.36** | **82,557.60** | **-0.0683** | 0.633 | Dynamic additive level, trend, and weekly seasonality |
| **SARIMA (Univariate)** | 0.1478 | 130,393.45 | 91,265.18 | -0.3638 | 0.927 | Univariate $(1,1,1) \times (1,1,0)_7$ on $\log(1+y)$ target |
| **SARIMAX (With Exogenous)** | **0.1437** | **121,980.13** | **90,145.81** | **-0.1935** | 2.929 | **State-space model incorporating promotions, paydays, holidays, oil** |

#### Part B: Representative Store-Family Cohort Benchmark ($N=15$ holdout days)
| Series Name | Model | RMSLE | RMSE | MAE | $R^2$ | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Store 44 - GROCERY I | **Seasonal Naive** | 0.1952 | 2,115.18 | 1,541.53 | -0.5613 | 0.000 |
| Store 44 - GROCERY I | **Holt-Winters** | **0.1877** | **2,065.61** | **1,487.48** | **-0.4889** | 0.215 |
| Store 44 - GROCERY I | **SARIMA** | 0.1948 | 2,123.00 | 1,541.98 | -0.5728 | 0.397 |
| Store 44 - GROCERY I | **SARIMAX** | 0.2708 | 2,116.27 | 1,920.24 | -0.5629 | 0.849 |
| Store 44 - BEVERAGES | **Seasonal Naive** | 0.2932 | 3,535.14 | 2,558.47 | -2.7395 | 0.000 |
| Store 44 - BEVERAGES | **Holt-Winters** | **0.2285** | **2,566.73** | **2,063.24** | **-0.9713** | 0.296 |
| Store 44 - BEVERAGES | **SARIMA** | 0.2925 | 3,540.66 | 2,538.01 | -2.7511 | 0.286 |
| Store 44 - BEVERAGES | **SARIMAX** | 0.2737 | 3,151.11 | 2,405.38 | -1.9711 | 0.960 |
| Store 45 - PRODUCE | **Seasonal Naive** | 0.1392 | 962.14 | 784.98 | 0.5241 | 0.000 |
| Store 45 - PRODUCE | **Holt-Winters** | **0.1023** | **742.60** | **569.28** | **0.7165** | 0.224 |
| Store 45 - PRODUCE | **SARIMA** | 0.1374 | 953.65 | 771.55 | 0.5324 | 0.308 |
| Store 45 - PRODUCE | **SARIMAX** | 0.1270 | 821.89 | 633.90 | 0.6527 | 0.622 |
| Store 47 - CLEANING | **Seasonal Naive** | 0.2374 | 608.38 | 461.20 | -0.8204 | 0.000 |
| Store 47 - CLEANING | **Holt-Winters** | **0.2334** | **582.88** | 486.04 | **-0.6710** | 0.382 |
| Store 47 - CLEANING | **SARIMA** | 0.2440 | 623.90 | 481.93 | -0.9145 | 0.483 |
| Store 47 - CLEANING | **SARIMAX** | 0.2985 | 674.80 | 565.86 | -1.2397 | 0.966 |
| Store 1 - DELI | **Seasonal Naive** | 0.3069 | 37.09 | 26.17 | -0.1321 | 0.000 |
| Store 1 - DELI | **Holt-Winters** | 0.3235 | 36.44 | 24.62 | -0.0925 | 0.364 |
| Store 1 - DELI | **SARIMA** | 0.3095 | 37.04 | 27.27 | -0.1289 | 0.390 |
| Store 1 - DELI | **SARIMAX** | **0.1937** | **20.86** | **17.94** | **0.6420** | 2.158 |

#### Part C: Cohort Mean Summary
| Model | Mean RMSLE | Mean RMSE | Mean MAE | Mean $R^2$ | Mean Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Holt-Winters** | **0.2151** | **1,198.85** | **926.13** | **-0.3014** | 0.296 |
| **SARIMAX** | 0.2327 | 1,356.99 | 1,108.67 | -0.4958 | 1.111 |
| **Seasonal Naive** | 0.2344 | 1,451.58 | 1,074.47 | -0.7458 | 0.000 |
| **SARIMA** | 0.2356 | 1,455.65 | 1,072.15 | -0.7670 | 0.373 |

#### Part D: Global Anchor Comparison (All 1,782 Series / 26,730 Validation Observations)
| Model | RMSLE | RMSE | MAE | $R^2$ | Information Scope |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Seasonal Naive (Immediate Lag 1-7)** | **0.6124** | **357.25** | **98.70** | **0.9176** | Full Store-Family Matrix ($N=26,730$) |
| **Ridge Regression (Phase 1 Selected)** | 0.6481 | 470.02 | 128.31 | 0.8573 | Pooled Machine Learning Linear Pipeline |
| **Lasso Regression (Phase 1 Sparse)** | 0.6539 | 456.40 | 121.99 | 0.8655 | Pooled $L_1$-regularized Sparse Pipeline |

---

## 7. Supply Chain Insights & Operational Trade-Offs

### 7.1 The Holding Cost vs. Stockout Cost Penalty Trade-Off
In grocery inventory management, misclassification costs are highly asymmetrical:
$$\text{Total Operational Cost} = C_{\text{stockout}} \cdot FN + C_{\text{holding}} \cdot FP$$
- **Balanced Class Weighting:** Optimal when $C_{\text{stockout}} \gg C_{\text{holding}}$ (e.g., center-store high-margin items, non-perishables). Captures **76.94%** of surges (only 1,029 missed surges), reducing stockout risk by >3x.
- **Tuned Threshold ($\tau = 0.2897$):** Optimal when working capital is constrained and holding costs/spoilage are severe (e.g., fresh produce, dairy). Achieves highest F1-score (**0.4241**) and cuts false alarms by nearly 50% (4,808 vs. 9,259 FP).

### 7.2 Why Seasonal Naive Outperforms Global Linear Regression on Micro Demand
Seasonal Naive achieves $R^2 = 0.9180$ and RMSE = 356.20 compared to Ridge ($R^2 = 0.8573$, RMSE = 470.02) because:
1. Ecuadorian grocery purchasing exhibits intense weekly routine persistence. Sunday shopping habits at a specific local store repeat with high fidelity week after week.
2. Global linear regression fits a pooled parameter space across all 1,782 series simultaneously. While one-hot indicators capture store and family intercepts, a single global regression slope struggles to model micro-localized variances as accurately as pure historical persistence for stable staple categories.

---

## 8. Directory Layout & Module Responsibilities

```text
retail-sales-prediction-and-forecasting/
├── .env.example              # Environment variables template
├── .gitignore                # Comprehensive Git exclusion rules
├── LICENSE                   # Open-source MIT License
├── PROJECT_DETAILS.md        # Technical architecture documentation
├── README.md                 # Project guide, installation, and overview
├── SUBMISSION_CHECKLIST.md   # Pre-submission verification checklist
├── VIVA_PREPARATION.md       # Comprehensive viva voce exam preparation guide
├── requirements.txt          # Production and experimentation dependencies
├── data/
│   ├── raw/                  # Kaggle Store Sales CSVs (git-ignored)
│   │   ├── README.md         # Download and placement instructions
│   │   └── .gitkeep
│   └── processed/            # Preprocessed intermediate files (git-ignored)
│       └── .gitkeep
├── models/                   # Serialized model artifacts (.joblib)
│   ├── best_linear_model.joblib      # Serialized Ridge pipeline (~6 KB)
│   ├── best_logistic_model.joblib    # Serialized Logistic pipeline & thresholds (~40 KB)
│   ├── best_time_series_model.joblib # Serialized SARIMAX forecaster (~18 MB)
│   └── .gitkeep
├── notebooks/                # Modular, sequentially ordered Jupyter notebooks
│   ├── 01_data_exploration_and_preprocessing.ipynb # Comprehensive EDA, data cleaning, feature engineering & parquet serialization
│   ├── 02_linear_regression.ipynb                  # Multi-algorithm continuous sales regression (OLS, Ridge, Lasso, ElasticNet, DecisionTree)
│   ├── 03_logistic_regression.ipynb                # Demand surge classification & threshold tuning (Logistic, Balanced, DecisionTree)
│   ├── 04_time_series_forecasting.ipynb            # Multi-algorithm time-series forecasting (Naive, MA, Holt-Winters, SARIMA, SARIMAX)
│   └── 05_final_model_comparison.ipynb             # Consolidated cross-task evaluation, executive synthesis & deployment guide
├── reports/                  # Project reports and documentation
│   ├── figures/              # Generated visualizations
│   │   ├── daily_sales_trend.png
│   │   ├── day_of_week_seasonality.png
│   │   ├── linear_regression_actual_vs_pred.png
│   │   ├── linear_regression_coefficients.png
│   │   ├── linear_regression_model_comparison.png
│   │   ├── linear_regression_residuals.png
│   │   ├── logistic_regression_coefficients.png
│   │   ├── logistic_regression_confusion_matrices.png
│   │   ├── logistic_regression_roc_pr_curves.png
│   │   ├── oil_sales_relationship.png
│   │   ├── payday_impact.png
│   │   ├── promotion_sales_impact.png
│   │   ├── store_types_and_clusters.png
│   │   ├── time_series_aggregated_forecast.png
│   │   ├── time_series_cohort_forecasts.png
│   │   ├── time_series_model_comparison.png
│   │   ├── time_series_residuals.png
│   │   └── top_families.png
│   ├── academic_project_report.md
│   ├── eda_summary.md
│   ├── linear_regression_comparison.csv
│   ├── linear_regression_comparison.md
│   ├── logistic_regression_evaluation.md
│   ├── logistic_regression_metrics.csv
│   ├── model_evaluation_metrics.md
│   ├── time_series_evaluation.md
│   └── time_series_metrics.csv
└── src/                      # Production Python source modules
    ├── __init__.py           # Package initializer
    ├── audit_data.py         # Dataset availability & validation checks
    ├── config.py             # Filepaths, seeds, cutoffs, and global parameters
    ├── data_loader.py        # Dataset ingestion & type optimization
    ├── eda.py                # Analytical plotting routines
    ├── evaluate.py           # RMSLE, regression, and classification metrics
    ├── linear_regression.py  # Dummy, OLS, Ridge, Lasso, ElasticNet pipelines
    ├── logistic_regression.py# Binary surge classification pipelines
    ├── preprocess.py         # Cleaning, calendar features, lags, & splits
    ├── run_eda.py            # Automated EDA runner
    ├── run_linear_regression.py # Automated Phase 1 runner
    ├── run_logistic_regression.py # Automated Phase 2 runner
    ├── run_time_series.py    # Automated Phase 3 runner
    └── time_series.py        # SeasonalNaive, HoltWinters, SARIMA, SARIMAX
```
