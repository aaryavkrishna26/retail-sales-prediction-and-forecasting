# Phase 1: Linear Regression Algorithm Comparison Report

## 1. Experimental Protocol
- **Dataset Window (Option B):** 1-Year Rolling Temporal Horizon (`2016-08-01` to `2017-07-31`)
- **Training Observations:** 648,648 rows across 54 stores and 33 product families
- **Validation Observations:** 26,730 rows (`2017-08-01` to `2017-08-15`, identical for all models)
- **Target Formulation:** $\log(1 + \text{sales})$ to directly optimize competition RMSLE
- **Physical Non-negativity:** Predictions are mapped back via $\max(\exp(\hat{z}) - 1, 0)$
- **Leakage Audit:** Lag features strictly look back $\ge 16$ days. `transactions.csv` is excluded.
- **Baseline Inclusion:** Compares `DummyRegressor` (mean baseline), `Seasonal Naive (Lag 16-22, Horizon-Consistent)` (strictly >= 16 days prior), and `Seasonal Naive (Lag 1-7, 1-Day Origin)` (immediate 7-day persistence).

## 2. Benchmark Comparison Table

| Model                                          |   RMSLE |     RMSE |      MAE |      R2 |   Fit_Time_Sec |   Negative_Predictions_Count |   Negative_Predictions_Pct |   Pred_Min |   Pred_Max |
|:-----------------------------------------------|--------:|---------:|---------:|--------:|---------------:|-----------------------------:|---------------------------:|-----------:|-----------:|
| DummyRegressor (Mean)                          |  2.5245 | 1317.35  | 458.251  | -0.1209 |           0.14 |                            0 |                       0    |      32.52 |      32.52 |
| Seasonal Naive (Lag 16-22, Horizon-Consistent) |  0.6388 |  356.203 |  92.1735 |  0.918  |           0.39 |                            0 |                       0    |       0    |   18221    |
| Seasonal Naive (Lag 1-7, 1-Day Origin)         |  0.6124 |  357.246 |  98.695  |  0.9176 |           0.3  |                            0 |                       0    |       0    |   18340    |
| Linear Regression (OLS)                        |  0.6472 |  470.968 | 128.563  |  0.8567 |          11.82 |                          441 |                       1.65 |       0    |   20238.8  |
| Ridge Regression                               |  0.6481 |  470.018 | 128.307  |  0.8573 |           8.85 |                          413 |                       1.55 |       0    |   20268.1  |
| Lasso Regression                               |  0.6539 |  456.405 | 121.993  |  0.8655 |          66.43 |                          230 |                       0.86 |       0    |   23346.2  |
| Elastic Net                                    |  0.6516 |  459.003 | 123.966  |  0.8639 |         158.85 |                          285 |                       1.07 |       0    |   21665.4  |

## 3. Key Observations & Findings
1. **Dramatic Performance Gain ($R^2 \approx 0.857 - 0.866$):** Incorporating dynamic trailing 365-day and day-of-week rolling baseline encodings (zero target leakage, zero train-val distribution shift) elevated linear regression $R^2$ from **0.0113** to **0.8572** (Ridge) and **0.8655** (Lasso), while slashing RMSE from **1,237.26** down to **456.40 - 470.19** (a 62% error reduction) and reducing MAE from 208.01 down to **121.99 - 128.33** (a 41% reduction). RMSLE dropped from 0.7987 down to **0.6472 - 0.6539**.
2. **Zero Target Leakage & Distribution Consistency:** All dynamic baselines use `shift(16)` on trailing 365-day windows. No training or validation row includes its own target or future sales, and both sets share identical rolling formulations.
3. **Seasonal Naive Benchmark Comparison:** Both the horizon-consistent Seasonal Naive ($R^2 = 0.9180$, RMSE = 356.20, MAE = 92.17) and the immediate 7-day Seasonal Naive ($R^2 = 0.9176$, RMSE = 357.25, MAE = 98.70) outperform all global linear models due to the dominant intra-week grocery habit persistence in Ecuador.
4. **Physical Validity:** All predictions remain non-negative via log-space target training and $\max(\exp(\hat{z}) - 1, 0)$ back-transformation.
