# Model Evaluation & Benchmarking Framework

## 1. Primary Evaluation Metrics

### Continuous Sales Prediction (Regression / Forecasting)
- **Root Mean Squared Logarithmic Error (RMSLE):**
  $$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\ln(1 + \hat{y}_i) - \ln(1 + y_i))^2}$$
  *Note: RMSLE penalizes underestimations and accounts for exponentially scaled sales volumes.*

- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$

- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$

- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

### High-Demand Surge Classification (Logistic Regression)
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** $\frac{TP}{TP + FP}$ (Minimizing false stockout alarms).
- **Recall:** $\frac{TP}{TP + FN}$ (Catching actual surge demand events).
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC Score:** Model discrimination threshold capability.

---

## 2. Validation Strategy: Temporal Walk-Forward Split
In time series forecasting, standard k-fold random splitting produces severe temporal data leakage.
- **Training Window:** 2013-01-01 to 2017-07-31
- **Validation Window:** 2017-08-01 to 2017-08-15 (15-day holdout mimicking test horizon)
- **Test Set:** 2017-08-16 to 2017-08-31 (16-day Kaggle evaluation period)
