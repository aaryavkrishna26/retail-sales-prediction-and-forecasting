# Exploratory Data Analysis (EDA) Comprehensive Report

## 1. Dataset Overview & Dimensions
- **Dataset:** Kaggle Store Sales - Time Series Forecasting (Corporación Favorita, Ecuador)
- **Historical Horizon:** `2013-01-01` to `2017-08-15` (1,684 unique dates)
- **Test Forecast Horizon:** `2017-08-16` to `2017-08-31` (16 days)
- **Total Training Records:** 3,000,888 observations
- **Total Units Sold:** 1,073,644,928 units
- **Network Scope:** 54 retail stores across 17 cities and 16 provinces
- **Product Hierarchy:** 33 unique product families ($54 \times 33 = 1,782$ distinct time series)

---

## 2. Quantitative Key Findings

### 2.1 Product Family Concentration
- **Highest Volume Category:** `GROCERY I` with 343,462,720 units sold (32.0% of network sales).
- **Other Dominant Categories:** `BEVERAGES`, `PRODUCE`, `CLEANING`, and `DAIRY` collectively constitute over 75% of total retail turnover.
- **Lowest Volume Category:** `BOOKS` with 6,438 units sold.
- **Distribution Insight:** Severe positive skewness in sales volumes justifies evaluating models using Root Mean Squared Logarithmic Error (**RMSLE**) and training linear models on $\log(1 + y)$.

### 2.2 Bi-Weekly Payday Dynamics (Ecuadorian Labor Practices)
- **Regular Day Mean Daily Network Sales:** 636,963 units
- **Payday Mean Daily Network Sales (15th & Month-End):** 645,966 units
- **Observed Payday Demand Lift:** **+1.41%**
- **Modeling Takeaway:** Binary indicator `is_payday` represents a crucial exogenous predictive feature.

### 2.3 Intra-Week Seasonality
- **Weekday Mean Sales:** 573,143 units
- **Weekend Mean Sales (Saturday & Sunday):** 798,657 units
- **Weekend Demand Lift:** **+39.35%**
- **Peak Day:** Saturday and Sunday exhibit the highest grocery footfall and transaction volume across all store types.

### 2.4 Store Hierarchy & Typology
- **Store Types Performance Ranking:**
  - **Type A:** Mean store volume of 39,227,092 units
  - **Type D:** Mean store volume of 19,504,628 units
  - **Type B:** Mean store volume of 18,157,580 units
  - **Type E:** Mean store volume of 14,955,609 units
  - **Type C:** Mean store volume of 10,962,317 units
- **Store Clusters:** 17 store clusters exhibit distinct demand profiles (e.g., urban flagship hypermarkets vs. regional convenience outlets).

### 2.5 Macroeconomic Oil Price Relationship
- Daily Brent crude oil prices dropped sharply from $\approx \$100$/barrel in early 2014 down to $\approx \$30$/barrel in early 2016.
- Ecuador's dollarized economy is heavily oil-export reliant; oil shocks directly impact consumer discretionary spending and import tariffs.
- Missing daily oil prices (43 records due to weekend market closures) require forward/linear interpolation.

### 2.6 Exogenous Shocks: 2016 Ecuador Earthquake
- On **April 16, 2016**, a magnitude 7.8 earthquake struck Manabí and Esmeraldas provinces.
- Followed by massive relief supply procurement and immediate demand spikes in non-perishables (Food, Canned Goods, Water).

---

## 3. Generated Visual Diagnostic Artifacts

All charts have been exported to `reports/figures/`:
1. **[daily_sales_trend.png](file:///c:/retail-sales-prediction-and-forecasting/reports/figures/daily_sales_trend.png)**: Complete 2013-2017 timeline showing December peaks and earthquake demand shocks.
2. **[top_families.png](file:///c:/retail-sales-prediction-and-forecasting/reports/figures/top_families.png)**: Bar plot ranking the top 10 product categories.
3. **[payday_impact.png](file:///c:/retail-sales-prediction-and-forecasting/reports/figures/payday_impact.png)**: Boxplot showing store-level demand lift on the 15th and end of each month.
4. **[day_of_week_seasonality.png](file:///c:/retail-sales-prediction-and-forecasting/reports/figures/day_of_week_seasonality.png)**: Intra-week distribution across Monday through Sunday.
5. **[store_types_and_clusters.png](file:///c:/retail-sales-prediction-and-forecasting/reports/figures/store_types_and_clusters.png)**: Sales variations across store types (A-E) and clusters (1-17).
6. **[oil_sales_relationship.png](file:///c:/retail-sales-prediction-and-forecasting/reports/figures/oil_sales_relationship.png)**: Weekly total sales plotted alongside Brent crude oil price movement.
7. **[promotion_sales_impact.png](file:///c:/retail-sales-prediction-and-forecasting/reports/figures/promotion_sales_impact.png)**: Distribution of sales by promotional intensity quartiles.

---

## 4. Modeling Implications for Next Phases
1. **Target Formulation:** Strong skew requires $\log(1 + \text{sales})$ target transformation for continuous regression.
2. **Feature Set:** Mandatory inclusion of `is_payday`, `dayofweek`, `is_weekend`, `month`, `store_type`, `cluster`, `onpromotion`, and interpolated `dcoilwtico`.
3. **Lag Generation:** Lags must be $\ge 16$ days to avoid data leakage during multi-step test horizon forecasting.
4. **Validation Strategy:** Chronological split (holdout August 1-15, 2017) to match test set structure.
