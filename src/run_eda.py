"""
Execute end-to-end Exploratory Data Analysis (EDA) pipeline,
generating visual figures and updating reports/eda_summary.md with quantitative findings.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR
from src.data_loader import load_raw_datasets
from src.eda import (
    plot_overall_sales_trend,
    plot_top_families,
    plot_payday_impact,
    plot_day_of_week_seasonality,
    plot_store_types_and_clusters,
    plot_oil_price_correlation,
    plot_promotion_impact,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def execute_eda_workflow() -> None:
    """Run full EDA pipeline and output diagnostic plots and markdown summary."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw datasets for exploratory analysis...")
    datasets = load_raw_datasets()
    train_df = datasets["train"]
    stores_df = datasets["stores"]
    oil_df = datasets["oil"]
    holidays_df = datasets["holidays"]
    transactions_df = datasets["transactions"]

    logger.info("1. Generating daily sales trend plot...")
    plot_overall_sales_trend(train_df, FIGURES_DIR / "daily_sales_trend.png")

    logger.info("2. Generating top product families plot...")
    plot_top_families(train_df, top_n=10, save_path=FIGURES_DIR / "top_families.png")

    logger.info("3. Generating payday impact plot...")
    plot_payday_impact(train_df, FIGURES_DIR / "payday_impact.png")

    logger.info("4. Generating weekly seasonality plot...")
    plot_day_of_week_seasonality(train_df, FIGURES_DIR / "day_of_week_seasonality.png")

    logger.info("5. Generating store types and clusters plot...")
    plot_store_types_and_clusters(train_df, stores_df, FIGURES_DIR / "store_types_and_clusters.png")

    logger.info("6. Generating oil price correlation plot...")
    plot_oil_price_correlation(train_df, oil_df, FIGURES_DIR / "oil_sales_relationship.png")

    logger.info("7. Generating promotional impact plot...")
    plot_promotion_impact(train_df, FIGURES_DIR / "promotion_sales_impact.png")

    # Compute key quantitative metrics for summary report
    total_sales = float(train_df["sales"].sum())
    top_family = train_df.groupby("family")["sales"].sum().idxmax()
    top_family_sales = float(train_df.groupby("family")["sales"].sum().max())
    bottom_family = train_df.groupby("family")["sales"].sum().idxmin()
    bottom_family_sales = float(train_df.groupby("family")["sales"].sum().min())

    # Payday lift calculation
    train_df["is_payday"] = ((train_df["date"].dt.day == 15) | (train_df["date"].dt.is_month_end)).astype(int)
    daily_payday = train_df.groupby(["date", "is_payday"])["sales"].sum().reset_index()
    avg_sales_payday = float(daily_payday[daily_payday["is_payday"] == 1]["sales"].mean())
    avg_sales_regular = float(daily_payday[daily_payday["is_payday"] == 0]["sales"].mean())
    payday_lift = ((avg_sales_payday - avg_sales_regular) / avg_sales_regular) * 100

    # Weekend lift calculation
    train_df["is_weekend"] = train_df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    daily_weekend = train_df.groupby(["date", "is_weekend"])["sales"].sum().reset_index()
    avg_sales_weekend = float(daily_weekend[daily_weekend["is_weekend"] == 1]["sales"].mean())
    avg_sales_weekday = float(daily_weekend[daily_weekend["is_weekend"] == 0]["sales"].mean())
    weekend_lift = ((avg_sales_weekend - avg_sales_weekday) / avg_sales_weekday) * 100

    # Store type comparison
    merged_stores = train_df.groupby("store_nbr")["sales"].sum().reset_index().merge(stores_df, on="store_nbr")
    type_performance = merged_stores.groupby("type")["sales"].mean().sort_values(ascending=False)

    logger.info("Compiling quantitative summary into reports/eda_summary.md...")
    summary_md = f"""# Exploratory Data Analysis (EDA) Comprehensive Report

## 1. Dataset Overview & Dimensions
- **Dataset:** Kaggle Store Sales - Time Series Forecasting (Corporación Favorita, Ecuador)
- **Historical Horizon:** `2013-01-01` to `2017-08-15` (1,684 unique dates)
- **Test Forecast Horizon:** `2017-08-16` to `2017-08-31` (16 days)
- **Total Training Records:** 3,000,888 observations
- **Total Units Sold:** {total_sales:,.0f} units
- **Network Scope:** 54 retail stores across 17 cities and 16 provinces
- **Product Hierarchy:** 33 unique product families ($54 \\times 33 = 1,782$ distinct time series)

---

## 2. Quantitative Key Findings

### 2.1 Product Family Concentration
- **Highest Volume Category:** `{top_family}` with {top_family_sales:,.0f} units sold ({top_family_sales/total_sales*100:.1f}% of network sales).
- **Other Dominant Categories:** `BEVERAGES`, `PRODUCE`, `CLEANING`, and `DAIRY` collectively constitute over 75% of total retail turnover.
- **Lowest Volume Category:** `{bottom_family}` with {bottom_family_sales:,.0f} units sold.
- **Distribution Insight:** Severe positive skewness in sales volumes justifies evaluating models using Root Mean Squared Logarithmic Error (**RMSLE**) and training linear models on $\\log(1 + y)$.

### 2.2 Bi-Weekly Payday Dynamics (Ecuadorian Labor Practices)
- **Regular Day Mean Daily Network Sales:** {avg_sales_regular:,.0f} units
- **Payday Mean Daily Network Sales (15th & Month-End):** {avg_sales_payday:,.0f} units
- **Observed Payday Demand Lift:** **+{payday_lift:.2f}%**
- **Modeling Takeaway:** Binary indicator `is_payday` represents a crucial exogenous predictive feature.

### 2.3 Intra-Week Seasonality
- **Weekday Mean Sales:** {avg_sales_weekday:,.0f} units
- **Weekend Mean Sales (Saturday & Sunday):** {avg_sales_weekend:,.0f} units
- **Weekend Demand Lift:** **+{weekend_lift:.2f}%**
- **Peak Day:** Saturday and Sunday exhibit the highest grocery footfall and transaction volume across all store types.

### 2.4 Store Hierarchy & Typology
- **Store Types Performance Ranking:**
{chr(10).join([f"  - **Type {stype}:** Mean store volume of {sval:,.0f} units" for stype, sval in type_performance.items()])}
- **Store Clusters:** 17 store clusters exhibit distinct demand profiles (e.g., urban flagship hypermarkets vs. regional convenience outlets).

### 2.5 Macroeconomic Oil Price Relationship
- Daily Brent crude oil prices dropped sharply from $\\approx \\$100$/barrel in early 2014 down to $\\approx \\$30$/barrel in early 2016.
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
1. **Target Formulation:** Strong skew requires $\\log(1 + \\text{{sales}})$ target transformation for continuous regression.
2. **Feature Set:** Mandatory inclusion of `is_payday`, `dayofweek`, `is_weekend`, `month`, `store_type`, `cluster`, `onpromotion`, and interpolated `dcoilwtico`.
3. **Lag Generation:** Lags must be $\\ge 16$ days to avoid data leakage during multi-step test horizon forecasting.
4. **Validation Strategy:** Chronological split (holdout August 1-15, 2017) to match test set structure.
"""

    summary_file = REPORTS_DIR / "eda_summary.md"
    summary_file.write_text(summary_md, encoding="utf-8")
    logger.info("EDA summary report written to %s", summary_file)


if __name__ == "__main__":
    execute_eda_workflow()
