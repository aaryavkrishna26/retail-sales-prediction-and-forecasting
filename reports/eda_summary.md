# Exploratory Data Analysis (EDA) Summary

This document records key exploratory data analysis findings and statistical distributions from the Kaggle Store Sales dataset.

## 1. Dataset Dimensions & Completeness
- **Training Range:** 2013-01-01 to 2017-08-15
- **Forecast Horizon:** 16 days (2017-08-16 to 2017-08-31)
- **Stores Count:** 54 stores across Ecuador (17 cities, 16 states)
- **Product Families:** 33 unique merchandise categories
- **Time Series Series Count:** 54 stores × 33 families = 1,782 distinct time series

## 2. Key Analytical Hypotheses to Investigate
1. **Payday Effects (Bi-weekly Seasonality):**
   - Higher sales surges occur around the 15th and end of each month.
2. **Promotion Elasticity (`onpromotion`):**
   - Correlation between number of items on promotion and category sales volume.
3. **Macroeconomic Shocks (`oil.csv`):**
   - Oil price drop in late 2014 affecting discretionary product families vs. grocery staples.
4. **Holidays & Natural Disasters:**
   - Impact of the 7.8-magnitude earthquake on April 16, 2016 in Manabí and surrounding provinces.
5. **Zero Sales Periods:**
   - Store closures (e.g. New Year's Day January 1st) and newly opened stores with zero sales before opening.
