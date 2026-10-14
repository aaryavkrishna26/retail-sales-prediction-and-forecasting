# Dataset Storage Guide: Kaggle Store Sales - Time Series Forecasting

This project utilizes the **Store Sales - Time Series Forecasting** dataset from Kaggle (Corporación Favorita, Ecuador).

## Raw Data Files Expected

Place the following CSV files into this directory (`data/raw/`):

1. **`train.csv`**:
   - Training time series data containing `id`, `date`, `store_nbr`, `family`, `sales`, and `onpromotion`.
   - Date range: January 1, 2013 to August 15, 2017.
   - Target variable: `sales` (total sales for a given product family at a particular store).

2. **`test.csv`**:
   - Testing time series data for the subsequent 16 days (August 16, 2017 to August 31, 2017).
   - Features: `id`, `date`, `store_nbr`, `family`, `onpromotion`.

3. **`stores.csv`**:
   - Store metadata including `store_nbr`, `city`, `state`, `type` (store type A-E), and `cluster` (grouping of similar stores).

4. **`oil.csv`**:
   - Daily Brent crude oil price (`date`, `dcoilwtico`). Ecuador is an oil-dependent economy; oil shocks directly affect purchasing power.

5. **`holidays_events.csv`**:
   - National, regional, and local holidays and events (`date`, `type`, `locale`, `locale_name`, `description`, `transferred`).

6. **`transactions.csv`**:
   - Daily store transaction counts (`date`, `store_nbr`, `transactions`).

7. **`sample_submission.csv`**:
   - Submission format containing `id` and predicted `sales`.

## Download Instructions

Download the dataset from Kaggle using either of the following methods:

### Option A: Kaggle CLI (Recommended)
Ensure your `kaggle.json` API token is configured in `~/.kaggle/kaggle.json`:
```bash
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/
cd data/raw
unzip store-sales-time-series-forecasting.zip
```

### Option B: Direct Web Download
1. Navigate to [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
2. Download all files and extract them into this folder (`data/raw/`).

> **Note:** Raw CSV files are strictly excluded from Git tracking via `.gitignore` to prevent committing large data files to GitHub.
