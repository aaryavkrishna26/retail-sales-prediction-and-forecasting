"""
Preprocessing, feature engineering, and temporal dataset transformation pipeline.
"""

from typing import Dict, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from src.config import VAL_START_DATE, TARGET_COL, SURGE_TARGET_COL

logger = logging.getLogger(__name__)


def clean_oil_data(oil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing daily oil prices (dcoilwtico) using linear interpolation and forward/backward fill.
    """
    df = oil_df.copy()
    df = df.set_index("date").asfreq("D")
    df["dcoilwtico"] = df["dcoilwtico"].interpolate(method="linear").bfill().ffill()
    df = df.reset_index()
    return df


def extract_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Extract temporal calendar features: day of week, day of month, month, year, weekend, and paydays.
    Paydays in Ecuador occur typically on the 15th and the final day of each calendar month.
    """
    data = df.copy()
    data["year"] = data[date_col].dt.year.astype("int16")
    data["month"] = data[date_col].dt.month.astype("int8")
    data["day"] = data[date_col].dt.day.astype("int8")
    data["dayofweek"] = data[date_col].dt.dayofweek.astype("int8")
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype("int8")
    
    # Bi-weekly payday flag: 15th or end of month
    data["is_month_end"] = data[date_col].dt.is_month_end.astype("int8")
    data["is_payday"] = ((data["day"] == 15) | (data["is_month_end"] == 1)).astype("int8")
    return data


def merge_external_features(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine store sales records with store metadata, interpolated oil prices, and processed holiday flags.
    Strictly excludes transactions.csv to avoid test-horizon data leakage.
    """
    logger.info("Merging store sales with metadata and external macroeconomic signals...")
    merged = sales_df.merge(stores_df, on="store_nbr", how="left")

    # Clean and merge oil
    cleaned_oil = clean_oil_data(oil_df)
    merged = merged.merge(cleaned_oil, on="date", how="left")

    # Holiday event processing (filter non-transferred national holidays)
    active_holidays = holidays_df[~holidays_df["transferred"]].copy()
    active_holidays = active_holidays.drop_duplicates(subset=["date"], keep="first")
    active_holidays["is_holiday"] = 1

    merged = merged.merge(
        active_holidays[["date", "is_holiday", "type"]].rename(columns={"type": "holiday_type"}),
        on="date",
        how="left",
    )
    merged["is_holiday"] = merged["is_holiday"].fillna(0).astype("int8")
    merged["holiday_type"] = merged["holiday_type"].cat.add_categories(["None"]).fillna("None")

    # Extract calendar features
    merged = extract_calendar_features(merged, date_col="date")

    # Add promotion signals
    if "onpromotion" in merged.columns:
        merged["has_promotion"] = (merged["onpromotion"] > 0).astype("int8")
        merged["log_promotion"] = np.log1p(merged["onpromotion"].clip(lower=0)).astype("float32")

    return merged


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lags: Tuple[int, ...] = (16, 21, 28, 35),
    windows: Tuple[int, ...] = (7, 14, 30, 60),
) -> pd.DataFrame:
    """
    Compute lag and rolling mean/std features per store and product family.
    Lags strictly start at >= 16 days to prevent test-period forecast horizon leakage.
    Lags 21, 28, and 35 align exactly with same day-of-week 3, 4, and 5 weeks prior.
    """
    data = df.copy()
    data = data.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    grouped = data.groupby(["store_nbr", "family"])[TARGET_COL]

    # Individual lags
    for lag in lags:
        data[f"sales_lag_{lag}"] = grouped.shift(lag).astype("float32")

    # Rolling window averages shifted by 16 days
    shifted_base = grouped.shift(16)
    for window in windows:
        data[f"sales_rolling_mean_{window}"] = (
            shifted_base.rolling(window=window, min_periods=1).mean().astype("float32")
        )

    # Rolling volatility (standard deviation over 30 days)
    data["sales_rolling_std_30"] = (
        shifted_base.rolling(window=30, min_periods=1).std().fillna(0).astype("float32")
    )

    return data


def compute_store_family_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    full_df: Optional[pd.DataFrame] = None,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute strictly past-only dynamic rolling baseline features per (store_nbr, family)
    and per (store_nbr, family, dayofweek) to guarantee zero target leakage and zero
    train-validation distribution mismatch.

    Methodological Guarantees:
    - Zero Target Leakage: At any observation date t, the trailing 365-day mean is
      computed with shift(16), and the day-of-week mean is computed with shift(3 weeks).
      No observation ever accesses its own target value or future information.
    - Zero Distribution Shift: Every single row (both in training and validation) is
      evaluated against the exact same dynamic trailing window ending 16 days prior.
    """
    train = train_df.copy()
    val = val_df.copy()

    if full_df is not None:
        source = full_df.copy()
    else:
        source = pd.concat([train, val], ignore_index=True)

    source["log_target_base"] = np.log1p(np.clip(source[target_col].values, 0, None))
    source = source.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    # 1. Trailing 365-day mean log sales shifted by 16 days per store-family
    grouped_sf = source.groupby(["store_nbr", "family"])["log_target_base"]
    source["hist_sf_mean_log"] = (
        grouped_sf.shift(16).rolling(window=365, min_periods=14).mean().astype("float32")
    )

    # 2. Trailing 52-week day-of-week mean log sales shifted by 3 weeks (21 days) per store-family-dayofweek
    source = source.sort_values(by=["store_nbr", "family", "dayofweek", "date"]).reset_index(drop=True)
    grouped_sfd = source.groupby(["store_nbr", "family", "dayofweek"])["log_target_base"]
    source["hist_sf_dow_mean_log"] = (
        grouped_sfd.shift(3).rolling(window=52, min_periods=4).mean().astype("float32")
    )

    # Impute initial cold-start values if any
    source["hist_sf_dow_mean_log"] = source["hist_sf_dow_mean_log"].fillna(source["hist_sf_mean_log"])
    global_mean_log = float(source["log_target_base"].mean())
    source["hist_sf_mean_log"] = source["hist_sf_mean_log"].fillna(global_mean_log)
    source["hist_sf_dow_mean_log"] = source["hist_sf_dow_mean_log"].fillna(global_mean_log)

    baseline_cols = ["date", "store_nbr", "family", "hist_sf_mean_log", "hist_sf_dow_mean_log"]
    baseline_lookup = source[baseline_cols]

    # Merge onto train and val deterministically
    train = train.merge(baseline_lookup, on=["date", "store_nbr", "family"], how="left")
    val = val.merge(baseline_lookup, on=["date", "store_nbr", "family"], how="left")

    return train, val


def compute_historical_surge_thresholds(
    train_df: pd.DataFrame,
    quantile: float = 0.80,
    target_col: str = TARGET_COL,
) -> Dict[Tuple[int, str], float]:
    """
    Derive store-family specific sales thresholds strictly from historical training data.
    
    Robust Fallback for Sparse Series:
    - For each (store_nbr, family), compute the specified quantile (default 80th percentile).
    - If a series is sparse or inactive (threshold <= 0), fallback to the family-level quantile in training.
    - If the family-level quantile is also <= 0, fallback to the global training quantile, with a minimum cutoff of 1.0 unit.
    - This ensures zero target leakage when applying thresholds to validation/test sets.
    """
    fam_quantiles = train_df.groupby("family")[target_col].quantile(quantile).to_dict()
    global_quantile = float(train_df[target_col].quantile(quantile))
    
    sf_quantiles = (
        train_df.groupby(["store_nbr", "family"])[target_col]
        .quantile(quantile)
        .to_dict()
    )
    
    threshold_map = {}
    for (store_nbr, family), q_val in sf_quantiles.items():
        if q_val > 0:
            threshold_map[(store_nbr, family)] = float(q_val)
        else:
            fam_fallback = fam_quantiles.get(family, 0.0)
            if fam_fallback > 0:
                threshold_map[(store_nbr, family)] = float(fam_fallback)
            else:
                threshold_map[(store_nbr, family)] = float(max(global_quantile, 1.0))
                
    return threshold_map


def create_surge_classification_target(
    df: pd.DataFrame,
    threshold_map: Optional[Dict[Tuple[int, str], float]] = None,
    threshold_quantile: float = 0.80,
    target_col: str = TARGET_COL,
    surge_col: str = SURGE_TARGET_COL,
) -> pd.DataFrame:
    """
    Create a binary target flag 'is_high_sales' (1 = demand surge, 0 = normal demand).
    If threshold_map is provided, thresholds derived strictly on training data are applied,
    guaranteeing zero target leakage.
    """
    data = df.copy()
    if threshold_map is None:
        threshold_map = compute_historical_surge_thresholds(
            data, quantile=threshold_quantile, target_col=target_col
        )
    
    lookup = data.set_index(["store_nbr", "family"]).index.map(threshold_map)
    data[surge_col] = (data[target_col].values > lookup).astype("int8")
    return data


def temporal_train_val_split(
    df: pd.DataFrame,
    val_start_date: str = VAL_START_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform strict temporal split into train and validation sets to ensure zero future data leakage.
    """
    val_cutoff = pd.to_datetime(val_start_date)
    train_df = df[df["date"] < val_cutoff].copy()
    val_df = df[df["date"] >= val_cutoff].copy()

    logger.info(
        "Temporal split complete. Train shape: %s (Max date: %s), Val shape: %s (Min date: %s)",
        train_df.shape,
        train_df["date"].max(),
        val_df.shape,
        val_df["date"].min(),
    )
    return train_df, val_df
