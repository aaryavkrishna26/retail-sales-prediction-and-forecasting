"""
Preprocessing, feature engineering, and temporal dataset transformation pipeline.
"""

from typing import Dict, Tuple
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
    return merged


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lags: Tuple[int, ...] = (16, 21, 28),
    windows: Tuple[int, ...] = (7, 14, 30),
) -> pd.DataFrame:
    """
    Compute lag and rolling mean features per store and product family.
    Lags start at >= 16 to prevent test-period forecast horizon leakage.
    """
    data = df.copy()
    data = data.sort_values(by=["store_nbr", "family", "date"]).reset_index(drop=True)

    grouped = data.groupby(["store_nbr", "family"])[TARGET_COL]

    for lag in lags:
        data[f"sales_lag_{lag}"] = grouped.shift(lag).astype("float32")

    for window in windows:
        data[f"sales_rolling_mean_{window}"] = (
            grouped.shift(16).rolling(window=window, min_periods=1).mean().astype("float32")
        )

    return data


def create_surge_classification_target(
    df: pd.DataFrame,
    threshold_quantile: float = 0.80,
) -> pd.DataFrame:
    """
    Create a binary target flag 'is_high_sales' (1 if sales exceeds 80th percentile for store/family, else 0).
    Used for Logistic Regression high-demand event prediction.
    """
    data = df.copy()
    quantile_thresh = data.groupby(["store_nbr", "family"])[TARGET_COL].transform(
        lambda s: s.quantile(threshold_quantile)
    )
    data[SURGE_TARGET_COL] = (data[TARGET_COL] > quantile_thresh).astype("int8")
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
