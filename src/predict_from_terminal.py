# -*- coding: utf-8 -*-
"""
Interactive terminal-based prediction system for Retail Sales Prediction & Forecasting.

Provides three prediction phases via an easy-to-use menu-driven interface:
  1. Linear Regression    - Predict daily continuous sales for any store, product family, and date.
  2. Logistic Regression  - Predict demand surge probability & stockout alert using calibrated threshold.
  3. Time Series          - Forecast future sales across enterprise network or specific store-family series.

Automatic Feature Generation:
  - For Phase 1 & Phase 2, users normally ONLY enter: Store Number, Product Family, and Prediction Date.
  - All remaining features (lags, rolling stats, dynamic baselines, store metadata, oil prices,
    holidays, promotions, calendar signals) are automatically retrieved or calculated from project data.
  - For future dates beyond historical coverage, prompts ONLY for genuinely unavailable values with
    transparent explanations and sensible defaults.
  - Zero target leakage is strictly maintained.

Launch from project root:
    python src/predict_from_terminal.py
"""

from __future__ import annotations

import io
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Force UTF-8 standard streams on Windows to prevent cp1252 Unicode errors
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.config import (
    DATA_DIR,
    HOLIDAYS_FILE,
    MODELS_DIR,
    OIL_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STORES_FILE,
    SURGE_TARGET_COL,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FILE,
)
from src.linear_regression import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.logistic_regression import (
    LOGISTIC_CATEGORICAL_FEATURES,
    LOGISTIC_NUMERIC_FEATURES,
)
from src.preprocess import (
    clean_oil_data,
    compute_historical_surge_thresholds,
    extract_calendar_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIVIDER = "=" * 65
SUBDIVIDER = "-" * 65

VALID_STORE_RANGE = (1, 54)

VALID_FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
    "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES", "HOME CARE",
    "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE", "LIQUOR,WINE,BEER",
    "MAGAZINES", "MEATS", "PERSONAL CARE", "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS", "PRODUCE",
    "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
]

COMMON_FAMILIES = [
    "GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY", "DELI",
]

# Dates marking boundaries of project datasets
HISTORICAL_TRAIN_START = pd.to_datetime("2016-08-01")
HISTORICAL_SALES_END = pd.to_datetime("2017-08-15")   # Last recorded sales in train/val
TEST_COVERAGE_END = pd.to_datetime("2017-08-31")      # Last date in test.csv / oil.csv
TUNED_SURGE_THRESHOLD = 0.289729                      # Optimal threshold from training


# ===========================================================================
# Console Input Helpers
# ===========================================================================

def print_header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def prompt(msg: str) -> str:
    """Read a stripped, non-empty string."""
    while True:
        val = input(f"  {msg}: ").strip()
        if val:
            return val


def prompt_int(msg: str, lo: int, hi: int, default: Optional[int] = None) -> int:
    hint = f" [{lo}-{hi}]" if default is None else f" [{lo}-{hi}, default={default}]"
    while True:
        raw = input(f"  {msg}{hint}: ").strip()
        if not raw and default is not None:
            return default
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  [X] Please enter an integer between {lo} and {hi}.")
        except ValueError:
            print("  [X] Invalid integer. Please try again.")


def prompt_float(msg: str, lo: float = 0.0, hi: float = 1e9, default: Optional[float] = None) -> float:
    hint = "" if default is None else f" [default={default}]"
    while True:
        raw = input(f"  {msg}{hint}: ").strip()
        if not raw and default is not None:
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  [X] Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("  [X] Invalid number. Please try again.")


def prompt_date(msg: str, default: Optional[str] = None) -> pd.Timestamp:
    hint = " (YYYY-MM-DD)" if default is None else f" (YYYY-MM-DD) [default={default}]"
    while True:
        raw = input(f"  {msg}{hint}: ").strip()
        if not raw and default is not None:
            raw = default
        try:
            parsed = pd.to_datetime(raw)
            return parsed
        except Exception:
            print("  [X] Invalid date format. Use YYYY-MM-DD (e.g. 2017-08-15).")


def confirm_yn(msg: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {msg} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def prompt_choice(msg: str, options: list[str]) -> str:
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input(f"  {msg} [1-{len(options)}]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"  [X] Enter a choice between 1 and {len(options)}.")
        except ValueError:
            print("  [X] Please enter a valid number.")


def resolve_family_name(raw: str) -> Optional[str]:
    """Resolve user input to canonical product family name."""
    clean = raw.strip().upper()
    if clean in VALID_FAMILIES:
        return clean
    # Common aliases / shortcuts
    alias_map = {
        "GROCERY 1": "GROCERY I",
        "GROCERY 2": "GROCERY II",
        "HOME 1": "HOME AND KITCHEN I",
        "HOME 2": "HOME AND KITCHEN II",
        "LIQUOR": "LIQUOR,WINE,BEER",
        "BEER": "LIQUOR,WINE,BEER",
        "WINE": "LIQUOR,WINE,BEER",
        "BAKERY": "BREAD/BAKERY",
        "BREAD": "BREAD/BAKERY",
        "KITCHEN 1": "HOME AND KITCHEN I",
        "KITCHEN 2": "HOME AND KITCHEN II",
    }
    if clean in alias_map:
        return alias_map[clean]
    # Substring matches
    matches = [f for f in VALID_FAMILIES if clean in f]
    if len(matches) == 1:
        return matches[0]
    return None


def prompt_product_family() -> str:
    """Prompt for product family with auto-resolution and interactive list fallback."""
    print("\n  Common product families:")
    print("    " + ", ".join(COMMON_FAMILIES))
    while True:
        raw = input("  Enter product family (name or ? to list all): ").strip()
        if raw == "?":
            print("\n  All 33 Product Families:")
            for i, fam in enumerate(VALID_FAMILIES, 1):
                print(f"    {i:2d}. {fam}")
            continue
        resolved = resolve_family_name(raw)
        if resolved:
            return resolved
        # Check for multiple partial matches
        clean = raw.upper()
        matches = [f for f in VALID_FAMILIES if clean in f]
        if matches:
            print("  Multiple matches found:")
            choice = prompt_choice("Select family", matches)
            return choice
        print("  [X] Unknown family. Enter exact name or '?' to browse all 33 families.")


# ===========================================================================
# Centralized Data Manager (In-Memory Cached Singleton)
# ===========================================================================

class TerminalDataManager:
    """
    Manages high-speed in-memory data tables and historical series lookups.
    Loads raw and preprocessed data once and caches indexed structures.
    """
    _instance: Optional[TerminalDataManager] = None

    def __init__(self):
        print("\n  Initializing project data context ... (takes ~0.3s)")
        # 1. Stores metadata
        if not STORES_FILE.exists():
            raise FileNotFoundError(f"stores.csv not found at {STORES_FILE}")
        self.stores_df = pd.read_csv(STORES_FILE).set_index("store_nbr")

        # 2. Daily continuous oil series
        if not OIL_FILE.exists():
            raise FileNotFoundError(f"oil.csv not found at {OIL_FILE}")
        oil_raw = pd.read_csv(OIL_FILE, parse_dates=["date"])
        self.oil_series = clean_oil_data(oil_raw).set_index("date")["dcoilwtico"]

        # 3. Holiday events
        if not HOLIDAYS_FILE.exists():
            raise FileNotFoundError(f"holidays_events.csv not found at {HOLIDAYS_FILE}")
        hol_raw = pd.read_csv(HOLIDAYS_FILE, parse_dates=["date"])
        active_hol = hol_raw[~hol_raw["transferred"]].drop_duplicates("date")
        self.holidays_dict = active_hol.set_index("date")["type"].to_dict()

        # 4. Test dataset for forward promotions (2017-08-16 to 2017-08-31)
        self.test_promo_lookup: dict[tuple[int, str, pd.Timestamp], int] = {}
        if TEST_FILE.exists():
            test_df = pd.read_csv(TEST_FILE, parse_dates=["date"])
            for _, r in test_df.iterrows():
                self.test_promo_lookup[(int(r["store_nbr"]), str(r["family"]), pd.to_datetime(r["date"]))] = int(r["onpromotion"])

        # 5. Pre-engineered historical datasets (train + val: 2016-08-01 to 2017-08-15)
        train_pq = PROCESSED_DATA_DIR / "train_engineered.parquet"
        val_pq = PROCESSED_DATA_DIR / "val_engineered.parquet"
        if not train_pq.exists() or not val_pq.exists():
            raise FileNotFoundError(
                "Processed datasets not found in data/processed/. "
                "Please run `python src/run_eda.py` or inspect processed data directory."
            )

        tr = pd.read_parquet(train_pq)
        va = pd.read_parquet(val_pq)
        self.full_hist = pd.concat([tr, va], ignore_index=True)
        self.full_hist["date"] = pd.to_datetime(self.full_hist["date"])

        # Indexed by (store_nbr, family, date) for instantaneous row retrieval
        self.hist_lookup = self.full_hist.set_index(["store_nbr", "family", "date"])

        # Daily continuous sales series by (store_nbr, family) with zero-filling for closed days
        self._sf_series_cache: dict[tuple[int, str], pd.Series] = {}

        # 80th-percentile surge threshold map computed strictly from historical training data
        self.threshold_map = compute_historical_surge_thresholds(tr, quantile=0.80)

        # Baseline cache for latest trailing stats
        self._baseline_cache: dict[tuple[int, str], tuple[float, float]] = {}

        print("  [OK] Data context initialized successfully.")

    @classmethod
    def get_instance(cls) -> TerminalDataManager:
        if cls._instance is None:
            cls._instance = TerminalDataManager()
        return cls._instance

    def get_store_family_sales_series(self, store_nbr: int, family: str) -> pd.Series:
        """Return daily continuous sales series for 2016-08-01 through 2017-08-15."""
        key = (store_nbr, family)
        if key in self._sf_series_cache:
            return self._sf_series_cache[key]

        sub = self.full_hist[(self.full_hist["store_nbr"] == store_nbr) & (self.full_hist["family"] == family)].copy()
        sub = sub.drop_duplicates("date").set_index("date")
        full_dates = pd.date_range(HISTORICAL_TRAIN_START, HISTORICAL_SALES_END, freq="D")
        sub = sub.reindex(full_dates)
        s = sub["sales"].fillna(0.0)
        self._sf_series_cache[key] = s
        return s

    def get_latest_baselines(self, store_nbr: int, family: str) -> tuple[float, float]:
        """Return (hist_sf_mean_log, hist_sf_dow_mean_log) trailing baseline as of 2017-08-15."""
        key = (store_nbr, family)
        if key in self._baseline_cache:
            return self._baseline_cache[key]

        try:
            row = self.hist_lookup.loc[(store_nbr, family, HISTORICAL_SALES_END)]
            b_mean = float(row["hist_sf_mean_log"])
            b_dow = float(row["hist_sf_dow_mean_log"])
        except KeyError:
            s = self.get_store_family_sales_series(store_nbr, family)
            log_s = np.log1p(np.clip(s, 0, None))
            b_mean = float(log_s.tail(365).mean())
            b_dow = float(log_s.tail(52).mean())

        self._baseline_cache[key] = (b_mean, b_dow)
        return b_mean, b_dow


# ===========================================================================
# Automated Feature Builder (Zero-Leakage & Minimal User Input)
# ===========================================================================

def build_auto_features(
    store_nbr: int,
    family: str,
    pred_date: pd.Timestamp,
    manager: TerminalDataManager,
) -> dict[str, Any]:
    """
    Construct complete feature dictionary for any (store_nbr, family, pred_date).

    - If pred_date <= 2017-08-15: Auto-retrieves exact zero-leakage precomputed row.
    - If 2017-08-16 <= pred_date <= 2017-08-31: Computes all lags and rolling features
      from historical sales ending 2017-08-15 (since lag_16 <= 2017-08-15).
      Retrieves promotions from test.csv, oil from oil.csv, holidays from holidays_events.csv.
      Zero manual feature entry required!
    - If pred_date > 2017-08-31: Genuinely unavailable future external values (oil, promotion)
      are prompted with clear explanations and sensible defaults.
    """
    # 1. Store metadata
    if store_nbr not in manager.stores_df.index:
        raise ValueError(f"Store number {store_nbr} is outside range 1-54.")
    store_row = manager.stores_df.loc[store_nbr]
    store_type = str(store_row["type"])
    store_cluster = int(store_row["cluster"])
    city = str(store_row["city"])

    # 2. Calendar signals
    dayofweek = pred_date.dayofweek
    day = pred_date.day
    month = pred_date.month
    year = pred_date.year
    is_weekend = int(dayofweek in [5, 6])
    is_month_end = int(pred_date.is_month_end)
    is_payday = int(day == 15 or is_month_end == 1)

    # 3. Check dataset coverage status
    lookup_key = (store_nbr, family, pred_date)
    is_historical = lookup_key in manager.hist_lookup.index
    is_test_horizon = (pred_date > HISTORICAL_SALES_END) and (pred_date <= TEST_COVERAGE_END)
    is_future = pred_date > TEST_COVERAGE_END

    actual_sales: Optional[float] = None
    retrieval_notes: list[str] = []

    # -----------------------------------------------------------------------
    # Case A: Date is within historical dataset (2016-08-01 to 2017-08-15)
    # -----------------------------------------------------------------------
    if is_historical:
        hrow = manager.hist_lookup.loc[lookup_key]
        actual_sales = float(hrow["sales"])

        feat: dict[str, Any] = {
            "hist_sf_mean_log": float(hrow["hist_sf_mean_log"]),
            "hist_sf_dow_mean_log": float(hrow["hist_sf_dow_mean_log"]),
            "sales_lag_16": float(hrow["sales_lag_16"]),
            "sales_lag_21": float(hrow["sales_lag_21"]),
            "sales_lag_28": float(hrow["sales_lag_28"]),
            "sales_lag_35": float(hrow["sales_lag_35"]),
            "sales_rolling_mean_7": float(hrow["sales_rolling_mean_7"]),
            "sales_rolling_mean_14": float(hrow["sales_rolling_mean_14"]),
            "sales_rolling_mean_30": float(hrow["sales_rolling_mean_30"]),
            "sales_rolling_mean_60": float(hrow["sales_rolling_mean_60"]),
            "sales_rolling_std_30": float(hrow["sales_rolling_std_30"]),
            "onpromotion": float(hrow["onpromotion"]),
            "has_promotion": float(hrow["has_promotion"]),
            "log_promotion": float(hrow["log_promotion"]),
            "dcoilwtico": float(hrow["dcoilwtico"]),
            "is_weekend": float(is_weekend),
            "is_payday": float(is_payday),
            "is_holiday": float(hrow["is_holiday"]),
            "day": int(day),
            "month": int(month),
            "year": int(year),
            "family": family,
            "store_nbr": store_nbr,
            "type": store_type,
            "cluster": store_cluster,
            "dayofweek": int(dayofweek),
            "actual_sales": actual_sales,
            "date_status": "HISTORICAL_HOLDOUT",
        }

        retrieval_notes = [
            f"Dataset Range: Historical observation ({pred_date.strftime('%Y-%m-%d')})",
            f"Store Metadata: Type '{store_type}', Cluster {store_cluster} ({city})",
            f"Calendar: {pred_date.strftime('%A')} | Weekend: {'Yes' if is_weekend else 'No'} | Payday: {'Yes' if is_payday else 'No'}",
            f"Macro Signals: Oil=${feat['dcoilwtico']:.2f}/bbl | Holiday: {'Yes' if feat['is_holiday'] else 'No'} | Promotion: {int(feat['onpromotion'])} items",
            f"Lag Sales (>=16d cutoff): lag_16={feat['sales_lag_16']:,.1f}, lag_21={feat['sales_lag_21']:,.1f}, lag_28={feat['sales_lag_28']:,.1f}",
            f"Rolling Stats: 7d_mean={feat['sales_rolling_mean_7']:,.1f}, 30d_mean={feat['sales_rolling_mean_30']:,.1f}, 30d_std={feat['sales_rolling_std_30']:,.1f}",
            f"Dynamic Baselines: hist_sf_mean_log={feat['hist_sf_mean_log']:.3f}, hist_sf_dow_mean_log={feat['hist_sf_dow_mean_log']:.3f}",
        ]
        return {"features": feat, "notes": retrieval_notes}

    # -----------------------------------------------------------------------
    # Case B: Date is in competition test horizon (2017-08-16 to 2017-08-31)
    # -----------------------------------------------------------------------
    if is_test_horizon:
        # Lags >= 16 days strictly fall into the known historical series (<= 2017-08-15)
        s = manager.get_store_family_sales_series(store_nbr, family)

        lag_16 = float(s.get(pred_date - pd.Timedelta(days=16), 0.0))
        lag_21 = float(s.get(pred_date - pd.Timedelta(days=21), 0.0))
        lag_28 = float(s.get(pred_date - pd.Timedelta(days=28), 0.0))
        lag_35 = float(s.get(pred_date - pd.Timedelta(days=35), 0.0))

        def _rolling_mean(window: int) -> float:
            dates = [pred_date - pd.Timedelta(days=d) for d in range(16, 16 + window)]
            vals = [s.get(d, np.nan) for d in dates]
            valid = [v for v in vals if not np.isnan(v)]
            return float(np.mean(valid)) if valid else 0.0

        def _rolling_std(window: int = 30) -> float:
            dates = [pred_date - pd.Timedelta(days=d) for d in range(16, 16 + window)]
            vals = [s.get(d, np.nan) for d in dates]
            valid = [v for v in vals if not np.isnan(v)]
            return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

        roll_7 = _rolling_mean(7)
        roll_14 = _rolling_mean(14)
        roll_30 = _rolling_mean(30)
        roll_60 = _rolling_mean(60)
        roll_std_30 = _rolling_std(30)

        # Baseline log statistics
        b_mean, b_dow = manager.get_latest_baselines(store_nbr, family)

        # External signals
        oil_price = float(manager.oil_series.get(pred_date, manager.oil_series.iloc[-1]))
        is_holiday = float(pred_date in manager.holidays_dict)

        # Promotion from test.csv
        onpromotion = float(manager.test_promo_lookup.get((store_nbr, family, pred_date), 0.0))
        has_promotion = float(onpromotion > 0)
        log_promotion = float(np.log1p(onpromotion))

        feat = {
            "hist_sf_mean_log": b_mean,
            "hist_sf_dow_mean_log": b_dow,
            "sales_lag_16": lag_16,
            "sales_lag_21": lag_21,
            "sales_lag_28": lag_28,
            "sales_lag_35": lag_35,
            "sales_rolling_mean_7": roll_7,
            "sales_rolling_mean_14": roll_14,
            "sales_rolling_mean_30": roll_30,
            "sales_rolling_mean_60": roll_60,
            "sales_rolling_std_30": roll_std_30,
            "onpromotion": onpromotion,
            "has_promotion": has_promotion,
            "log_promotion": log_promotion,
            "dcoilwtico": oil_price,
            "is_weekend": float(is_weekend),
            "is_payday": float(is_payday),
            "is_holiday": is_holiday,
            "day": int(day),
            "month": int(month),
            "year": int(year),
            "family": family,
            "store_nbr": store_nbr,
            "type": store_type,
            "cluster": store_cluster,
            "dayofweek": int(dayofweek),
            "actual_sales": None,
            "date_status": "TEST_HORIZON",
        }

        retrieval_notes = [
            f"Dataset Range: Kaggle Test Horizon ({pred_date.strftime('%Y-%m-%d')})",
            f"Store Metadata: Type '{store_type}', Cluster {store_cluster} ({city})",
            f"Calendar: {pred_date.strftime('%A')} | Weekend: {'Yes' if is_weekend else 'No'} | Payday: {'Yes' if is_payday else 'No'}",
            f"External Signals: Oil=${oil_price:.2f}/bbl (oil.csv) | Holiday: {'Yes' if is_holiday else 'No'} | Promotion: {int(onpromotion)} (test.csv)",
            f"Lag Sales (from training series): lag_16={lag_16:,.1f}, lag_21={lag_21:,.1f}, lag_28={lag_28:,.1f}",
            f"Rolling Stats (computed <= 2017-08-15): 7d_mean={roll_7:,.1f}, 30d_mean={roll_30:,.1f}",
            f"Baselines: hist_sf_mean_log={b_mean:.3f}, hist_sf_dow_mean_log={b_dow:.3f}",
        ]
        return {"features": feat, "notes": retrieval_notes}

    # -----------------------------------------------------------------------
    # Case C: Date is beyond historical & test coverage (> 2017-08-31)
    # -----------------------------------------------------------------------
    print(f"\n  [INFO] Date {pred_date.strftime('%Y-%m-%d')} is beyond historical dataset coverage (ending 2017-08-31).")
    print("         Retrieving available historical store profiles and prompting only for missing future signals:")

    # Known store metadata & calendar
    print(f"  [Auto-retrieved] Store {store_nbr}: Type '{store_type}', Cluster {store_cluster} ({city})")
    print(f"  [Auto-retrieved] Calendar: {pred_date.strftime('%A')}, Weekend: {'Yes' if is_weekend else 'No'}, Payday: {'Yes' if is_payday else 'No'}")

    # Holiday
    if pred_date in manager.holidays_dict:
        is_holiday = 1.0
        print(f"  [Auto-retrieved] Holiday: Yes ({manager.holidays_dict[pred_date]})")
    else:
        is_holiday = 1.0 if confirm_yn("Is this date a known public holiday in Ecuador?", default=False) else 0.0

    # Oil price
    last_oil = float(manager.oil_series.iloc[-1])
    print(f"  [Notice] Daily oil price is unrecorded beyond 2017-08-31 (last recorded: ${last_oil:.2f}).")
    oil_price = prompt_float("Enter expected crude oil price (USD/bbl)", lo=0.0, hi=300.0, default=last_oil)

    # Promotion
    print("  [Notice] Promotional schedules are unrecorded for future dates.")
    onpromotion = float(prompt_int("Number of product items on promotion", lo=0, hi=9999, default=0))
    has_promotion = float(onpromotion > 0)
    log_promotion = float(np.log1p(onpromotion))

    # Lags and baselines from store-family historical profile
    s = manager.get_store_family_sales_series(store_nbr, family)
    b_mean, b_dow = manager.get_latest_baselines(store_nbr, family)
    typical_sales = float(s.tail(30).mean())

    print(f"  [Notice] Sales at lag >= 16 days fall beyond recorded dates. Store-family recent daily avg: {typical_sales:,.1f} units.")
    use_baseline = confirm_yn("Use store-family historical profile for lag and rolling features?", default=True)
    if use_baseline:
        lag_16 = typical_sales
        lag_21 = typical_sales
        lag_28 = typical_sales
        lag_35 = typical_sales
        roll_7 = typical_sales
        roll_14 = typical_sales
        roll_30 = typical_sales
        roll_60 = typical_sales
        roll_std_30 = float(s.tail(30).std(ddof=1)) if len(s) > 1 else 0.0
    else:
        user_sales = prompt_float("Enter estimated baseline daily sales for this store-family", lo=0.0, hi=1e7, default=typical_sales)
        lag_16 = lag_21 = lag_28 = lag_35 = user_sales
        roll_7 = roll_14 = roll_30 = roll_60 = user_sales
        roll_std_30 = user_sales * 0.2

    feat = {
        "hist_sf_mean_log": b_mean,
        "hist_sf_dow_mean_log": b_dow,
        "sales_lag_16": lag_16,
        "sales_lag_21": lag_21,
        "sales_lag_28": lag_28,
        "sales_lag_35": lag_35,
        "sales_rolling_mean_7": roll_7,
        "sales_rolling_mean_14": roll_14,
        "sales_rolling_mean_30": roll_30,
        "sales_rolling_mean_60": roll_60,
        "sales_rolling_std_30": roll_std_30,
        "onpromotion": onpromotion,
        "has_promotion": has_promotion,
        "log_promotion": log_promotion,
        "dcoilwtico": oil_price,
        "is_weekend": float(is_weekend),
        "is_payday": float(is_payday),
        "is_holiday": is_holiday,
        "day": int(day),
        "month": int(month),
        "year": int(year),
        "family": family,
        "store_nbr": store_nbr,
        "type": store_type,
        "cluster": store_cluster,
        "dayofweek": int(dayofweek),
        "actual_sales": None,
        "date_status": "FUTURE_PROJECTION",
    }

    retrieval_notes = [
        f"Dataset Range: Future Operational Projection ({pred_date.strftime('%Y-%m-%d')})",
        f"Store Metadata: Type '{store_type}', Cluster {store_cluster} ({city})",
        f"Calendar: {pred_date.strftime('%A')} | Weekend: {'Yes' if is_weekend else 'No'} | Payday: {'Yes' if is_payday else 'No'}",
        f"User/Derived Signals: Oil=${oil_price:.2f}/bbl | Holiday: {'Yes' if is_holiday else 'No'} | Promotion: {int(onpromotion)} items",
        f"Lags & Rolling Features: Populated from store-family historical baseline profile ({typical_sales:,.1f} units)",
        f"Baselines: hist_sf_mean_log={b_mean:.3f}, hist_sf_dow_mean_log={b_dow:.3f}",
    ]
    return {"features": feat, "notes": retrieval_notes}


# ===========================================================================
# Phase 1: Linear Regression (Predict Daily Sales)
# ===========================================================================

def _load_linear_model():
    """Load serialized Lasso pipeline."""
    model_path = MODELS_DIR / "best_linear_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Linear model not found at {model_path}. "
            "Please run `python src/run_linear_regression.py` to train and save the model."
        )
    return joblib.load(model_path)


def predict_linear_regression(cached_data: dict) -> None:
    print_header("PHASE 1 - Linear Regression: Predict Daily Sales")
    print("  Model: Lasso Regularized Pipeline (L1 alpha=0.001)")
    print("  Inputs needed: Store number, Product family, Prediction date.")
    print("  All features will be automatically retrieved and calculated.")

    manager = TerminalDataManager.get_instance()
    model = _load_linear_model()

    # Minimal User Inputs
    print(f"\n{SUBDIVIDER}")
    print("  Step 1: Enter Prediction Target Details")
    print(SUBDIVIDER)
    store_nbr = prompt_int("Store number", *VALID_STORE_RANGE, default=44)
    family = prompt_product_family()
    pred_date = prompt_date("Prediction date", default="2017-08-15")

    # Automated Feature Retrieval
    print(f"\n{SUBDIVIDER}")
    print("  Step 2: Automated Feature Retrieval & Pipeline Construction")
    print(SUBDIVIDER)
    built = build_auto_features(store_nbr, family, pred_date, manager)
    feat = built["features"]

    for note in built["notes"]:
        print(f"  * {note}")

    # Build warning-free DataFrame formatted specifically for the linear pipeline
    # (type, family: str; cluster, dayofweek: int8; numeric: float)
    df_row = pd.DataFrame([{
        **{k: float(feat[k]) for k in NUMERIC_FEATURES},
        "family": str(feat["family"]),
        "type": str(feat["type"]),
        "cluster": np.int8(feat["cluster"]),
        "dayofweek": np.int8(feat["dayofweek"]),
    }])

    # Model Inference
    pred_log = float(model.predict(df_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0])
    pred_sales = float(np.expm1(np.clip(pred_log, 0, None)))

    # Display Results
    print(f"\n{DIVIDER}")
    print(f"  PREDICTION RESULT - Store {store_nbr} | {family}")
    print(f"  Date: {pred_date.strftime('%Y-%m-%d')} ({pred_date.strftime('%A')})")
    print(DIVIDER)
    print(f"  >> Predicted Sales (Units):    {pred_sales:>14,.2f} units  (Rounded: {round(pred_sales):,})")
    print(f"  >> Predicted Log-Sales z:      {pred_log:>14.4f}  [formula: sales = exp(z) - 1]")

    # Ground truth comparison if historical
    if feat.get("actual_sales") is not None:
        actual = feat["actual_sales"]
        abs_err = abs(pred_sales - actual)
        pct_err = ((pred_sales - actual) / (actual + 1e-5)) * 100.0
        print(f"\n  Ground Truth Holdout Comparison:")
        print(f"    * Actual Observed Sales:     {actual:>14,.2f} units")
        print(f"    * Absolute Error:            {abs_err:>14,.2f} units")
        print(f"    * Percentage Deviation:      {pct_err:>+14.1f}%")
    elif feat.get("date_status") == "TEST_HORIZON":
        print("\n  [Note] Date is in competition test window (actual sales is private holdout).")
    else:
        print("\n  [Note] Future projection beyond dataset boundary.")

    # Model Insights
    try:
        reg = model.named_steps["regressor"]
        prep = model.named_steps["preprocessor"]
        ohe = prep.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = NUMERIC_FEATURES + cat_names
        coefs = reg.coef_
        top_positive = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:3]
        top_negative = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:3]
        print("\n  Top Feature Weights (Lasso learned coefficients):")
        print("    Strongest Positive: " + ", ".join(f"{k} (+{v:.3f})" for k, v in top_positive if v > 0))
        print("    Strongest Negative: " + ", ".join(f"{k} ({v:.3f})" for k, v in top_negative if v < 0))
    except Exception:
        pass


# ===========================================================================
# Phase 2: Logistic Regression (Predict Demand Surge)
# ===========================================================================

def _load_logistic_model():
    """Load serialized Logistic pipeline and optimal threshold."""
    model_path = MODELS_DIR / "best_logistic_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Logistic model not found at {model_path}. "
            "Please run `python src/run_logistic_regression.py` to train and save the model."
        )
    artifact = joblib.load(model_path)
    return artifact


def predict_logistic_regression(cached_data: dict) -> None:
    print_header("PHASE 2 - Logistic Regression: Predict Demand Surge")
    print("  Task: Detect whether sales will exceed historical 80th-percentile surge level.")
    print("  Inputs needed: Store number, Product family, Prediction date.")
    print("  Reuses the automated zero-leakage feature pipeline.")

    manager = TerminalDataManager.get_instance()
    art = _load_logistic_model()
    pipeline = art["pipeline"]
    feats_req = art["features"]
    optimal_threshold = float(art.get("optimal_threshold", TUNED_SURGE_THRESHOLD))

    # Minimal User Inputs
    print(f"\n{SUBDIVIDER}")
    print("  Step 1: Enter Prediction Target Details")
    print(SUBDIVIDER)
    store_nbr = prompt_int("Store number", *VALID_STORE_RANGE, default=44)
    family = prompt_product_family()
    pred_date = prompt_date("Prediction date", default="2017-08-15")

    # Automated Feature Retrieval
    print(f"\n{SUBDIVIDER}")
    print("  Step 2: Automated Feature Retrieval & Pipeline Construction")
    print(SUBDIVIDER)
    built = build_auto_features(store_nbr, family, pred_date, manager)
    feat = built["features"]

    for note in built["notes"]:
        print(f"  * {note}")

    # Build warning-free DataFrame formatted specifically for logistic pipeline (all cat columns as str)
    cat_cols_logistic = ["family", "store_nbr", "type", "cluster", "dayofweek"]
    num_cols_logistic = [c for c in feats_req if c not in cat_cols_logistic]

    df_row = pd.DataFrame([{
        **{k: float(feat[k]) for k in num_cols_logistic},
        "family": str(feat["family"]),
        "store_nbr": str(feat["store_nbr"]),
        "type": str(feat["type"]),
        "cluster": str(feat["cluster"]),
        "dayofweek": str(feat["dayofweek"]),
    }])

    # Model Inference
    prob = float(pipeline.predict_proba(df_row[feats_req])[0, 1])
    is_surge = int(prob >= optimal_threshold)

    # Surge volume threshold in units from training baseline
    sf_threshold_units = manager.threshold_map.get((store_nbr, family), None)

    # Display Results
    print(f"\n{DIVIDER}")
    print(f"  CLASSIFICATION RESULT - Store {store_nbr} | {family}")
    print(f"  Date: {pred_date.strftime('%Y-%m-%d')} ({pred_date.strftime('%A')})")
    print(DIVIDER)
    print(f"  >> Surge Probability P(Surge):  {prob:>10.4f}  ({prob*100:.1f}%)")
    print(f"  >> Calibrated Decision Cutoff: {optimal_threshold:>10.4f}  (Tuned for Stockout Mitigation)")

    if is_surge == 1:
        print("\n  [ALERT] CLASSIFICATION: *** DEMAND SURGE DETECTED ***")
        print("          Sales expected to spike significantly above normal volume.")
        print("          Recommended Action: Proactively initiate warehouse inventory replenishment")
        print("          and increase shelf restocking frequency to prevent costly stockouts.")
    else:
        print("\n  [OK]    CLASSIFICATION: Normal Demand Pattern")
        print("          Sales projected within standard operating levels. Standard replenishment cycle.")

    if sf_threshold_units is not None:
        print(f"\n  Store-Family 80th-Percentile Benchmark:")
        print(f"    * Surge Volume Cutoff:       {sf_threshold_units:>14,.2f} units")

    # Ground truth validation if historical
    if feat.get("actual_sales") is not None and sf_threshold_units is not None:
        actual = feat["actual_sales"]
        actual_surge = int(actual >= sf_threshold_units)
        is_correct = (is_surge == actual_surge)
        print(f"\n  Ground Truth Holdout Comparison:")
        print(f"    * Actual Observed Sales:     {actual:>14,.2f} units")
        print(f"    * Actual Surge Status:       {'DEMAND SURGE' if actual_surge else 'Normal Demand'}")
        print(f"    * Classifier Decision:       {'CORRECT [True Match]' if is_correct else 'DISCREPANCY'}")


# ===========================================================================
# Phase 3: Time Series Econometric Forecasting
# ===========================================================================

def predict_time_series(cached_data: dict) -> None:
    print_header("PHASE 3 - Time Series Econometric Forecasting")
    print("  Models: Seasonal Naive, Holt-Winters Exponential Smoothing, SARIMA, SARIMAX.")
    print("  Historical series and forward exogenous drivers are automatically retrieved.")

    manager = TerminalDataManager.get_instance()

    # 1. Select Scope
    print(f"\n{SUBDIVIDER}")
    print("  Step 1: Select Forecasting Scope")
    print(SUBDIVIDER)
    scope = prompt_choice("Select forecast scope", [
        "Enterprise Aggregated (Total national network across all 54 stores)",
        "Store-Family Micro Series (Specific store and product family)",
    ])
    is_aggregated = "Enterprise" in scope

    store_nbr, family = None, None
    if not is_aggregated:
        store_nbr = prompt_int("Store number", *VALID_STORE_RANGE, default=44)
        family = prompt_product_family()

    # 2. Select Forecast Horizon
    print(f"\n{SUBDIVIDER}")
    print("  Step 2: Forecast Horizon & Model Selection")
    print(SUBDIVIDER)
    horizon = prompt_int("Forecast horizon (number of days forward)", lo=1, hi=60, default=15)

    model_choice = prompt_choice("Select forecasting model", [
        "Seasonal Naive (Weekly persistence baseline)",
        "Holt-Winters (Additive level + trend + weekly seasonality)",
        "SARIMA (Seasonal ARIMA (1,1,1)x(1,1,0)_7 on log-sales)",
        "SARIMAX (SARIMA with forward promotions, holidays, paydays, oil)",
    ])
    model_name = model_choice.split(" ")[0]

    # 3. Retrieve Historical Series Automatically
    print(f"\n{SUBDIVIDER}")
    print("  Step 3: Automated Historical Series Retrieval & Preprocessing")
    print(SUBDIVIDER)

    full_dates = pd.date_range(HISTORICAL_TRAIN_START, HISTORICAL_SALES_END, freq="D")

    if is_aggregated:
        print("  [Auto-retrieved] Aggregating national daily sales across all stores & families...")
        agg = manager.full_hist.groupby("date").agg({
            "sales": "sum",
            "onpromotion": "sum",
            "dcoilwtico": "first",
            "is_payday": "first",
            "is_holiday": "max",
        }).reindex(full_dates)
        agg["sales"] = agg["sales"].fillna(0.0)
        agg["onpromotion"] = agg["onpromotion"].fillna(0.0)
        agg["dcoilwtico"] = agg["dcoilwtico"].interpolate().bfill().ffill()
        agg["is_payday"] = ((full_dates.day == 15) | (full_dates.is_month_end)).astype(float)
        agg["is_holiday"] = agg["is_holiday"].fillna(0.0)
        series_df = agg
        label = "Enterprise Aggregated (National)"
    else:
        print(f"  [Auto-retrieved] Retrieving daily sales for Store {store_nbr} - {family}...")
        sub = manager.full_hist[(manager.full_hist["store_nbr"] == store_nbr) & (manager.full_hist["family"] == family)].copy()
        sub = sub.drop_duplicates("date").set_index("date").reindex(full_dates)
        sub["sales"] = sub["sales"].fillna(0.0)
        sub["onpromotion"] = sub["onpromotion"].fillna(0.0)
        sub["dcoilwtico"] = sub["dcoilwtico"].interpolate().bfill().ffill()
        sub["is_payday"] = ((full_dates.day == 15) | (full_dates.is_month_end)).astype(float)
        sub["is_holiday"] = sub["is_holiday"].fillna(0.0)
        series_df = sub
        label = f"Store {store_nbr} - {family}"

    # Use standard 1-year training window
    train_series = series_df.loc["2016-08-01":"2017-07-31"]["sales"]
    forecast_start = pd.to_datetime("2017-08-01")
    target_dates = pd.date_range(forecast_start, periods=horizon, freq="D")

    print(f"  Historical series length: {len(train_series)} days (2016-08-01 to 2017-07-31)")
    print(f"  Forecast target window:   {horizon} days ({target_dates[0].strftime('%Y-%m-%d')} to {target_dates[-1].strftime('%Y-%m-%d')})")

    # 4. Exogenous variables for SARIMAX
    exog_future: Optional[pd.DataFrame] = None
    if "SARIMAX" in model_choice:
        print("  [Auto-retrieved] Preparing forward exogenous covariates (promotions, oil, paydays, holidays)...")
        exog_cols = ["onpromotion", "is_payday", "is_holiday", "dcoilwtico"]
        exog_train = series_df.loc["2016-08-01":"2017-07-31"][exog_cols].copy()

        # Build future exogenous frame automatically
        promo_future: list[float] = []
        oil_future: list[float] = []
        holiday_future: list[float] = []
        payday_future: list[float] = []

        last_known_oil = float(manager.oil_series.iloc[-1])

        for d in target_dates:
            # Payday
            payday_future.append(float(d.day == 15 or d.is_month_end))
            # Holiday
            holiday_future.append(float(d in manager.holidays_dict))
            # Oil
            oil_val = manager.oil_series.get(d, last_known_oil)
            oil_future.append(float(oil_val))
            # Promotion
            if is_aggregated:
                # Sum promotions from full_hist or test
                p = manager.full_hist[manager.full_hist["date"] == d]["onpromotion"].sum()
                promo_future.append(float(p))
            else:
                p_val = manager.test_promo_lookup.get((store_nbr, family, d), None)
                if p_val is None:
                    # check historical val
                    if (store_nbr, family, d) in manager.hist_lookup.index:
                        p_val = manager.hist_lookup.loc[(store_nbr, family, d)]["onpromotion"]
                    else:
                        p_val = 0.0
                promo_future.append(float(p_val))

        exog_future = pd.DataFrame({
            "onpromotion": promo_future,
            "is_payday": payday_future,
            "is_holiday": holiday_future,
            "dcoilwtico": oil_future,
        }, index=target_dates)

    # 5. Fit & Forecast
    from src.time_series import (
        HoltWintersForecaster,
        SARIMAForecaster,
        SARIMAXForecaster,
        SeasonalNaiveForecaster,
    )

    print(f"\n  Fitting {model_name} model and generating {horizon}-day forecast...")
    preds: np.ndarray

    try:
        if "Seasonal" in model_choice:
            forecaster = SeasonalNaiveForecaster(season_period=7)
            forecaster.fit(train_series)
            preds = forecaster.forecast(steps=horizon)

        elif "Holt-Winters" in model_choice:
            forecaster = HoltWintersForecaster(season_period=7, trend="add", seasonal="add")
            forecaster.fit(train_series)
            preds = forecaster.forecast(steps=horizon)

        elif "SARIMAX" in model_choice:
            forecaster = SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 0, 7), use_log=False)
            forecaster.fit(train_series, exog=exog_train)
            preds = forecaster.forecast(steps=horizon, exog_future=exog_future)

        elif "SARIMA" in model_choice:
            forecaster = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 0, 7), use_log=True)
            forecaster.fit(train_series)
            preds = forecaster.forecast(steps=horizon)

        else:
            print("  [X] Unknown model choice.")
            return

    except Exception as e:
        print(f"\n  [X] Model fitting/forecasting error: {e}")
        return

    # 6. Display Forecast Trajectory
    preds = np.clip(preds, 0, None)
    print(f"\n{DIVIDER}")
    print(f"  FORECAST RESULTS - {label}")
    print(f"  Model:   {model_choice}")
    print(f"  Horizon: {horizon} Days ({target_dates[0].strftime('%Y-%m-%d')} to {target_dates[-1].strftime('%Y-%m-%d')})")
    print(DIVIDER)
    print(f"  {'Date':<12} {'Day':<10} {'Forecast Sales (Units)':>22}  {'Rounded':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*22}  {'-'*12}")

    for i, (d, val) in enumerate(zip(target_dates, preds)):
        day_name = d.strftime("%A")
        print(f"  {d.strftime('%Y-%m-%d'):<12} {day_name:<10} {val:>22,.2f}  {round(val):>12,}")

    total_units = float(np.sum(preds))
    avg_units = float(np.mean(preds))
    print(DIVIDER)
    print(f"  Summary over {horizon} days:")
    print(f"    * Cumulative Projected Volume: {total_units:>14,.2f} units")
    print(f"    * Daily Average Volume:        {avg_units:>14,.2f} units/day")


# ===========================================================================
# Main Interactive Menu Loop
# ===========================================================================

def main() -> None:
    print("\n" + DIVIDER)
    print("  Retail Sales Prediction and Forecasting - Terminal System")
    print("  Dataset: Corporacion Favorita (Kaggle Store Sales)")
    print("  Automated Feature Retrieval Engine Enabled")
    print(DIVIDER)

    cached_data: dict = {}

    while True:
        print("""
  MAIN MENU
  ---------
    1. Linear Regression    - Predict Daily Sales
    2. Logistic Regression  - Predict Demand Surge
    3. Time Series          - Forecast Future Sales
    4. Exit
""")
        choice = input("  Select option [1-4]: ").strip()

        if choice == "1":
            try:
                predict_linear_regression(cached_data)
            except KeyboardInterrupt:
                print("\n  [Interrupted - returning to main menu]")
            except Exception as e:
                print(f"\n  [X] Error during prediction: {e}")
                import traceback; traceback.print_exc()

        elif choice == "2":
            try:
                predict_logistic_regression(cached_data)
            except KeyboardInterrupt:
                print("\n  [Interrupted - returning to main menu]")
            except Exception as e:
                print(f"\n  [X] Error during prediction: {e}")
                import traceback; traceback.print_exc()

        elif choice == "3":
            try:
                predict_time_series(cached_data)
            except KeyboardInterrupt:
                print("\n  [Interrupted - returning to main menu]")
            except Exception as e:
                print(f"\n  [X] Error during forecasting: {e}")
                import traceback; traceback.print_exc()

        elif choice == "4":
            print("\n  Exiting Retail Sales Terminal System. Goodbye!\n")
            sys.exit(0)

        else:
            print("  [X] Invalid choice. Please enter 1, 2, 3, or 4.")

        input("\n  Press Enter to return to the main menu ...")


if __name__ == "__main__":
    main()
