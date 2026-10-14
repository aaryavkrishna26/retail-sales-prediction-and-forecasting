"""
Global configuration and directory paths for the retail sales predictive analytics project.
"""

from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models and Reports directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset file names (Kaggle Store Sales - Time Series Forecasting)
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
STORES_FILE = RAW_DATA_DIR / "stores.csv"
OIL_FILE = RAW_DATA_DIR / "oil.csv"
HOLIDAYS_FILE = RAW_DATA_DIR / "holidays_events.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"

# Global Random Seed for reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Temporal Train / Validation split cutoffs
VAL_START_DATE = "2017-08-01"
TEST_START_DATE = "2017-08-16"

# Key target column names
TARGET_COL = "sales"
SURGE_TARGET_COL = "is_high_sales"
