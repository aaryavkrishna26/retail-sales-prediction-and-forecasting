"""
Data loading and ingestion utilities for the Kaggle Store Sales dataset.
"""

from typing import Dict, Optional, Tuple
import logging
import pandas as pd
from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    STORES_FILE,
    OIL_FILE,
    HOLIDAYS_FILE,
    TRANSACTIONS_FILE,
    RAW_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_raw_data_availability() -> Dict[str, bool]:
    """Check existence of all required raw CSV files in data/raw/."""
    expected_files = {
        "train": TRAIN_FILE,
        "test": TEST_FILE,
        "stores": STORES_FILE,
        "oil": OIL_FILE,
        "holidays_events": HOLIDAYS_FILE,
        "transactions": TRANSACTIONS_FILE,
    }
    status = {name: path.exists() for name, path in expected_files.items()}
    for name, exists in status.items():
        if exists:
            logger.info("Found raw data file: %s (%s)", name, expected_files[name].name)
        else:
            logger.warning("Missing raw data file: %s (Expected at %s)", name, expected_files[name])
    return status


def load_raw_datasets(nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Load raw Kaggle Store Sales datasets into memory with optimized types.
    
    Args:
        nrows: Optional limit on rows to read (useful for rapid testing / debugging).
        
    Returns:
        Dictionary containing loaded DataFrames for each available table.
    """
    datasets: Dict[str, pd.DataFrame] = {}

    if TRAIN_FILE.exists():
        logger.info("Loading train.csv...")
        datasets["train"] = pd.read_csv(
            TRAIN_FILE,
            parse_dates=["date"],
            nrows=nrows,
            dtype={"store_nbr": "int16", "family": "category", "sales": "float32", "onpromotion": "int32"},
        )

    if TEST_FILE.exists():
        logger.info("Loading test.csv...")
        datasets["test"] = pd.read_csv(
            TEST_FILE,
            parse_dates=["date"],
            nrows=nrows,
            dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"},
        )

    if STORES_FILE.exists():
        logger.info("Loading stores.csv...")
        datasets["stores"] = pd.read_csv(
            STORES_FILE,
            dtype={"store_nbr": "int16", "city": "category", "state": "category", "type": "category", "cluster": "int8"},
        )

    if OIL_FILE.exists():
        logger.info("Loading oil.csv...")
        datasets["oil"] = pd.read_csv(
            OIL_FILE,
            parse_dates=["date"],
            dtype={"dcoilwtico": "float32"},
        )

    if HOLIDAYS_FILE.exists():
        logger.info("Loading holidays_events.csv...")
        datasets["holidays"] = pd.read_csv(
            HOLIDAYS_FILE,
            parse_dates=["date"],
            dtype={
                "type": "category",
                "locale": "category",
                "locale_name": "category",
                "description": "string",
                "transferred": "bool",
            },
        )

    if TRANSACTIONS_FILE.exists():
        logger.info("Loading transactions.csv...")
        datasets["transactions"] = pd.read_csv(
            TRANSACTIONS_FILE,
            parse_dates=["date"],
            dtype={"store_nbr": "int16", "transactions": "int32"},
        )

    logger.info("Successfully loaded %d datasets.", len(datasets))
    return datasets
