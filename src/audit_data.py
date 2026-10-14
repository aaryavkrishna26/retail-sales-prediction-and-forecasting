"""
Dataset integrity verification and audit module.
Verifies shapes, columns, date ranges, missing values, and duplicate records.
"""

from typing import Dict, Any
import logging
import pandas as pd
from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    STORES_FILE,
    OIL_FILE,
    HOLIDAYS_FILE,
    TRANSACTIONS_FILE,
    SAMPLE_SUBMISSION_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def audit_single_dataframe(name: str, df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
    """Audit single DataFrame for shape, date range, nulls, and duplicates."""
    total_rows = len(df)
    null_counts = df.isnull().sum()
    null_summary = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    duplicate_rows = int(df.duplicated().sum())

    date_range = None
    if date_col in df.columns:
        date_series = pd.to_datetime(df[date_col])
        date_range = {
            "min_date": str(date_series.min().date()),
            "max_date": str(date_series.max().date()),
            "unique_dates": int(date_series.nunique()),
        }

    return {
        "dataset": name,
        "rows": total_rows,
        "columns_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "date_range": date_range,
        "missing_values": null_summary,
        "duplicate_rows": duplicate_rows,
    }


def run_full_dataset_audit() -> Dict[str, Any]:
    """Execute complete dataset audit across all raw CSV files."""
    files_to_audit = {
        "train": (TRAIN_FILE, "date"),
        "test": (TEST_FILE, "date"),
        "stores": (STORES_FILE, None),
        "oil": (OIL_FILE, "date"),
        "holidays_events": (HOLIDAYS_FILE, "date"),
        "transactions": (TRANSACTIONS_FILE, "date"),
        "sample_submission": (SAMPLE_SUBMISSION_FILE, None),
    }

    results = {}
    for name, (path, date_col) in files_to_audit.items():
        if not path.exists():
            logger.warning("File %s not found at %s", name, path)
            continue
        logger.info("Auditing %s...", name)
        df = pd.read_csv(path)
        audit_res = audit_single_dataframe(name, df, date_col=date_col)
        results[name] = audit_res

    return results


if __name__ == "__main__":
    audit_results = run_full_dataset_audit()
    for name, info in audit_results.items():
        print("=" * 60)
        print(f"Dataset: {name.upper()}")
        print(f"Shape: {info['rows']} rows x {info['columns_count']} columns")
        print(f"Columns: {info['columns']}")
        if info['date_range']:
            print(f"Date Range: {info['date_range']['min_date']} to {info['date_range']['max_date']} ({info['date_range']['unique_dates']} unique dates)")
        print(f"Missing Values: {info['missing_values'] if info['missing_values'] else 'None (0 missing)'}")
        print(f"Duplicate Rows: {info['duplicate_rows']}")
