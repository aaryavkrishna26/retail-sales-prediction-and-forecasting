"""
Model evaluation metrics for both regression and classification pipelines.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE):
        RMSLE = sqrt( mean( (log1p(pred) - log1p(true))^2 ) )
    Predictions are clipped at 0 to avoid invalid logarithm values.
    """
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    y_true_clipped = np.clip(y_true, a_min=0, a_max=None)
    log_true = np.log1p(y_true_clipped)
    log_pred = np.log1p(y_pred_clipped)
    return float(np.sqrt(mean_squared_error(log_true, log_pred)))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics: RMSLE, RMSE, MAE, and R^2.
    """
    y_pred_clean = np.clip(y_pred, a_min=0, a_max=None)
    rmsle = calculate_rmsle(y_true, y_pred_clean)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clean)))
    mae = float(mean_absolute_error(y_true, y_pred_clean))
    r2 = float(r2_score(y_true, y_pred_clean))

    return {
        "RMSLE": round(rmsle, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred_labels: np.ndarray,
    y_pred_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred_labels)), 4),
        "Precision": round(float(precision_score(y_true, y_pred_labels, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred_labels, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred_labels, zero_division=0)), 4),
    }

    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics["ROC_AUC"] = round(float(roc_auc_score(y_true, y_pred_proba)), 4)
        metrics["PR_AUC"] = round(float(average_precision_score(y_true, y_pred_proba)), 4)
    else:
        metrics["ROC_AUC"] = float("nan")
        metrics["PR_AUC"] = float("nan")

    return metrics


def metrics_to_markdown_row(model_name: str, task: str, metrics: Dict[str, float]) -> str:
    """Format evaluation metrics as a GitHub-flavored Markdown table row."""
    formatted_vals = " | ".join(f"{k}: {v}" for k, v in metrics.items())
    return f"| **{model_name}** | {task} | {formatted_vals} |"
