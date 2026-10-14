"""
Logistic Regression models for high-demand surge and promotional lift classification.
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.config import MODELS_DIR, RANDOM_SEED, SURGE_TARGET_COL
from src.evaluate import evaluate_classification

logger = logging.getLogger(__name__)


def build_logistic_regression_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
) -> Pipeline:
    """
    Construct an end-to-end scikit-learn pipeline for logistic classification.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_features),
        ]
    )

    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        max_iter=1000,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_and_evaluate_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_col: str = SURGE_TARGET_COL,
    C: float = 1.0,
    threshold: float = 0.5,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a Logistic Regression classifier on train_df and evaluate on val_df.
    Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    feature_cols = numeric_features + categorical_features
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values

    pipeline = build_logistic_regression_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        C=C,
    )

    logger.info("Fitting Logistic Regression classification pipeline...")
    pipeline.fit(X_train, y_train)

    # Probabilities and thresholded decision labels
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred_labels = (y_pred_proba >= threshold).astype(int)

    metrics = evaluate_classification(y_val, y_pred_labels, y_pred_proba)
    logger.info("Logistic Regression Validation Metrics (Threshold=%.2f): %s", threshold, metrics)

    # Save model artifact
    model_path = MODELS_DIR / "logistic_regression_classifier.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Saved trained classifier to %s", model_path)

    return pipeline, metrics
