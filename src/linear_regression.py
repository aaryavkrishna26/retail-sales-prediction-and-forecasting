"""
Linear Regression models (OLS, Ridge, Lasso) for continuous sales prediction.
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.config import MODELS_DIR, RANDOM_SEED, TARGET_COL
from src.evaluate import evaluate_regression

logger = logging.getLogger(__name__)


def build_linear_regression_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    model_type: str = "ridge",
    alpha: float = 1.0,
) -> Pipeline:
    """
    Construct an end-to-end scikit-learn pipeline for linear regression modeling.
    Features are scaled and one-hot encoded appropriately.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_features),
        ]
    )

    if model_type == "ols":
        regressor = LinearRegression()
    elif model_type == "lasso":
        regressor = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=2000)
    elif model_type == "ridge":
        regressor = Ridge(alpha=alpha, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from 'ols', 'ridge', 'lasso'.")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    return pipeline


def train_and_evaluate_linear_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    model_type: str = "ridge",
    alpha: float = 1.0,
    use_log_target: bool = True,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a linear model on train_df and evaluate on val_df.
    If use_log_target is True, fits on log1p(sales) to directly minimize RMSLE and handles zero sales safely.
    """
    feature_cols = numeric_features + categorical_features
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values

    X_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET_COL].values

    if use_log_target:
        y_train_fit = np.log1p(np.clip(y_train, a_min=0, a_max=None))
    else:
        y_train_fit = y_train

    pipeline = build_linear_regression_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type,
        alpha=alpha,
    )

    logger.info("Fitting %s regression pipeline...", model_type.upper())
    pipeline.fit(X_train, y_train_fit)

    # Predict
    preds_raw = pipeline.predict(X_val)
    if use_log_target:
        y_pred = np.expm1(preds_raw)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
    else:
        y_pred = np.clip(preds_raw, a_min=0, a_max=None)

    metrics = evaluate_regression(y_val, y_pred)
    logger.info("%s Regression Validation Metrics: %s", model_type.upper(), metrics)

    # Save model artifact
    model_path = MODELS_DIR / f"{model_type}_regression_model.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Saved trained pipeline to %s", model_path)

    return pipeline, metrics
