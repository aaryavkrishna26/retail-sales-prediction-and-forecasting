"""
Linear Regression models (DummyRegressor, Seasonal Naive, OLS, Ridge, Lasso, Elastic Net)
for continuous sales prediction.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_SEED, TARGET_COL
from src.evaluate import evaluate_regression

logger = logging.getLogger(__name__)

# Enhanced feature specifications for retail sales predictive modeling
NUMERIC_FEATURES = [
    "hist_sf_mean_log",
    "hist_sf_dow_mean_log",
    "sales_lag_16",
    "sales_lag_21",
    "sales_lag_28",
    "sales_lag_35",
    "sales_rolling_mean_7",
    "sales_rolling_mean_14",
    "sales_rolling_mean_30",
    "sales_rolling_mean_60",
    "sales_rolling_std_30",
    "onpromotion",
    "has_promotion",
    "log_promotion",
    "dcoilwtico",
    "is_weekend",
    "is_payday",
    "is_holiday",
]

CATEGORICAL_FEATURES = [
    "family",
    "type",
    "cluster",
    "dayofweek",
]


class SeasonalNaiveRegressor(BaseEstimator, RegressorMixin):
    """
    Seasonal Naive Forecaster: Predicts sales based strictly on the last
    observed sales value for the same store, product family, and day of week
    from the training period on or before cutoff_date (respecting the forecast horizon origin).
    """
    def __init__(
        self,
        key_cols=("store_nbr", "family", "dayofweek"),
        target_col=TARGET_COL,
        cutoff_date: Optional[str] = None,
    ):
        self.key_cols = list(key_cols)
        self.target_col = target_col
        self.cutoff_date = cutoff_date
        self.lookup_ = None
        self.global_mean_ = 0.0

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if y is not None:
            df[self.target_col] = y
        self.global_mean_ = float(df[self.target_col].mean())
        if "date" in df.columns:
            if self.cutoff_date is not None:
                cutoff = pd.to_datetime(self.cutoff_date)
                df = df[df["date"] <= cutoff]
            df_sorted = df.sort_values(by="date")
        else:
            df_sorted = df
        self.lookup_ = df_sorted.groupby(self.key_cols, observed=False)[self.target_col].last().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = list(zip(*(X[col] for col in self.key_cols)))
        preds = [self.lookup_.get(k, self.global_mean_) for k in keys]
        return np.array(preds, dtype=np.float32)


def build_linear_regression_pipeline(
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
    model_type: str = "ridge",
    alpha: float = 10.0,
    l1_ratio: float = 0.5,
    cutoff_date: Optional[str] = None,
) -> Any:
    """
    Construct an end-to-end scikit-learn pipeline for linear regression modeling.
    Features are scaled and one-hot encoded using memory-efficient sparse matrices.
    
    Supported model types:
        - 'dummy': Baseline predictor using empirical mean
        - 'snaive': Seasonal Naive forecaster (repeats last weekly cycle per store-family)
        - 'ols': Ordinary Least Squares (unregularized)
        - 'ridge': L2 regularized regression
        - 'lasso': L1 regularized sparse regression
        - 'elasticnet': Combined L1 and L2 regularized regression
    """
    model_key = model_type.lower()
    
    if model_key == "dummy":
        return Pipeline(steps=[("regressor", DummyRegressor(strategy="mean"))])
    
    if model_key in ["snaive", "seasonal_naive"]:
        return SeasonalNaiveRegressor(cutoff_date=cutoff_date)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True),
                categorical_features,
            ),
        ]
    )

    if model_key == "ols":
        regressor = LinearRegression()
    elif model_key == "ridge":
        regressor = Ridge(alpha=alpha, random_state=RANDOM_SEED)
    elif model_key == "lasso":
        regressor = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=2000, tol=1e-3)
    elif model_key in ["elasticnet", "elastic_net"]:
        regressor = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=RANDOM_SEED,
            max_iter=2000,
            tol=1e-3,
        )
    elif model_key in ["decision_tree", "dt"]:
        regressor = DecisionTreeRegressor(max_depth=12, min_samples_leaf=50, random_state=RANDOM_SEED)
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Choose from 'dummy', 'snaive', 'ols', 'ridge', 'lasso', 'elasticnet', 'decision_tree'."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    return pipeline


def train_and_evaluate_model(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_name: str,
    use_log_target: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Fit pipeline on training data and evaluate on validation set.
    Handles non-negative constraint explicitly and tracks negative prediction statistics.
    """
    logger.info("Fitting %s...", model_name)
    
    # Seasonal naive directly predicts physical sales scale
    if isinstance(pipeline, SeasonalNaiveRegressor):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        metrics = evaluate_regression(y_val, y_pred)
        metrics["Model"] = model_name
        metrics["Negative_Predictions_Count"] = 0
        metrics["Negative_Predictions_Pct"] = 0.0
        metrics["Pred_Min"] = round(float(y_pred.min()), 2)
        metrics["Pred_Max"] = round(float(y_pred.max()), 2)
        logger.info("%s Metrics: %s", model_name, metrics)
        return metrics, y_pred

    if use_log_target:
        y_train_fit = np.log1p(np.clip(y_train, a_min=0, a_max=None))
    else:
        y_train_fit = y_train

    pipeline.fit(X_train, y_train_fit)
    preds_raw = pipeline.predict(X_val)

    if use_log_target:
        preds_untransformed = np.expm1(preds_raw)
        neg_count = int((preds_untransformed < 0).sum())
        neg_pct = float((preds_untransformed < 0).mean() * 100)
        y_pred = np.clip(preds_untransformed, a_min=0, a_max=None)
    else:
        neg_count = int((preds_raw < 0).sum())
        neg_pct = float((preds_raw < 0).mean() * 100)
        y_pred = np.clip(preds_raw, a_min=0, a_max=None)

    metrics = evaluate_regression(y_val, y_pred)
    metrics["Model"] = model_name
    metrics["Negative_Predictions_Count"] = neg_count
    metrics["Negative_Predictions_Pct"] = round(neg_pct, 2)
    metrics["Pred_Min"] = round(float(y_pred.min()), 2)
    metrics["Pred_Max"] = round(float(y_pred.max()), 2)

    logger.info("%s Metrics: %s", model_name, metrics)
    return metrics, y_pred


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    val_dates: Optional[pd.Series] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate diagnostic Actual vs Predicted scatter and time series plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Scatter plot (sample 5,000 points if large)
    n_samples = min(5000, len(y_true))
    idx = np.random.RandomState(RANDOM_SEED).choice(len(y_true), n_samples, replace=False)
    
    max_val = max(np.percentile(y_true, 99.5), np.percentile(y_pred, 99.5))
    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.3, color="#1f77b4", edgecolor="none", s=18)
    axes[0].plot([0, max_val], [0, max_val], color="#d62728", linestyle="--", linewidth=1.8, label="Ideal Fit (y = x)")
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_title(f"{model_name}: Actual vs Predicted Sales", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual Sales", fontsize=11)
    axes[0].set_ylabel("Predicted Sales", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Aggregated Daily Sales Timeline
    if val_dates is not None:
        plot_df = pd.DataFrame({"date": val_dates, "Actual": y_true, "Predicted": y_pred})
        daily_agg = plot_df.groupby("date").sum().reset_index()
        axes[1].plot(daily_agg["date"], daily_agg["Actual"], marker="o", color="#2ca02c", linewidth=2.0, label="Actual Total Sales")
        axes[1].plot(daily_agg["date"], daily_agg["Predicted"], marker="s", color="#1f77b4", linestyle="--", linewidth=2.0, label=f"Predicted ({model_name})")
        axes[1].set_title("Aggregated Daily Network Sales (Aug 1–15, 2017)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Date", fontsize=11)
        axes[1].set_ylabel("Total Units Sold", fontsize=11)
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved plot to %s", save_path)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate residual distribution and residual vs predicted diagnostic plot.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Residual Histogram
    p01, p99 = np.percentile(residuals, [1, 99])
    filtered_res = residuals[(residuals >= p01) & (residuals <= p99)]
    sns.histplot(filtered_res, kde=True, ax=axes[0], color="#9467bd", bins=40)
    axes[0].axvline(0, color="#d62728", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"{model_name}: Residuals Distribution (1st-99th percentile)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Residual (Actual - Predicted)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Residual vs Predicted
    n_samples = min(5000, len(y_true))
    idx = np.random.RandomState(RANDOM_SEED).choice(len(y_true), n_samples, replace=False)
    axes[1].scatter(y_pred[idx], residuals[idx], alpha=0.3, color="#ff7f0e", s=18)
    axes[1].axhline(0, color="#d62728", linestyle="--", linewidth=1.5)
    axes[1].set_xlim(0, np.percentile(y_pred, 99.5))
    axes[1].set_ylim(p01, p99)
    axes[1].set_title(f"{model_name}: Residuals vs Predicted", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Sales", fontsize=11)
    axes[1].set_ylabel("Residual", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved residual plot to %s", save_path)
    plt.close(fig)


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate comparative bar plots across evaluated models for RMSLE, RMSE, MAE, and R^2.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    palette = sns.color_palette("muted", len(comparison_df))

    # 1. RMSLE
    sns.barplot(data=comparison_df, x="Model", y="RMSLE", hue="Model", legend=False, ax=axes[0], palette=palette)
    axes[0].set_title("Validation RMSLE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. RMSE
    sns.barplot(data=comparison_df, x="Model", y="RMSE", hue="Model", legend=False, ax=axes[1], palette=palette)
    axes[1].set_title("Validation RMSE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. MAE
    sns.barplot(data=comparison_df, x="Model", y="MAE", hue="Model", legend=False, ax=axes[2], palette=palette)
    axes[2].set_title("Validation MAE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    # 4. R-squared
    sns.barplot(data=comparison_df, x="Model", y="R2", hue="Model", legend=False, ax=axes[3], palette=palette)
    axes[3].set_title("Validation R² (Higher is Better)", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("")
    axes[3].tick_params(axis="x", rotation=35)
    axes[3].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved comparison bar plot to %s", save_path)
    plt.close(fig)


def plot_feature_importance_coefficients(
    pipeline: Pipeline,
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
    save_path: Optional[Path] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract and plot top positive and negative regression coefficients.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    # Extract transformed feature names
    cat_ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    coefs = regressor.coef_
    coef_df = pd.DataFrame({"Feature": all_feature_names, "Coefficient": coefs})
    coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()

    # Plot top positive and negative
    top_pos = coef_df.sort_values(by="Coefficient", ascending=False).head(top_n)
    top_neg = coef_df.sort_values(by="Coefficient", ascending=True).head(top_n)
    plot_df = pd.concat([top_pos, top_neg]).drop_duplicates().sort_values(by="Coefficient")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in plot_df["Coefficient"]]
    ax.barh(plot_df["Feature"], plot_df["Coefficient"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Top Linear Regression Standardized Coefficients", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient Magnitude (Log Sales Scale)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved coefficients plot to %s", save_path)
    plt.close(fig)

    return coef_df
