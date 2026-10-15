"""
Logistic Regression models for high-demand surge classification and stockout prevention.
Implements:
1. DummyClassifier (Most Frequent / Majority Class Baseline)
2. Logistic Regression (Default Class Weighting)
3. Logistic Regression (Balanced Class Weighting)
4. Training-derived Decision Threshold Optimization (Zero Leakage)
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import joblib

from src.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_SEED, SURGE_TARGET_COL
from src.evaluate import evaluate_classification

logger = logging.getLogger(__name__)

# Feature specifications for Logistic Classification
LOGISTIC_NUMERIC_FEATURES = [
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
    "day",
    "month",
    "year",
]

LOGISTIC_CATEGORICAL_FEATURES = [
    "family",
    "store_nbr",
    "type",
    "cluster",
    "dayofweek",
]


def build_logistic_regression_pipeline(
    numeric_features: List[str] = LOGISTIC_NUMERIC_FEATURES,
    categorical_features: List[str] = LOGISTIC_CATEGORICAL_FEATURES,
    model_type: str = "default",
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """
    Construct an end-to-end scikit-learn pipeline for logistic classification.
    Supports 'dummy', 'default', and 'balanced' configurations.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True),
                categorical_features,
            ),
        ]
    )

    if model_type == "dummy":
        classifier = DummyClassifier(strategy="most_frequent", random_state=random_state)
    elif model_type == "balanced" or class_weight == "balanced":
        classifier = LogisticRegression(
            C=C,
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
        )
    elif model_type in ["decision_tree", "dt"]:
        classifier = DecisionTreeClassifier(
            max_depth=10,
            class_weight=class_weight or "balanced",
            random_state=random_state,
        )
    else:
        classifier = LogisticRegression(
            C=C,
            class_weight=None,
            random_state=random_state,
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


def find_optimal_threshold_on_train(
    pipeline: Optional[Pipeline] = None,
    X_train: pd.DataFrame = None,
    y_train: np.ndarray = None,
    train_dates: pd.Series = None,
    cutoff_date: str = "2017-06-01",
    numeric_features: List[str] = LOGISTIC_NUMERIC_FEATURES,
    categorical_features: List[str] = LOGISTIC_CATEGORICAL_FEATURES,
    model_type: str = "default",
    class_weight: Optional[str] = None,
    C: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """
    Derive an optimal probability decision threshold strictly using out-of-sample calibration.
    
    Calibration Protocol (Zero In-Sample Leakage):
    - Splits training data chronologically into:
      1. Sub-train set: observations strictly before cutoff_date (e.g. 2016-08-01 to 2017-05-31: 539,946 rows).
      2. Calibration holdout set: observations on or after cutoff_date (e.g. 2017-06-01 to 2017-07-31: 108,702 rows).
    - A dedicated calibration model is fitted ONLY on the sub-train set.
    - Probabilities are predicted on the calibration holdout set (strictly out-of-sample; the calibration model has never seen these rows).
    - The decision threshold maximizing F1-score is derived from these out-of-sample probabilities.
    - Zero validation observations are touched.
    """
    calib_mask = (train_dates >= cutoff_date).values
    fit_mask = ~calib_mask

    if not fit_mask.any() or not calib_mask.any():
        logger.warning("Invalid calibration split mask; defaulting threshold to 0.5.")
        return 0.5, 0.0

    X_subtrain = X_train[fit_mask].copy()
    y_subtrain = y_train[fit_mask]
    X_calib = X_train[calib_mask].copy()
    y_calib = y_train[calib_mask]

    logger.info(
        "Fitting out-of-sample calibration model on %d rows; evaluating threshold on %d holdout rows...",
        len(X_subtrain),
        len(X_calib),
    )

    calib_pipeline = build_logistic_regression_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
    )
    calib_pipeline.fit(X_subtrain, y_subtrain)

    if hasattr(calib_pipeline.named_steps["classifier"], "predict_proba"):
        y_calib_proba = calib_pipeline.predict_proba(X_calib)[:, 1]
    else:
        return 0.5, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_calib, y_calib_proba)
    numerator = 2 * (precisions * recalls)
    denominator = np.maximum(precisions + recalls, 1e-8)
    f1_scores = numerator / denominator

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    best_f1 = float(f1_scores[best_idx])

    logger.info(
        "Out-of-sample threshold optimization complete. Optimal threshold: %.4f (Calibration F1: %.4f)",
        best_threshold,
        best_f1,
    )
    return best_threshold, best_f1


def train_and_evaluate_classifier(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Train classification pipeline and evaluate performance on validation set.
    Returns metrics dictionary, predicted probabilities, and predicted binary labels.
    """
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 2)

    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "predict_proba"):
        y_proba = pipeline.predict_proba(X_val)[:, 1]
    else:
        # DummyClassifier predicting majority class (class 0)
        y_proba = np.zeros(len(y_val), dtype=float)

    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, zero_division=0))
    rec = float(recall_score(y_val, y_pred, zero_division=0))
    f1 = float(f1_score(y_val, y_pred, zero_division=0))

    if len(np.unique(y_val)) > 1 and hasattr(clf, "predict_proba"):
        roc_auc = float(roc_auc_score(y_val, y_proba))
        pr_auc = float(average_precision_score(y_val, y_proba))
    else:
        roc_auc = 0.5
        pr_auc = float(y_val.mean())

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Threshold": round(threshold, 4),
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
        "Fit_Time_Sec": fit_time,
    }

    return metrics, y_proba, y_pred


def plot_confusion_matrices(
    cm_dict: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Render and save side-by-side confusion matrix heatmaps.
    """
    num_models = len(cm_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(5.5 * num_models, 4.5), sharey=True)
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Normal (0)", "Surge (1)"],
            yticklabels=["Normal (0)", "Surge (1)"],
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix plots to %s", output_path)


def plot_roc_and_pr_curves(
    results_dict: Dict[str, Dict[str, Any]],
    y_val: np.ndarray,
    output_path: Path,
) -> None:
    """
    Generate ROC curves and Precision-Recall curves side by side.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]

    for (name, res), color in zip(results_dict.items(), colors):
        y_proba = res["proba"]
        roc_auc = res["metrics"]["ROC_AUC"]
        pr_auc = res["metrics"]["PR_AUC"]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=color, lw=2)

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_val, y_proba)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, lw=2)

    # ROC diagonal
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax_roc.grid(True, linestyle="--", alpha=0.5)
    ax_roc.legend(loc="lower right", fontsize=9)

    # PR baseline
    pos_rate = float(y_val.mean())
    ax_pr.axhline(pos_rate, color="k", linestyle="--", alpha=0.5, label=f"No-Skill Baseline ({pos_rate:.4f})")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax_pr.grid(True, linestyle="--", alpha=0.5)
    ax_pr.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC and PR curves to %s", output_path)


def plot_logistic_coefficients(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Extract and plot the top positive and negative feature coefficients (log-odds).
    """
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "coef_"):
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + encoded_cat_names

    coefficients = clf.coef_[0]
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
    coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()
    top_df = coef_df.sort_values(by="Abs_Coef", ascending=False).head(top_n).sort_values(by="Coefficient")

    plt.figure(figsize=(10, 8))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in top_df["Coefficient"]]
    plt.barh(top_df["Feature"], top_df["Coefficient"], color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
    plt.xlabel("Log-Odds Coefficient (Effect on Demand Surge Probability)", fontsize=11)
    plt.title(f"Top {top_n} Most Influential Features (Logistic Regression)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved logistic coefficient plot to %s", output_path)
