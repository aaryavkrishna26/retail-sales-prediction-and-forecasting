"""
Time Series statistical models, seasonal baselines, and econometric forecasting pipelines.
Provides modular forecasters for:
- Seasonal Naive (weekly persistence baseline)
- Holt-Winters Exponential Smoothing (level, trend, and weekly seasonality)
- SARIMA (Seasonal Autoregressive Integrated Moving Average)
- SARIMAX (State-space SARIMA with exogenous promotion, payday, holiday, and oil covariates)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults
from statsmodels.graphics.tsaplots import plot_acf

from src.evaluate import evaluate_regression, calculate_rmsle

logger = logging.getLogger(__name__)


class SeasonalNaiveForecaster:
    """
    Seasonal Naive forecaster repeating the last observed seasonal cycle.
    For daily retail sales, season_period=7 captures day-of-week persistent patterns.
    """

    def __init__(self, season_period: int = 7):
        self.season_period = season_period
        self.last_cycle_: Optional[np.ndarray] = None

    def fit(self, y: Union[pd.Series, np.ndarray]) -> "SeasonalNaiveForecaster":
        values = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
        if len(values) < self.season_period:
            raise ValueError(f"Series length ({len(values)}) must be >= season_period ({self.season_period}).")
        self.last_cycle_ = values[-self.season_period:]
        return self

    def forecast(self, steps: int = 15) -> np.ndarray:
        if self.last_cycle_ is None:
            raise ValueError("Forecaster must be fit before generating forecasts.")
        preds = [self.last_cycle_[i % self.season_period] for i in range(steps)]
        return np.clip(np.array(preds, dtype=float), a_min=0, a_max=None)


class MovingAverageForecaster:
    """
    Moving Average Forecaster: projects the trailing rolling window mean into the future horizon.
    """

    def __init__(self, window: int = 7):
        self.window = window
        self.mean_value_: Optional[float] = None

    def fit(self, y: Union[pd.Series, np.ndarray]) -> "MovingAverageForecaster":
        values = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
        if len(values) < self.window:
            self.mean_value_ = float(np.mean(values))
        else:
            self.mean_value_ = float(np.mean(values[-self.window:]))
        return self

    def forecast(self, steps: int = 15) -> np.ndarray:
        if self.mean_value_ is None:
            raise ValueError("Forecaster must be fit before generating forecasts.")
        return np.clip(np.full(steps, fill_value=self.mean_value_, dtype=float), a_min=0, a_max=None)


class HoltWintersForecaster:
    """
    Holt-Winters Exponential Smoothing forecaster with additive trend and seasonal components.
    """

    def __init__(
        self,
        season_period: int = 7,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool = False,
    ):
        self.season_period = season_period
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.model_: Optional[HoltWintersResults] = None

    def fit(self, y: pd.Series) -> "HoltWintersForecaster":
        clean_y = y.copy()
        if self.seasonal == "mul" and (clean_y <= 0).any():
            clean_y = clean_y.clip(lower=1.0)

        model = ExponentialSmoothing(
            clean_y,
            seasonal_periods=self.season_period,
            trend=self.trend,
            seasonal=self.seasonal,
            damped_trend=self.damped_trend,
            initialization_method="estimated",
        )
        self.model_ = model.fit(optimized=True)
        return self

    def forecast(self, steps: int = 15) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Forecaster must be fit before generating forecasts.")
        preds = self.model_.forecast(steps)
        return np.clip(np.asarray(preds, dtype=float), a_min=0, a_max=None)


class SARIMAForecaster:
    """
    Seasonal ARIMA forecaster $(p, d, q) \\times (P, D, Q)_s$.
    Supports optional logarithmic target transform $\\log(1 + y)$ to stabilize variance
    and guarantee physically non-negative projections.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
        use_log: bool = True,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_log = use_log
        self.model_: Optional[SARIMAXResults] = None

    def fit(self, y: pd.Series) -> "SARIMAForecaster":
        endog = np.log1p(np.clip(y, a_min=0, a_max=None)) if self.use_log else y
        model = SARIMAX(
            endog=endog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_ = model.fit(disp=False)
        return self

    def forecast(self, steps: int = 15) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Forecaster must be fit before generating forecasts.")
        raw_pred = self.model_.forecast(steps=steps)
        if self.use_log:
            pred = np.expm1(raw_pred)
        else:
            pred = raw_pred
        return np.clip(np.asarray(pred, dtype=float), a_min=0, a_max=None)


class SARIMAXForecaster:
    """
    State-Space Seasonal ARIMA with Exogenous Regressors (SARIMAX).
    Incorporates forward-known exogenous drivers across the forecast horizon:
    promotional counts, Ecuador paydays (15th and month-end), holidays, and oil price signals.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
        use_log: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_log = use_log
        self.model_: Optional[SARIMAXResults] = None
        self.feature_names_: List[str] = []

    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
    ) -> "SARIMAXForecaster":
        endog = np.log1p(np.clip(y, a_min=0, a_max=None)) if self.use_log else y
        if exog is not None:
            self.feature_names_ = list(exog.columns)
            exog_data = exog.astype(float)
        else:
            exog_data = None

        model = SARIMAX(
            endog=endog,
            exog=exog_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_ = model.fit(disp=False)
        return self

    def forecast(
        self,
        steps: int = 15,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Forecaster must be fit before generating forecasts.")
        exog_data = exog_future.astype(float) if exog_future is not None else None
        raw_pred = self.model_.forecast(steps=steps, exog=exog_data)
        if self.use_log:
            pred = np.expm1(raw_pred)
        else:
            pred = raw_pred
        return np.clip(np.asarray(pred, dtype=float), a_min=0, a_max=None)


# Backward-compatible convenience functions matching initial API
def generate_naive_seasonal_baseline(
    series: pd.Series,
    horizon: int = 16,
    season_period: int = 7,
) -> pd.Series:
    """
    Generate naive seasonal forecast using the previous season_period observations.
    """
    forecaster = SeasonalNaiveForecaster(season_period=season_period)
    forecaster.fit(series)
    return pd.Series(forecaster.forecast(steps=horizon))


def fit_sarimax_forecast(
    train_series: pd.Series,
    steps: int = 16,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
    exog_train: Optional[pd.DataFrame] = None,
    exog_forecast: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Fit a SARIMAX state-space model on a univariate time series and generate out-of-sample forecasts.
    """
    try:
        forecaster = SARIMAXForecaster(order=order, seasonal_order=seasonal_order, use_log=False)
        forecaster.fit(train_series, exog=exog_train)
        preds = forecaster.forecast(steps=steps, exog_future=exog_forecast)
        return preds, forecaster.model_
    except Exception as e:
        logger.error("SARIMAX fitting error: %s. Falling back to naive seasonal baseline.", e)
        fallback = generate_naive_seasonal_baseline(train_series, horizon=steps, season_period=7).values
        return fallback, None


def evaluate_time_series_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate time series forecast using standard regression metrics.
    """
    return evaluate_regression(y_true, y_pred)


# Diagnostic and comparative visualization functions
def plot_time_series_forecast(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    history_dates: Optional[pd.DatetimeIndex] = None,
    history_y: Optional[np.ndarray] = None,
    title: str = "Retail Sales Out-of-Sample Time Series Forecast",
    ylabel: str = "Sales Volume",
    output_path: Optional[Any] = None,
    save_path: Optional[Any] = None,
) -> None:
    """
    Generate high-contrast visualization comparing forecast trajectories against ground truth.
    """
    out_file = output_path or save_path
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Plot recent history if provided
    if history_dates is not None and history_y is not None:
        ax.plot(
            history_dates,
            history_y,
            label="Historical Sales (Train Window)",
            color="#2c3e50",
            linewidth=1.8,
            alpha=0.75,
        )

    # Ground truth validation observations
    ax.plot(
        dates,
        y_true,
        label="Actual Sales (Validation Holdout)",
        color="#111111",
        linewidth=2.5,
        marker="o",
        markersize=5,
    )

    # Model forecasts with distinct styling
    palette = {
        "Seasonal Naive": ("#7f8c8d", "--"),
        "Moving Average": ("#16a085", ":"),
        "Holt-Winters": ("#e67e22", "-."),
        "SARIMA (Univariate)": ("#2980b9", "-."),
        "SARIMAX (With Exogenous)": ("#27ae60", "-"),
    }

    for name, pred in predictions.items():
        color, linestyle = palette.get(name, ("#8e44ad", "-"))
        ax.plot(
            dates,
            pred,
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="s" if "SARIMAX" in name else None,
            markersize=4,
        )

    # Forecast origin line
    ax.axvline(x=dates[0], color="#c0392b", linestyle=":", linewidth=1.5, label="Forecast Origin (Cutoff)")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.xticks(rotation=25)
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        logger.info("Saved forecast visualization to %s", out_file)
    plt.close()


def plot_cohort_time_series_forecasts(
    cohort_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Any] = None,
    save_path: Optional[Any] = None,
) -> None:
    """
    Generate a multi-panel subplot displaying forecasts across representative store-family series.
    """
    out_file = output_path or save_path
    plt.style.use("seaborn-v0_8-whitegrid")
    n_series = len(cohort_results)
    fig, axes = plt.subplots(n_series, 1, figsize=(13, 3.2 * n_series), dpi=150, sharex=True)
    if n_series == 1:
        axes = [axes]

    colors = {
        "Actual": "#111111",
        "Seasonal Naive": "#7f8c8d",
        "Holt-Winters": "#e67e22",
        "SARIMA": "#2980b9",
        "SARIMAX": "#27ae60",
    }

    for idx, (series_name, data) in enumerate(cohort_results.items()):
        ax = axes[idx]
        dates = data["dates"]
        ax.plot(dates, data["y_true"], label="Actual", color=colors["Actual"], linewidth=2.2, marker="o", markersize=4)

        for model_key in ["Seasonal Naive", "Holt-Winters", "SARIMA", "SARIMAX"]:
            if model_key in data["predictions"]:
                style = "-" if model_key == "SARIMAX" else ("--" if model_key == "Seasonal Naive" else "-.")
                ax.plot(
                    dates,
                    data["predictions"][model_key],
                    label=model_key,
                    color=colors[model_key],
                    linestyle=style,
                    linewidth=1.7,
                )

        ax.set_title(f"Series: {series_name} — Out-of-Sample 15-Day Forecast", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sales Units", fontsize=9)
        if idx == 0:
            ax.legend(loc="upper left", ncol=5, frameon=True, fontsize=8)

    axes[-1].set_xlabel("Date", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=25)
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        logger.info("Saved cohort forecasts visualization to %s", out_file)
    plt.close()


def plot_time_series_comparison_bar(
    metrics_df: pd.DataFrame,
    output_path: Optional[Any] = None,
    save_path: Optional[Any] = None,
) -> None:
    """
    Generate grouped comparative bar chart for RMSLE, RMSE, and MAE across models.
    """
    out_file = output_path or save_path
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # Plot metrics
    for ax, metric in zip(axes, ["RMSLE", "RMSE", "MAE"]):
        hue_col = "Series_Level" if "Series_Level" in metrics_df.columns else None
        sns.barplot(
            data=metrics_df,
            x="Model",
            y=metric,
            hue=hue_col,
            palette="Set2" if hue_col else "Blues_d",
            ax=ax,
        )
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=10, fontweight="semibold")
        ax.tick_params(axis="x", rotation=30)
        if hue_col and ax != axes[0]:
            ax.get_legend().remove()

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        logger.info("Saved time series comparison bar chart to %s", out_file)
    plt.close()


def plot_time_series_residuals(
    residuals: np.ndarray,
    dates: pd.DatetimeIndex,
    model_name: str = "SARIMAX",
    output_path: Optional[Any] = None,
    save_path: Optional[Any] = None,
) -> None:
    """
    Generate residual diagnostics: Residuals over time, ACF correlogram, and histogram.
    """
    out_file = output_path or save_path
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)

    # 1. Residuals over time
    axes[0].plot(dates, residuals, marker="o", color="#2c3e50", linewidth=1.5)
    axes[0].axhline(0, color="#c0392b", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"{model_name}: Residuals Over Horizon", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Error ($y - \\hat{y}$)")
    axes[0].tick_params(axis="x", rotation=30)

    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2980b9", bins=10)
    axes[1].axvline(0, color="#c0392b", linestyle="--")
    axes[1].set_title("Residual Distribution & Density", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual")

    # 3. Autocorrelation of Residuals
    lags = min(len(residuals) // 2, 7)
    if lags >= 2:
        plot_acf(residuals, lags=lags, ax=axes[2], title="Autocorrelation Function (ACF)")
    else:
        axes[2].text(0.5, 0.5, "Insufficient sample size for ACF", ha="center", va="center")
        axes[2].set_title("ACF Diagnostics")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved residual diagnostics to %s", output_path)
    plt.close()
