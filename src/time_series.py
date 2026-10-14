"""
Time Series statistical models and baseline forecasting pipelines.
"""

from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluate import evaluate_regression

logger = logging.getLogger(__name__)


def generate_naive_seasonal_baseline(
    series: pd.Series,
    horizon: int = 16,
    season_period: int = 7,
) -> pd.Series:
    """
    Generate naive seasonal forecast using the previous season_period observations.
    """
    history = series.values
    forecast = []
    for step in range(horizon):
        lag_idx = -season_period + (step % season_period)
        forecast.append(history[lag_idx])
    return pd.Series(forecast)


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
    logger.info("Fitting SARIMAX%s x %s model...", order, seasonal_order)
    try:
        model = SARIMAX(
            endog=train_series,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=steps, exog=exog_forecast)
        forecast_clipped = np.clip(np.array(forecast), a_min=0, a_max=None)
        return forecast_clipped, fitted
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
