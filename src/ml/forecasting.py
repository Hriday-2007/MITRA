"""
MITRA — Demand Forecasting Module
Simple but rigorous forecasting pipeline with time-based validation.

Methods compared:
1. Rolling Moving Average (baseline)
2. Exponential Smoothing (Holt-Winters via statsmodels)
3. Linear Regression with time + seasonality features
4. Gradient Boosting (scikit-learn) with feature engineering

Time-series train/test split: chronological (never random).
Metric: MAPE (Mean Absolute Percentage Error)

All forecasts are labelled as MODEL PROJECTIONS.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from config.settings import FORECASTING as FC_CFG

logger = get_logger("mitra.ml.forecasting")

warnings.filterwarnings("ignore")


def _make_weekly_series(df: pd.DataFrame, stock_code: Optional[str] = None) -> pd.Series:
    """Aggregate to weekly revenue. Optionally filter to a single product."""
    fdf = df.copy()
    if stock_code:
        fdf = fdf[fdf["StockCode"] == stock_code]

    weekly = (
        fdf.assign(week=fdf["InvoiceDate"].dt.to_period("W"))
        .groupby("week")["Revenue"]
        .sum()
    )
    weekly.index = weekly.index.to_timestamp()
    weekly = weekly.sort_index()
    return weekly


def _moving_average_forecast(series: pd.Series, horizon: int, window: int = 4) -> np.ndarray:
    """Simple rolling mean forecast."""
    last_window = series.iloc[-window:].mean()
    return np.full(horizon, last_window)


def _exponential_smoothing_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Holt-Winters exponential smoothing with additive seasonality."""
    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=min(52, len(series) // 2),
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)
        return fit.forecast(horizon).values
    except Exception as e:
        logger.warning("Exponential smoothing failed: %s. Falling back to MA.", e)
        return _moving_average_forecast(series, horizon)


def _features_from_index(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create time-based features for regression/GB models."""
    df_feat = pd.DataFrame(index=dates)
    df_feat["week_num"] = range(len(dates))   # linear time trend
    df_feat["month"] = dates.month
    df_feat["quarter"] = dates.quarter
    df_feat["week_of_year"] = dates.isocalendar().week.astype(int)
    df_feat["is_q4"] = (df_feat["quarter"] == 4).astype(int)
    # Fourier terms for weekly seasonality (52-week cycle)
    df_feat["sin_52"] = np.sin(2 * np.pi * df_feat["week_of_year"] / 52)
    df_feat["cos_52"] = np.cos(2 * np.pi * df_feat["week_of_year"] / 52)
    df_feat["sin_26"] = np.sin(2 * np.pi * df_feat["week_of_year"] / 26)
    df_feat["cos_26"] = np.cos(2 * np.pi * df_feat["week_of_year"] / 26)
    return df_feat


def _regression_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Ridge regression with time + seasonal Fourier features."""
    try:
        X = _features_from_index(series.index)
        y = series.values

        split = int(len(y) * FC_CFG["train_ratio"])
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)

        # Build future dates
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
        X_future = _features_from_index(future_dates)
        X_future["week_num"] = range(len(series), len(series) + horizon)
        X_future_s = scaler.transform(X_future)

        return model.predict(X_future_s).clip(min=0)
    except Exception as e:
        logger.warning("Regression forecast failed: %s. Falling back to MA.", e)
        return _moving_average_forecast(series, horizon)


def _gb_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Gradient Boosting with lag features and time features."""
    try:
        n_lags = min(4, len(series) // 4)
        df_tmp = series.to_frame(name="revenue")
        for lag in range(1, n_lags + 1):
            df_tmp[f"lag_{lag}"] = df_tmp["revenue"].shift(lag)
        df_tmp = df_tmp.dropna()

        feat_df = _features_from_index(df_tmp.index)
        df_feats = pd.concat([feat_df, df_tmp.drop(columns=["revenue"])], axis=1)
        y = df_tmp["revenue"].values

        split = int(len(y) * FC_CFG["train_ratio"])
        X_train, X_test = df_feats.iloc[:split], df_feats.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Iterative forecast (walk-forward)
        last_known = list(series.values[-n_lags:])
        future_preds = []
        last_date = series.index[-1]
        for step in range(horizon):
            future_date = last_date + pd.Timedelta(weeks=step + 1)
            feat_row = _features_from_index(pd.DatetimeIndex([future_date]))
            feat_row["week_num"] = len(series) + step
            for lag_i, lag_val in enumerate(reversed(last_known[-n_lags:]), 1):
                feat_row[f"lag_{lag_i}"] = lag_val
            # Align columns
            for col in df_feats.columns:
                if col not in feat_row.columns:
                    feat_row[col] = 0
            feat_row = feat_row[df_feats.columns]
            pred = float(model.predict(feat_row)[0])
            pred = max(0, pred)
            future_preds.append(pred)
            last_known.append(pred)

        return np.array(future_preds)
    except Exception as e:
        logger.warning("GB forecast failed: %s. Falling back to regression.", e)
        return _regression_forecast(series, horizon)


def evaluate_models(series: pd.Series) -> dict:
    """
    Evaluate all forecast models on a hold-out period.

    Uses chronological train/test split.
    Returns MAPE for each model and selects the best.
    """
    if len(series) < 8:
        return {"best_model": "moving_average", "mape": {}}

    split = int(len(series) * FC_CFG["train_ratio"])
    train = series.iloc[:split]
    test = series.iloc[split:]
    horizon = len(test)

    mapes = {}
    models = {
        "moving_average": _moving_average_forecast,
        "exponential_smoothing": _exponential_smoothing_forecast,
        "regression": _regression_forecast,
    }

    for name, fn in models.items():
        try:
            preds = fn(train, horizon)
            preds = np.maximum(preds, 0)
            # MAPE only where actual > 0
            actuals = test.values
            mask = actuals > 0
            if mask.sum() > 0:
                mape = mean_absolute_percentage_error(actuals[mask], preds[:len(actuals)][mask])
                mapes[name] = round(float(mape), 4)
            else:
                mapes[name] = 99.0
        except Exception as e:
            logger.warning("Model %s evaluation failed: %s", name, e)
            mapes[name] = 99.0

    best = min(mapes, key=mapes.get)
    logger.info("Model evaluation: %s | Best: %s (MAPE=%.2f)", mapes, best, mapes[best])
    return {"best_model": best, "mape": mapes}


def forecast_total_sales(df: pd.DataFrame, horizon_weeks: Optional[int] = None) -> dict:
    """
    Forecast total weekly revenue for the business.

    Returns
    -------
    dict:
        historical (date, revenue), forecast (date, revenue, lower, upper),
        best_model, mape, horizon_weeks
    """
    horizon = horizon_weeks or FC_CFG["horizon_weeks"]
    series = _make_weekly_series(df)

    eval_result = evaluate_models(series)
    best_model_name = eval_result["best_model"]

    model_fns = {
        "moving_average": _moving_average_forecast,
        "exponential_smoothing": _exponential_smoothing_forecast,
        "regression": _regression_forecast,
        "gradient_boosting": _gb_forecast,
    }

    fn = model_fns[best_model_name]
    preds = fn(series, horizon)
    preds = np.maximum(preds, 0)

    # Build forecast dates
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W"
    )

    # Simple uncertainty: ±std of recent residuals
    recent_std = float(series.iloc[-12:].std()) if len(series) >= 12 else float(series.std())
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast_revenue": preds.round(2),
        "lower_bound": np.maximum(0, preds - 1.96 * recent_std).round(2),
        "upper_bound": (preds + 1.96 * recent_std).round(2),
        "is_model_projection": True,
    })

    historical_df = series.reset_index()
    historical_df.columns = ["date", "revenue"]

    return {
        "historical": historical_df.to_dict(orient="records"),
        "forecast": forecast_df.to_dict(orient="records"),
        "best_model": best_model_name,
        "mape": eval_result["mape"],
        "horizon_weeks": horizon,
        "disclosure": "MODEL PROJECTION — Based on historical transaction patterns. Not a guarantee.",
    }


def forecast_product(
    df: pd.DataFrame, stock_code: str, horizon_weeks: Optional[int] = None
) -> dict:
    """
    Forecast weekly revenue for a single product.

    Returns
    -------
    dict with historical, forecast, model info.
    """
    horizon = horizon_weeks or FC_CFG["horizon_weeks"]
    series = _make_weekly_series(df, stock_code=stock_code)

    if len(series) < 8:
        logger.warning("Insufficient data for product %s forecast.", stock_code)
        return {"error": "Insufficient data", "stock_code": stock_code}

    eval_result = evaluate_models(series)
    best_model_name = eval_result["best_model"]

    model_fns = {
        "moving_average": _moving_average_forecast,
        "exponential_smoothing": _exponential_smoothing_forecast,
        "regression": _regression_forecast,
    }
    fn = model_fns.get(best_model_name, _moving_average_forecast)
    preds = fn(series, horizon)
    preds = np.maximum(preds, 0)

    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W"
    )
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast_revenue": preds.round(2),
        "is_model_projection": True,
    })
    historical_df = series.reset_index()
    historical_df.columns = ["date", "revenue"]

    return {
        "stock_code": stock_code,
        "historical": historical_df.to_dict(orient="records"),
        "forecast": forecast_df.to_dict(orient="records"),
        "best_model": best_model_name,
        "mape": eval_result["mape"],
        "disclosure": "MODEL PROJECTION — Based on historical transaction patterns.",
    }
