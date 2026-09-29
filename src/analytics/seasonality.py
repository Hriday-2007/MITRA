"""
MITRA — Seasonality & Time Intelligence
Analyzes day-of-week, hour, monthly, and yearly demand patterns.
Identifies seasonal products and holiday/peak-period effects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from config.settings import SEASONALITY_FEATURES_PATH

logger = get_logger("mitra.analytics.seasonality")

# Known peak months (UK retail calendar)
PEAK_MONTHS = {11: "Pre-Christmas", 12: "Christmas", 1: "New Year Sale", 2: "Valentine"}


def calculate_seasonality(df: pd.DataFrame) -> dict:
    """
    Full seasonality analysis.

    Returns
    -------
    dict with keys:
        day_of_week, hour_of_day, monthly, yearly,
        seasonal_products, peak_periods, seasonality_features_df
    """
    logger.info("Computing seasonality analysis…")

    result: dict = {}

    # ── Day of week ───────────────────────────────────────────────────────────
    dow = (
        df.groupby(["DayOfWeek", "DayName"])
        .agg(
            avg_revenue=("Revenue", "mean"),
            total_revenue=("Revenue", "sum"),
            avg_orders=("Invoice", "nunique"),
        )
        .reset_index()
        .sort_values("DayOfWeek")
    )
    dow["index_vs_avg"] = round(100 * dow["avg_revenue"] / dow["avg_revenue"].mean(), 1)
    result["day_of_week"] = dow

    # ── Hour of day ───────────────────────────────────────────────────────────
    hour = (
        df.groupby("Hour")
        .agg(
            avg_revenue=("Revenue", "mean"),
            total_revenue=("Revenue", "sum"),
            avg_orders=("Invoice", "nunique"),
        )
        .reset_index()
        .sort_values("Hour")
    )
    hour["index_vs_avg"] = round(100 * hour["avg_revenue"] / hour["avg_revenue"].mean(), 1)
    result["hour_of_day"] = hour

    # ── Monthly pattern ───────────────────────────────────────────────────────
    monthly = (
        df.groupby("Month")
        .agg(
            avg_revenue=("Revenue", "mean"),
            total_revenue=("Revenue", "sum"),
        )
        .reset_index()
    )
    monthly["month_name"] = monthly["Month"].apply(
        lambda m: pd.Timestamp(2000, m, 1).strftime("%B")
    )
    monthly["index_vs_avg"] = round(100 * monthly["avg_revenue"] / monthly["avg_revenue"].mean(), 1)
    monthly["is_peak"] = monthly["Month"].isin(PEAK_MONTHS.keys())
    monthly["peak_label"] = monthly["Month"].map(PEAK_MONTHS).fillna("")
    result["monthly"] = monthly

    # ── Yearly ───────────────────────────────────────────────────────────────
    yearly = (
        df.groupby("Year")
        .agg(
            total_revenue=("Revenue", "sum"),
            total_orders=("Invoice", "nunique"),
        )
        .reset_index()
    )
    result["yearly"] = yearly

    # ── Seasonal Products ─────────────────────────────────────────────────────
    # Find products with strong month-to-month variance (coefficient of variation)
    monthly_prod = (
        df.groupby(["StockCode", "Description", "Month"])["Revenue"]
        .sum()
        .reset_index()
    )
    prod_stats = (
        monthly_prod.groupby(["StockCode", "Description"])["Revenue"]
        .agg(["mean", "std", "max"])
        .reset_index()
    )
    prod_stats.columns = ["StockCode", "Description", "mean_monthly_rev", "std_monthly_rev", "max_monthly_rev"]
    prod_stats["cv"] = prod_stats["std_monthly_rev"] / prod_stats["mean_monthly_rev"].clip(lower=0.01)

    # Only include products with meaningful revenue
    min_rev_threshold = prod_stats["mean_monthly_rev"].quantile(0.5)
    seasonal_products = (
        prod_stats[prod_stats["mean_monthly_rev"] >= min_rev_threshold]
        .nlargest(30, "cv")
        .copy()
    )

    # Find each product's peak month
    peak_month_idx = monthly_prod.groupby("StockCode")["Revenue"].idxmax()
    peak_months_df = monthly_prod.loc[peak_month_idx, ["StockCode", "Month"]].rename(
        columns={"Month": "peak_month"}
    )
    peak_months_df["peak_month_name"] = peak_months_df["peak_month"].apply(
        lambda m: pd.Timestamp(2000, int(m), 1).strftime("%B")
    )
    seasonal_products = seasonal_products.merge(peak_months_df, on="StockCode", how="left")
    seasonal_products["cv"] = seasonal_products["cv"].round(3)
    result["seasonal_products"] = seasonal_products

    # ── Peak Period Detection ─────────────────────────────────────────────────
    # Q4 (Oct–Dec) uplift vs rest of year
    df_peak = df.copy()
    df_peak["is_q4"] = df_peak["Month"].isin([10, 11, 12])
    q4_rev = df_peak[df_peak["is_q4"]]["Revenue"].sum()
    non_q4_rev = df_peak[~df_peak["is_q4"]]["Revenue"].sum()
    peak_periods = {
        "q4_revenue": round(float(q4_rev), 2),
        "non_q4_revenue": round(float(non_q4_rev), 2),
        "q4_uplift_pct": round(
            100 * (q4_rev / (q4_rev + non_q4_rev) * 4 - 1), 2
        ) if non_q4_rev > 0 else 0.0,
        "busiest_month": int(monthly.loc[monthly["total_revenue"].idxmax(), "Month"]),
        "slowest_month": int(monthly.loc[monthly["total_revenue"].idxmin(), "Month"]),
        "busiest_day": dow.loc[dow["total_revenue"].idxmax(), "DayName"],
        "peak_hour": int(hour.loc[hour["total_revenue"].idxmax(), "Hour"]),
    }
    result["peak_periods"] = peak_periods

    # ── Seasonality Features DataFrame (for forecasting) ─────────────────────
    # Create a time-indexed dataset with seasonal flags
    daily = (
        df.groupby("Date")
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["Date"])
    daily["month"] = daily["date"].dt.month
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_q4"] = daily["month"].isin([10, 11, 12])
    daily["is_christmas_run"] = daily["month"] == 12
    daily["is_weekend"] = daily["day_of_week"] >= 5
    result["seasonality_features_df"] = daily

    logger.info(
        "Seasonality analysis complete. Busiest day: %s, Peak month: %s",
        peak_periods["busiest_day"],
        pd.Timestamp(2000, peak_periods["busiest_month"], 1).strftime("%B"),
    )
    return result


def save_seasonality(season_result: dict) -> None:
    """Save seasonality features to CSV."""
    df_feat = season_result.get("seasonality_features_df")
    if df_feat is not None:
        df_feat.to_csv(SEASONALITY_FEATURES_PATH, index=False)
        logger.info("Saved seasonality features → %s", SEASONALITY_FEATURES_PATH)
