"""
MITRA — Product Intelligence Module
Calculates per-product trends, growth rates, velocity, and lifecycle stage.
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
from config.settings import PRODUCT_PERFORMANCE_PATH

logger = get_logger("mitra.analytics.product_intel")


def _compute_trend_slope(series: pd.Series) -> float:
    """
    Fit a linear trend to a time series and return the normalised slope.
    Normalised = slope / mean(series) so products are comparable.
    """
    if len(series) < 3 or series.std() == 0:
        return 0.0
    x = np.arange(len(series))
    slope, *_ = linregress(x, series.values)
    mean_val = series.mean()
    if mean_val == 0:
        return 0.0
    return float(slope / mean_val)   # normalised growth rate per period


def calculate_product_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich product performance table with growth trends and lifecycle labels.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.

    Returns
    -------
    pd.DataFrame
        One row per product with:
        StockCode, Description, total_units, total_revenue, n_orders,
        avg_qty_per_order, revenue_contribution_pct, n_customers,
        avg_unit_price, monthly_revenue_trend (normalised slope),
        last_3m_revenue, prev_3m_revenue, revenue_change_pct,
        lifecycle_stage
    """
    logger.info("Computing product intelligence…")

    # ── Base metrics ─────────────────────────────────────────────────────────
    base = (
        df.groupby(["StockCode", "Description"])
        .agg(
            total_units=("Quantity", "sum"),
            total_revenue=("Revenue", "sum"),
            n_orders=("Invoice", "nunique"),
            n_customers=("Customer ID", "nunique"),
            avg_unit_price=("Price", "mean"),
            first_seen=("InvoiceDate", "min"),
            last_seen=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    # Avg qty per order
    avg_qty = (
        df.groupby(["StockCode", "Invoice"])["Quantity"].sum()
        .reset_index()
        .groupby("StockCode")["Quantity"].mean()
        .rename("avg_qty_per_order")
        .reset_index()
    )
    base = base.merge(avg_qty, on="StockCode", how="left")

    # Revenue contribution
    total_rev = base["total_revenue"].sum()
    base["revenue_contribution_pct"] = round(
        100 * base["total_revenue"] / total_rev, 4
    )

    # ── Monthly revenue per product ───────────────────────────────────────────
    df_m = df.copy()
    df_m["YearMonth"] = df_m["InvoiceDate"].dt.to_period("M")
    monthly_prod = (
        df_m.groupby(["StockCode", "YearMonth"])["Revenue"].sum().reset_index()
    )

    # Trend slope per product (normalised monthly growth rate)
    trends = (
        monthly_prod.groupby("StockCode")["Revenue"]
        .apply(_compute_trend_slope)
        .rename("monthly_revenue_trend")
        .reset_index()
    )
    base = base.merge(trends, on="StockCode", how="left")

    # ── Last 3 months vs previous 3 months ───────────────────────────────────
    snapshot = df["InvoiceDate"].max()
    last3_start = snapshot - pd.DateOffset(months=3)
    prev3_start = snapshot - pd.DateOffset(months=6)

    last3 = (
        df[df["InvoiceDate"] >= last3_start]
        .groupby("StockCode")["Revenue"].sum()
        .rename("last_3m_revenue")
        .reset_index()
    )
    prev3 = (
        df[(df["InvoiceDate"] >= prev3_start) & (df["InvoiceDate"] < last3_start)]
        .groupby("StockCode")["Revenue"].sum()
        .rename("prev_3m_revenue")
        .reset_index()
    )
    base = base.merge(last3, on="StockCode", how="left")
    base = base.merge(prev3, on="StockCode", how="left")
    base["last_3m_revenue"] = base["last_3m_revenue"].fillna(0)
    base["prev_3m_revenue"] = base["prev_3m_revenue"].fillna(0)

    base["revenue_change_pct"] = base.apply(
        lambda r: (
            round(100 * (r["last_3m_revenue"] - r["prev_3m_revenue"]) / r["prev_3m_revenue"], 2)
            if r["prev_3m_revenue"] > 0
            else (100.0 if r["last_3m_revenue"] > 0 else 0.0)
        ),
        axis=1,
    )

    # ── Lifecycle Stage ───────────────────────────────────────────────────────
    # Based on trend slope + recent revenue:
    # growing    : positive trend AND recent revenue
    # declining  : negative trend AND had prior revenue
    # steady     : flat trend
    # new        : only in last 3 months
    # dormant    : not seen in last 3 months
    def _lifecycle(row) -> str:
        if row["last_3m_revenue"] == 0 and row["prev_3m_revenue"] > 0:
            return "dormant"
        if row["last_3m_revenue"] > 0 and row["prev_3m_revenue"] == 0:
            return "new"
        if row["monthly_revenue_trend"] > 0.05:
            return "growing"
        if row["monthly_revenue_trend"] < -0.05:
            return "declining"
        return "steady"

    base["lifecycle_stage"] = base.apply(_lifecycle, axis=1)

    # ── Purchase frequency (avg months between purchases) ────────────────────
    first_last = base[["StockCode", "first_seen", "last_seen"]].copy()
    first_last["active_months"] = (
        (base["last_seen"] - base["first_seen"]).dt.days / 30.44
    ).clip(lower=1)
    monthly_prod_count = (
        monthly_prod.groupby("StockCode")["YearMonth"].count().rename("n_active_months").reset_index()
    )
    base = base.merge(monthly_prod_count, on="StockCode", how="left")
    base["purchase_frequency_score"] = round(
        base["n_active_months"] / first_last["active_months"].clip(lower=1).values, 2
    )

    base = base.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    base["rank"] = base.index + 1

    logger.info("Product intelligence computed for %d products.", len(base))
    return base


def classify_products(prod_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Classify products into strategic buckets.

    Returns
    -------
    dict with keys:
        top_revenue, fast_moving, slow_moving, growing, declining, dormant, new
    """
    top_rev = prod_df.nlargest(20, "total_revenue")["StockCode"].tolist()
    fast = prod_df.nlargest(20, "total_units")["StockCode"].tolist()
    slow = prod_df.nsmallest(20, "total_units").query("total_revenue > 0")["StockCode"].tolist()
    growing = prod_df[prod_df["lifecycle_stage"] == "growing"]["StockCode"].tolist()
    declining = prod_df[prod_df["lifecycle_stage"] == "declining"]["StockCode"].tolist()
    dormant = prod_df[prod_df["lifecycle_stage"] == "dormant"]["StockCode"].tolist()
    new = prod_df[prod_df["lifecycle_stage"] == "new"]["StockCode"].tolist()

    return {
        "top_revenue": top_rev,
        "fast_moving": fast,
        "slow_moving": slow,
        "growing": growing,
        "declining": declining,
        "dormant": dormant,
        "new": new,
    }


def save_product_intelligence(prod_df: pd.DataFrame) -> None:
    """Save product performance table to CSV."""
    prod_df.to_csv(PRODUCT_PERFORMANCE_PATH, index=False)
    logger.info("Saved product performance → %s", PRODUCT_PERFORMANCE_PATH)
