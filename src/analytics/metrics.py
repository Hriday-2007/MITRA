"""
MITRA — Core Business Analytics Engine
All revenue, order, and customer calculations come from HERE, never from the LLM.

Functions are reusable, composable, and take a clean DataFrame as input.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger

logger = get_logger("mitra.analytics.metrics")

# ─── Revenue ──────────────────────────────────────────────────────────────────

def calculate_revenue(df: pd.DataFrame) -> float:
    """Total revenue across all transactions. Revenue = Quantity × Price."""
    return float(df["Revenue"].sum())


def calculate_aov(df: pd.DataFrame) -> float:
    """Average Order Value = Total Revenue / Number of Unique Invoices."""
    n_orders = df["Invoice"].nunique()
    if n_orders == 0:
        return 0.0
    return round(float(df["Revenue"].sum() / n_orders), 2)


def calculate_basket_size(df: pd.DataFrame) -> float:
    """Average Basket Size = Average number of distinct products per invoice."""
    basket = df.groupby("Invoice")["StockCode"].nunique()
    return round(float(basket.mean()), 2)


def calculate_units_sold(df: pd.DataFrame) -> int:
    """Total units sold."""
    return int(df["Quantity"].sum())


def calculate_order_count(df: pd.DataFrame) -> int:
    """Number of unique invoices (orders)."""
    return int(df["Invoice"].nunique())


# ─── Product Performance ─────────────────────────────────────────────────────

def calculate_product_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-product performance table.

    Returns
    -------
    pd.DataFrame with columns:
        StockCode, Description, total_units, total_revenue, n_orders,
        avg_qty_per_order, revenue_contribution_pct, n_customers,
        avg_unit_price
    """
    agg = (
        df.groupby(["StockCode", "Description"])
        .agg(
            total_units=("Quantity", "sum"),
            total_revenue=("Revenue", "sum"),
            n_orders=("Invoice", "nunique"),
            n_customers=("Customer ID", "nunique"),
            avg_unit_price=("Price", "mean"),
        )
        .reset_index()
    )

    # Average quantity per order
    avg_qty = (
        df.groupby(["StockCode", "Invoice"])["Quantity"].sum()
        .reset_index()
        .groupby("StockCode")["Quantity"].mean()
        .rename("avg_qty_per_order")
        .reset_index()
    )
    agg = agg.merge(avg_qty, on="StockCode", how="left")

    total_revenue = agg["total_revenue"].sum()
    agg["revenue_contribution_pct"] = round(
        100 * agg["total_revenue"] / total_revenue, 4
    )

    agg = agg.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1

    return agg


def calculate_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer summary.

    Excludes rows where Customer ID == 'UNKNOWN'.

    Returns
    -------
    pd.DataFrame with columns:
        Customer ID, total_revenue, n_orders, n_products, avg_order_value,
        first_purchase, last_purchase, customer_lifespan_days
    """
    cdf = df[df["Customer ID"] != "UNKNOWN"].copy()

    agg = (
        cdf.groupby("Customer ID")
        .agg(
            total_revenue=("Revenue", "sum"),
            n_orders=("Invoice", "nunique"),
            n_products=("StockCode", "nunique"),
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    agg["avg_order_value"] = round(agg["total_revenue"] / agg["n_orders"], 2)
    agg["customer_lifespan_days"] = (
        agg["last_purchase"] - agg["first_purchase"]
    ).dt.days

    return agg.sort_values("total_revenue", ascending=False).reset_index(drop=True)


def calculate_time_metrics(df: pd.DataFrame) -> dict:
    """
    Revenue and order metrics broken down by various time granularities.

    Returns
    -------
    dict with keys:
        daily, weekly, monthly, yearly,
        by_day_of_week, by_hour,
        growth_mom (month-over-month growth rates)
    """
    result: dict = {}

    # Daily
    daily = (
        df.groupby("Date")
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["Date"])
    result["daily"] = daily

    # Weekly
    weekly = (
        df.assign(YearWeek=df["InvoiceDate"].dt.to_period("W"))
        .groupby("YearWeek")
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
    )
    weekly["YearWeek"] = weekly["YearWeek"].astype(str)
    result["weekly"] = weekly

    # Monthly
    monthly = (
        df.assign(YearMonth=df["InvoiceDate"].dt.to_period("M"))
        .groupby("YearMonth")
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
    )
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)
    monthly["growth_mom"] = monthly["revenue"].pct_change() * 100
    result["monthly"] = monthly

    # Yearly
    yearly = (
        df.groupby("Year")
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
    )
    result["yearly"] = yearly

    # By Day of Week
    dow = (
        df.groupby(["DayOfWeek", "DayName"])
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
        .sort_values("DayOfWeek")
    )
    result["by_day_of_week"] = dow

    # By Hour
    hour = (
        df.groupby("Hour")
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
        .sort_values("Hour")
    )
    result["by_hour"] = hour

    # Month-over-month growth
    monthly_rev = monthly.set_index("YearMonth")["revenue"]
    result["growth_mom"] = monthly["growth_mom"].dropna().mean()

    return result


def calculate_country_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and orders by country."""
    agg = (
        df.groupby("Country")
        .agg(
            revenue=("Revenue", "sum"),
            orders=("Invoice", "nunique"),
            customers=("Customer ID", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    total = agg["revenue"].sum()
    agg["revenue_pct"] = round(100 * agg["revenue"] / total, 2)
    return agg


def calculate_repeat_rate(df: pd.DataFrame) -> dict:
    """
    Customer repeat purchase rate.

    Returns
    -------
    dict:
        total_customers, repeat_customers, one_time_customers,
        repeat_rate_pct, new_vs_returning breakdown
    """
    cdf = df[df["Customer ID"] != "UNKNOWN"]
    order_counts = cdf.groupby("Customer ID")["Invoice"].nunique()
    repeat = int((order_counts > 1).sum())
    one_time = int((order_counts == 1).sum())
    total = int(len(order_counts))
    return {
        "total_customers": total,
        "repeat_customers": repeat,
        "one_time_customers": one_time,
        "repeat_rate_pct": round(100 * repeat / total, 2) if total else 0,
    }


def calculate_summary(df: pd.DataFrame) -> dict:
    """
    High-level business summary — the top-level numbers used by the API
    and the LLM business brain.

    Returns
    -------
    dict: all key business KPIs
    """
    total_revenue = calculate_revenue(df)
    n_orders = calculate_order_count(df)
    aov = calculate_aov(df)
    units = calculate_units_sold(df)
    basket = calculate_basket_size(df)
    repeat = calculate_repeat_rate(df)
    country = calculate_country_metrics(df)
    time_m = calculate_time_metrics(df)

    # Top products by revenue
    prod = calculate_product_performance(df)
    top10 = prod.head(10)[
        ["rank", "StockCode", "Description", "total_revenue", "total_units", "n_orders"]
    ].to_dict(orient="records")

    # Date range
    date_min = df["InvoiceDate"].min()
    date_max = df["InvoiceDate"].max()

    # Customer counts
    n_customers = int(df[df["Customer ID"] != "UNKNOWN"]["Customer ID"].nunique())
    n_products = int(df["StockCode"].nunique())

    # MoM growth (last available month vs previous)
    monthly = time_m["monthly"]
    growth_mom = float(monthly["growth_mom"].dropna().iloc[-1]) if len(monthly) > 1 else 0.0

    return {
        "analysis_period": {
            "start": str(date_min.date()),
            "end": str(date_max.date()),
            "days": (date_max - date_min).days,
        },
        "total_revenue": round(total_revenue, 2),
        "total_orders": n_orders,
        "total_units_sold": units,
        "total_customers": n_customers,
        "total_products": n_products,
        "average_order_value": aov,
        "average_basket_size": basket,
        "repeat_rate_pct": repeat["repeat_rate_pct"],
        "revenue_growth_mom_pct": round(growth_mom, 2),
        "top_country": country.iloc[0]["Country"] if len(country) > 0 else "N/A",
        "top_10_products": top10,
    }
