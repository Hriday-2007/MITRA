"""
MITRA — Demo Inventory Model
⚠️  IMPORTANT DISCLOSURE ⚠️
The Online Retail II dataset contains NO inventory data.
All inventory levels here are MODELLED estimates based on observed sales velocity.
They are for demonstration purposes only.

Model assumptions (configurable in config/settings.py):
- Initial stock = 30 days of average daily demand
- Safety stock = 7 days of average daily demand
- Lead time = 14 days
- Reorder point = 21 days of stock remaining
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from config.settings import INVENTORY as INV_CFG

logger = get_logger("mitra.ml.inventory")

DISCLOSURE = (
    "⚠️  DEMO INVENTORY MODEL — Estimated from sales velocity only. "
    "Not real inventory data."
)


def build_inventory_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a demo inventory model for each product.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.

    Returns
    -------
    pd.DataFrame with columns:
        StockCode, Description, avg_daily_units, avg_daily_revenue,
        modelled_stock_units, days_of_inventory_remaining,
        reorder_point_units, safety_stock_units, recommended_reorder_qty,
        stockout_risk, overstock_risk, velocity_label, is_demo_inventory
    """
    logger.info("Building demo inventory model… [%s]", DISCLOSURE)

    # Date range for daily demand calculation
    date_range_days = max(
        (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days, 1
    )

    # Per-product demand
    prod = (
        df.groupby(["StockCode", "Description"])
        .agg(
            total_units=("Quantity", "sum"),
            total_revenue=("Revenue", "sum"),
            n_days_active=("Date", "nunique"),
        )
        .reset_index()
    )

    # Daily demand = total units / active selling days
    prod["avg_daily_units"] = (prod["total_units"] / prod["n_days_active"]).round(2)
    prod["avg_daily_revenue"] = (prod["total_revenue"] / prod["n_days_active"]).round(2)

    # Modelled starting inventory (30 days of average demand)
    prod["modelled_stock_units"] = (
        prod["avg_daily_units"] * INV_CFG["initial_stock_days"]
    ).round(0).astype(int)

    # Days of inventory remaining (at current demand rate)
    prod["days_of_inventory_remaining"] = INV_CFG["initial_stock_days"]  # starting point

    # Safety stock
    prod["safety_stock_units"] = (
        prod["avg_daily_units"] * INV_CFG["safety_stock_days"]
    ).round(0).astype(int)

    # Reorder point = lead_time demand + safety stock
    prod["reorder_point_units"] = (
        prod["avg_daily_units"] * (INV_CFG["lead_time_days"] + INV_CFG["safety_stock_days"])
    ).round(0).astype(int)

    # Recommended reorder quantity (Economic order quantity approximation)
    # EOQ = sqrt(2 * D * S / H) — we use simplified version
    # D = annual demand, S = order cost (assumed 50), H = holding cost (assumed 20% of unit cost)
    avg_price = df.groupby("StockCode")["Price"].mean().rename("avg_unit_price").reset_index()
    prod = prod.merge(avg_price, on="StockCode", how="left")
    prod["avg_unit_price"] = prod["avg_unit_price"].fillna(1.0)

    annual_demand = prod["avg_daily_units"] * 365
    order_cost = 50  # assumed fixed order cost (GBP)
    holding_cost_rate = 0.20
    holding_cost = prod["avg_unit_price"] * holding_cost_rate
    prod["recommended_reorder_qty"] = (
        np.sqrt(2 * annual_demand * order_cost / holding_cost.clip(lower=0.01))
    ).round(0).astype(int)

    # Risk classification
    def _stockout_risk(row) -> str:
        days = row["days_of_inventory_remaining"]
        if days <= INV_CFG["lead_time_days"]:
            return "HIGH"
        if days <= INV_CFG["reorder_point_days"]:
            return "MEDIUM"
        return "LOW"

    def _overstock_risk(row) -> str:
        days = row["days_of_inventory_remaining"]
        if days > 90:
            return "HIGH"
        if days > 60:
            return "MEDIUM"
        return "LOW"

    def _velocity_label(row) -> str:
        daily = row["avg_daily_units"]
        p75 = prod["avg_daily_units"].quantile(0.75)
        p25 = prod["avg_daily_units"].quantile(0.25)
        if daily >= p75:
            return "fast_moving"
        if daily <= p25:
            return "slow_moving"
        return "moderate"

    prod["stockout_risk"] = prod.apply(_stockout_risk, axis=1)
    prod["overstock_risk"] = prod.apply(_overstock_risk, axis=1)
    prod["velocity_label"] = prod.apply(_velocity_label, axis=1)
    prod["is_demo_inventory"] = True

    prod = prod.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    logger.info(
        "Inventory model built: %d products | HIGH stockout risk: %d",
        len(prod),
        int((prod["stockout_risk"] == "HIGH").sum()),
    )
    return prod


def get_inventory_risks(inv_df: pd.DataFrame) -> dict:
    """
    Extract high-risk inventory items for the business brain.

    Returns
    -------
    dict with stockout_risk_products, overstock_risk_products, fast_movers, slow_movers
    """
    high_stockout = inv_df[inv_df["stockout_risk"] == "HIGH"][
        ["StockCode", "Description", "avg_daily_units", "days_of_inventory_remaining"]
    ].head(10).to_dict(orient="records")

    high_overstock = inv_df[inv_df["overstock_risk"] == "HIGH"][
        ["StockCode", "Description", "avg_daily_units", "modelled_stock_units"]
    ].head(10).to_dict(orient="records")

    fast_movers = inv_df[inv_df["velocity_label"] == "fast_moving"][
        ["StockCode", "Description", "avg_daily_units", "total_revenue"]
    ].head(10).to_dict(orient="records")

    slow_movers = inv_df[inv_df["velocity_label"] == "slow_moving"][
        ["StockCode", "Description", "avg_daily_units", "total_revenue"]
    ].head(10).to_dict(orient="records")

    return {
        "stockout_risk_products": high_stockout,
        "overstock_risk_products": high_overstock,
        "fast_movers": fast_movers,
        "slow_movers": slow_movers,
        "disclosure": DISCLOSURE,
    }
