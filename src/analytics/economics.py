"""
MITRA — Synthetic COGS & Product Economics
⚠️  IMPORTANT DISCLOSURE ⚠️
The Online Retail II dataset does NOT contain real COGS data.
All cost figures here are ESTIMATED using synthetic cost ratios.
These are for demonstration purposes only and must NOT be presented as
actual merchant costs.

The cost ratios are configurable in config/settings.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from config.settings import (
    COST_RATIO_BY_KEYWORD,
    COST_RATIO_DEFAULT,
    PRODUCT_ECONOMICS_PATH,
)

logger = get_logger("mitra.analytics.economics")

DISCLOSURE = (
    "⚠️  SYNTHETIC COGS MODEL — Estimated cost ratios for demonstration only. "
    "Not real merchant data."
)


def _estimate_cost_ratio(description: str) -> float:
    """
    Estimate cost ratio from product description keywords.

    The Online Retail II dataset has no COGS column.
    We assign cost ratios based on common product category cost structures.
    """
    desc_lower = str(description).lower()
    for keyword, ratio in COST_RATIO_BY_KEYWORD.items():
        if keyword in desc_lower:
            return ratio
    return COST_RATIO_DEFAULT


def calculate_product_economics(prod_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic COGS and margin estimates to the product performance table.

    Parameters
    ----------
    prod_df : pd.DataFrame
        Output of calculate_product_performance() or calculate_product_intelligence().

    Returns
    -------
    pd.DataFrame with additional columns:
        estimated_cost_ratio, estimated_cogs, gross_profit, gross_margin_pct,
        is_synthetic_cogs (always True)
    """
    logger.info("Computing synthetic COGS model… [%s]", DISCLOSURE)

    df = prod_df.copy()

    df["estimated_cost_ratio"] = df["Description"].apply(_estimate_cost_ratio)
    df["estimated_cogs"] = round(df["total_revenue"] * df["estimated_cost_ratio"], 2)
    df["gross_profit"] = round(df["total_revenue"] - df["estimated_cogs"], 2)
    df["gross_margin_pct"] = round(
        100 * df["gross_profit"] / df["total_revenue"].clip(lower=0.01), 2
    )
    df["is_synthetic_cogs"] = True   # Always mark synthetic

    logger.info(
        "Economics computed: avg gross margin = %.1f%%",
        float(df["gross_margin_pct"].mean()),
    )
    return df


def calculate_transaction_economics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic COGS to the cleaned transaction DataFrame for simulation.

    Returns
    -------
    pd.DataFrame with additional columns:
        estimated_cost_ratio, estimated_cogs, gross_profit
    """
    result = df.copy()
    result["estimated_cost_ratio"] = result["Description"].apply(_estimate_cost_ratio)
    result["estimated_cogs"] = result["Revenue"] * result["estimated_cost_ratio"]
    result["gross_profit"] = result["Revenue"] - result["estimated_cogs"]
    result["is_synthetic_cogs"] = True
    return result


def calculate_portfolio_economics(prod_economics: pd.DataFrame) -> dict:
    """
    Portfolio-level economics summary.

    Returns
    -------
    dict: total_revenue, total_cogs, total_gross_profit, portfolio_margin_pct,
          disclosure
    """
    total_rev = float(prod_economics["total_revenue"].sum())
    total_cogs = float(prod_economics["estimated_cogs"].sum())
    gross_profit = total_rev - total_cogs
    margin = 100 * gross_profit / total_rev if total_rev > 0 else 0.0
    return {
        "total_revenue": round(total_rev, 2),
        "total_estimated_cogs": round(total_cogs, 2),
        "total_gross_profit": round(gross_profit, 2),
        "portfolio_gross_margin_pct": round(margin, 2),
        "disclosure": DISCLOSURE,
    }


def save_product_economics(prod_economics: pd.DataFrame) -> None:
    """Save product economics to CSV."""
    prod_economics.to_csv(PRODUCT_ECONOMICS_PATH, index=False)
    logger.info("Saved product economics → %s", PRODUCT_ECONOMICS_PATH)
