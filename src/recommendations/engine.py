"""
MITRA — Rule-Based Recommendation Engine
Generates structured recommendations from quantitative analytics.

CRITICAL: Every recommendation is traceable to actual metrics.
The LLM never invents evidence — it only explains outputs from here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger

logger = get_logger("mitra.recommendations.engine")


def _rec(
    rec_type: str,
    title: str,
    evidence: list[Any],
    expected_impact: str,
    confidence: float,
    recommended_action: str,
    metrics_used: list[str],
) -> dict:
    """Helper to construct a standardised recommendation object."""
    return {
        "type": rec_type,
        "title": title,
        "evidence": evidence,
        "expected_impact": expected_impact,
        "confidence": round(confidence, 2),
        "recommended_action": recommended_action,
        "metrics_used": metrics_used,
    }


def generate_bundle_recommendations(
    rules: list[dict],
    prod_intel: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    Bundle opportunities from association rules.
    Filter for revenue-relevant, high-lift combinations.
    """
    recs = []
    for rule in rules[:top_n]:
        ant = rule.get("antecedent_label", "")
        con = rule.get("consequent_label", "")
        lift = rule.get("lift", 0)
        conf = rule.get("confidence", 0)
        txns = rule.get("transaction_count", 0)
        rev = rule.get("combined_revenue", 0)

        recs.append(_rec(
            rec_type="bundle",
            title=f"Bundle opportunity: {ant[:30]} + {con[:30]}",
            evidence=[
                f"Customers buying {ant[:40]} also buy {con[:40]} {conf*100:.0f}% of the time",
                f"Lift = {lift:.2f}x (expected co-purchase rate vs. random)",
                f"Observed in {txns:,} transactions",
                f"Combined product revenue: £{rev:,.0f}",
            ],
            expected_impact=f"Bundling could increase average order value by ~{min(conf * 15, 20):.0f}% on qualifying orders",
            confidence=min(conf * lift / 5, 0.9),
            recommended_action=(
                f"Create a bundle discount for {ant[:30]} + {con[:30]}. "
                f"Offer 5-10% discount when bought together. "
                f"Promote on POS and product pages."
            ),
            metrics_used=["association_rules", "support", "confidence", "lift", "revenue"],
        ))
    return recs


def generate_inventory_recommendations(inv_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Inventory action recommendations for high-risk products.
    """
    recs = []

    # Stockout risks (high demand, at risk of running out)
    high_risk = inv_df[inv_df["stockout_risk"] == "HIGH"].nlargest(top_n, "total_revenue")
    for _, row in high_risk.iterrows():
        recs.append(_rec(
            rec_type="inventory_stockout",
            title=f"Reorder now: {row['Description'][:40]}",
            evidence=[
                f"Current modelled stock: {row['modelled_stock_units']:,} units",
                f"Average daily demand: {row['avg_daily_units']:.1f} units/day",
                f"Days of inventory remaining: {row['days_of_inventory_remaining']} days",
                f"Supplier lead time: 14 days (config assumption)",
                f"Product revenue: £{row['total_revenue']:,.0f}",
            ],
            expected_impact=f"Avoiding stockout prevents potential lost revenue of ~£{row['avg_daily_revenue']*14:,.0f} over lead time",
            confidence=0.75,
            recommended_action=(
                f"Place reorder for {row['Description'][:30]} immediately. "
                f"Recommended quantity: {row['recommended_reorder_qty']:,} units."
            ),
            metrics_used=["inventory_model", "daily_demand", "stockout_risk"],
        ))

    # Overstock (slow movers with too much stock)
    overstock = inv_df[
        (inv_df["overstock_risk"] == "HIGH") & (inv_df["velocity_label"] == "slow_moving")
    ].head(top_n)
    for _, row in overstock.iterrows():
        recs.append(_rec(
            rec_type="inventory_overstock",
            title=f"Reduce slow inventory: {row['Description'][:40]}",
            evidence=[
                f"Modelled days of stock: {row['days_of_inventory_remaining']} days (target: 30)",
                f"Average daily demand: {row['avg_daily_units']:.2f} units/day",
                f"Classified as: slow-moving",
            ],
            expected_impact="Reducing overstock frees up working capital and storage costs",
            confidence=0.65,
            recommended_action=(
                f"Consider a 10-15% promotional discount on {row['Description'][:30]} "
                f"to accelerate sell-through. Avoid reordering until stock normalises."
            ),
            metrics_used=["inventory_model", "velocity_label", "overstock_risk"],
        ))

    return recs


def generate_growth_recommendations(prod_intel: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Recommend increasing investment in growing products.
    """
    growing = prod_intel[prod_intel["lifecycle_stage"] == "growing"].nlargest(top_n, "total_revenue")
    recs = []
    for _, row in growing.iterrows():
        recs.append(_rec(
            rec_type="growth_product",
            title=f"Scale up: {row['Description'][:40]}",
            evidence=[
                f"Revenue trend slope: +{row['monthly_revenue_trend']*100:.1f}% per month",
                f"Revenue last 3 months: £{row.get('last_3m_revenue', 0):,.0f}",
                f"Revenue change vs prior 3 months: +{row['revenue_change_pct']:.1f}%",
                f"Current total revenue: £{row['total_revenue']:,.0f}",
            ],
            expected_impact=f"Increasing stock and marketing could capture {min(row['monthly_revenue_trend']*100+5, 30):.0f}% more revenue at current growth rate",
            confidence=min(abs(row["monthly_revenue_trend"]) * 3, 0.85),
            recommended_action=(
                f"Increase stock levels and marketing spend for {row['Description'][:30]}. "
                f"Consider featuring it prominently in displays/promotions."
            ),
            metrics_used=["revenue_trend", "revenue_change_pct", "lifecycle_stage"],
        ))
    return recs


def generate_declining_recommendations(prod_intel: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Recommend actions for declining products.
    """
    declining = prod_intel[prod_intel["lifecycle_stage"] == "declining"].nlargest(top_n, "total_revenue")
    recs = []
    for _, row in declining.iterrows():
        recs.append(_rec(
            rec_type="declining_product",
            title=f"Address declining sales: {row['Description'][:40]}",
            evidence=[
                f"Revenue trend slope: {row['monthly_revenue_trend']*100:.1f}% per month",
                f"Revenue last 3 months: £{row.get('last_3m_revenue', 0):,.0f}",
                f"Revenue change vs prior 3 months: {row['revenue_change_pct']:.1f}%",
            ],
            expected_impact="Proactive action can slow or reverse revenue decline",
            confidence=min(abs(row["monthly_revenue_trend"]) * 3, 0.80),
            recommended_action=(
                f"Investigate why {row['Description'][:30]} is declining. "
                f"Options: promotional discount, bundling with complementary product, "
                f"or reducing reorder quantity to avoid overstock."
            ),
            metrics_used=["revenue_trend", "revenue_change_pct", "lifecycle_stage"],
        ))
    return recs


def generate_customer_retention_recommendations(rfm: pd.DataFrame) -> list[dict]:
    """
    Customer retention opportunities based on RFM segments.
    """
    recs = []

    # At-risk high-value customers
    at_risk = rfm[rfm["segment"] == "At Risk"].nlargest(1, "monetary")
    if len(at_risk) > 0:
        n = len(rfm[rfm["segment"] == "At Risk"])
        avg_m = rfm[rfm["segment"] == "At Risk"]["monetary"].mean()
        recs.append(_rec(
            rec_type="customer_retention",
            title=f"Re-engage {n:,} at-risk customers",
            evidence=[
                f"{n:,} previously good customers have not purchased recently",
                f"Average historical spend: £{avg_m:,.0f} per customer",
                f"Combined at-risk revenue at stake: £{n * avg_m:,.0f}",
                f"RFM segment: 'At Risk' (low recency, good historical frequency)",
            ],
            expected_impact=f"If 20% respond: £{n * avg_m * 0.20:,.0f} recovered revenue",
            confidence=0.60,
            recommended_action=(
                "Send personalised re-engagement email/SMS to At Risk customers. "
                "Offer a 10% loyalty discount valid for 30 days. "
                "Personalise with their previously purchased product categories."
            ),
            metrics_used=["rfm_segment", "recency_days", "monetary", "frequency"],
        ))

    # Cant Lose Them
    cant_lose = rfm[rfm["segment"] == "Cant Lose Them"]
    if len(cant_lose) > 0:
        n = len(cant_lose)
        avg_m = cant_lose["monetary"].mean()
        recs.append(_rec(
            rec_type="customer_retention",
            title=f"Win back {n:,} high-value lapsed customers",
            evidence=[
                f"{n:,} customers who used to buy frequently are now inactive",
                f"Average historical spend: £{avg_m:,.0f} per customer",
                f"RFM segment: 'Cant Lose Them' — very high frequency, very low recency",
            ],
            expected_impact=f"Recovering 15% could restore £{n * avg_m * 0.15:,.0f} annual revenue",
            confidence=0.55,
            recommended_action=(
                "Launch a 'We miss you' win-back campaign. "
                "Offer a significant discount (15-20%) or free shipping. "
                "Highlight new products since their last purchase."
            ),
            metrics_used=["rfm_segment", "recency_days", "monetary"],
        ))

    return recs


def generate_seasonal_recommendations(season: dict, prod_intel: pd.DataFrame) -> list[dict]:
    """
    Seasonal demand recommendations.
    """
    recs = []
    peak = season.get("peak_periods", {})
    seasonal_prods = season.get("seasonal_products", pd.DataFrame())

    if not seasonal_prods.empty:
        top_seasonal = seasonal_prods.head(3)
        for _, row in top_seasonal.iterrows():
            recs.append(_rec(
                rec_type="seasonal",
                title=f"Prepare for seasonal peak: {row['Description'][:35]}",
                evidence=[
                    f"Coefficient of variation: {row['cv']:.2f} (high = strong seasonality)",
                    f"Peak month: {row.get('peak_month_name', 'N/A')}",
                    f"Average monthly revenue: £{row['mean_monthly_rev']:,.0f}",
                    f"Peak monthly revenue: £{row['max_monthly_rev']:,.0f}",
                ],
                expected_impact=f"Peak month can generate {row['max_monthly_rev'] / max(row['mean_monthly_rev'], 1):.1f}x average monthly revenue",
                confidence=0.70,
                recommended_action=(
                    f"Stock up on {row['Description'][:30]} before {row.get('peak_month_name', 'peak month')}. "
                    f"Plan promotional campaigns 4-6 weeks in advance."
                ),
                metrics_used=["seasonality_cv", "monthly_revenue", "peak_month"],
            ))

    # Q4 / Christmas
    q4_uplift = peak.get("q4_uplift_pct", 0)
    if q4_uplift and q4_uplift > 5:
        recs.append(_rec(
            rec_type="seasonal",
            title="Prepare for Q4 Christmas surge",
            evidence=[
                f"Q4 (Oct-Dec) revenue is {q4_uplift:.1f}% above average quarterly rate",
                f"Q4 total revenue: £{peak.get('q4_revenue', 0):,.0f}",
                f"Busiest month: Month {peak.get('busiest_month', 'N/A')}",
            ],
            expected_impact=f"Q4 represents ~{q4_uplift:.0f}% more revenue than average quarter",
            confidence=0.80,
            recommended_action=(
                "Stock up on high-velocity products in September. "
                "Prepare Christmas bundles by October. "
                "Ensure staffing/fulfilment capacity for peak period."
            ),
            metrics_used=["q4_revenue", "seasonality_index", "monthly_revenue"],
        ))

    return recs


def generate_all_recommendations(
    profile: dict,
    prod_intel: pd.DataFrame,
    rfm: pd.DataFrame,
    season: dict,
    inv_df: pd.DataFrame,
) -> list[dict]:
    """
    Generate all recommendation types and return sorted by confidence.

    Parameters
    ----------
    profile : dict — merchant business profile
    prod_intel : pd.DataFrame — product intelligence table
    rfm : pd.DataFrame — RFM customer table
    season : dict — seasonality analysis
    inv_df : pd.DataFrame — inventory model

    Returns
    -------
    list[dict] — sorted recommendations
    """
    recs: list[dict] = []

    rules = profile.get("product_relationships", [])
    recs += generate_bundle_recommendations(rules, prod_intel)
    recs += generate_inventory_recommendations(inv_df)
    recs += generate_growth_recommendations(prod_intel)
    recs += generate_declining_recommendations(prod_intel)
    recs += generate_customer_retention_recommendations(rfm)
    recs += generate_seasonal_recommendations(season, prod_intel)

    # Sort by confidence descending
    recs.sort(key=lambda r: r["confidence"], reverse=True)
    logger.info("Generated %d total recommendations.", len(recs))
    return recs
