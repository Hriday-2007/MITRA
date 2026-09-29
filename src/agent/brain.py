"""
MITRA — Business Brain / Merchant Profile Builder
Assembles the structured business state that the LLM reasons over.

The LLM receives this profile, NOT raw CSV data.
All numbers in this profile come from Python analytics — never from the LLM.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from config.settings import BUSINESS_PROFILE_PATH

logger = get_logger("mitra.agent.brain")


def _safe_records(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Convert a DataFrame slice to JSON-safe records."""
    return (
        df.head(n)
        .replace({float("inf"): None, float("-inf"): None})
        .where(pd.notnull(df.head(n)), None)
        .to_dict(orient="records")
    )


def build_business_profile(
    df_clean: pd.DataFrame,
    summary: dict,
    prod_intel: pd.DataFrame,
    prod_classifications: dict,
    rules: list[dict],
    rfm: pd.DataFrame,
    rfm_segments: pd.DataFrame,
    season: dict,
    forecast: dict,
    inv_risks: dict,
    portfolio_economics: dict,
    recommendations: list[dict] = None,
) -> dict:
    """
    Assemble the MITRA merchant business profile.

    This is the structured state that:
    1. Gets stored in the database
    2. Is loaded at the start of each AI agent session
    3. Is passed (not the raw CSV) to the LLM

    Parameters
    ----------
    All inputs come from the analytics pipeline — never from the LLM.

    Returns
    -------
    dict: structured merchant profile
    """
    logger.info("Building merchant business profile…")

    # ── Seasonal patterns summary ─────────────────────────────────────────────
    peak = season.get("peak_periods", {})
    seasonal_summary = {
        "busiest_day": peak.get("busiest_day"),
        "peak_hour": peak.get("peak_hour"),
        "busiest_month": peak.get("busiest_month"),
        "slowest_month": peak.get("slowest_month"),
        "q4_uplift_pct": peak.get("q4_uplift_pct"),
        "top_seasonal_products": _safe_records(
            season.get("seasonal_products", pd.DataFrame())
            [["StockCode", "Description", "cv", "peak_month_name"]]
            if not season.get("seasonal_products", pd.DataFrame()).empty
            else pd.DataFrame(),
            10,
        ),
    }

    # ── Customer segments summary ─────────────────────────────────────────────
    seg_summary = _safe_records(rfm_segments, 10)

    # ── Product intelligence lists ────────────────────────────────────────────
    def _prod_records(stock_codes: list[str]) -> list[dict]:
        subset = prod_intel[prod_intel["StockCode"].isin(stock_codes)][
            ["StockCode", "Description", "total_revenue", "total_units",
             "revenue_change_pct", "lifecycle_stage"]
        ]
        return _safe_records(subset, 15)

    # ── Forecast summary ─────────────────────────────────────────────────────
    fc_summary = {
        "best_model": forecast.get("best_model"),
        "mape": forecast.get("mape"),
        "horizon_weeks": forecast.get("horizon_weeks"),
        "next_4_weeks": forecast.get("forecast", [])[:4],
        "disclosure": forecast.get("disclosure"),
    }

    profile = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "business_type": "Online Retail",
        "dataset": "Online Retail II (UCI / Kaggle)",
        "currency": "GBP (£)",

        "analysis_period": summary.get("analysis_period"),

        # ── Top-line KPIs ─────────────────────────────────────────────────────
        "total_revenue": summary.get("total_revenue"),
        "total_orders": summary.get("total_orders"),
        "total_units_sold": summary.get("total_units_sold"),
        "total_customers": summary.get("total_customers"),
        "total_products": summary.get("total_products"),
        "average_order_value": summary.get("average_order_value"),
        "average_basket_size": summary.get("average_basket_size"),
        "repeat_rate_pct": summary.get("repeat_rate_pct"),
        "revenue_growth_mom_pct": summary.get("revenue_growth_mom_pct"),
        "top_country": summary.get("top_country"),

        # ── Economics (synthetic) ─────────────────────────────────────────────
        "portfolio_economics": portfolio_economics,

        # ── Product Lists ─────────────────────────────────────────────────────
        "top_products": _safe_records(
            prod_intel[["StockCode", "Description", "total_revenue", "total_units",
                         "n_orders", "gross_margin_pct"
                         if "gross_margin_pct" in prod_intel.columns else "total_revenue"]],
            20,
        ),
        "fast_growing_products": _prod_records(prod_classifications.get("growing", [])[:15]),
        "declining_products": _prod_records(prod_classifications.get("declining", [])[:15]),
        "fast_moving_products": _prod_records(prod_classifications.get("fast_moving", [])[:15]),
        "slow_moving_products": _prod_records(prod_classifications.get("slow_moving", [])[:15]),
        "new_products": _prod_records(prod_classifications.get("new", [])[:10]),
        "dormant_products": _prod_records(prod_classifications.get("dormant", [])[:10]),

        # ── Product Relationships ─────────────────────────────────────────────
        "product_relationships": rules[:20],

        # ── Customer Segments ─────────────────────────────────────────────────
        "customer_segments": seg_summary,

        # ── Seasonality ───────────────────────────────────────────────────────
        "seasonal_patterns": seasonal_summary,

        # ── Forecast ─────────────────────────────────────────────────────────
        "forecast": fc_summary,

        # ── Inventory Risks ───────────────────────────────────────────────────
        "inventory_risks": {
            "high_stockout_risk": inv_risks.get("stockout_risk_products", [])[:5],
            "high_overstock_risk": inv_risks.get("overstock_risk_products", [])[:5],
            "fast_movers": inv_risks.get("fast_movers", [])[:5],
            "slow_movers": inv_risks.get("slow_movers", [])[:5],
            "disclosure": inv_risks.get("disclosure"),
        },

        # ── Recommendations ───────────────────────────────────────────────────
        "recommendations": recommendations or [],

        # ── Disclosures ───────────────────────────────────────────────────────
        "disclosures": [
            "Revenue figures are calculated from transaction data (Quantity × UnitPrice).",
            "COGS is a synthetic estimate and does NOT represent actual merchant costs.",
            "Inventory levels are modelled from sales velocity, not actual stock counts.",
            "Demand forecasts are model projections, not guarantees.",
            "Customer segments are based on historical transaction patterns.",
        ],
    }

    logger.info("Business profile assembled with %d top-level keys.", len(profile))
    return profile


def save_business_profile(profile: dict) -> None:
    """Save the merchant business profile as JSON."""
    BUSINESS_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _json_serialisable(obj: Any) -> Any:
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(BUSINESS_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=_json_serialisable)
    logger.info("Saved merchant business profile → %s", BUSINESS_PROFILE_PATH)


def load_business_profile() -> dict:
    """Load the cached merchant business profile."""
    if not BUSINESS_PROFILE_PATH.exists():
        raise FileNotFoundError(
            "Business profile not found. Run the pipeline first."
        )
    with open(BUSINESS_PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
