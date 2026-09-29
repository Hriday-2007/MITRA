"""
MITRA — RFM Customer Segmentation
Recency, Frequency, Monetary analysis with data-driven segmentation.

Methodology:
- Snapshot date = last transaction date + 1 day
- Recency  = days since last purchase (lower = better → inverted score)
- Frequency = number of unique invoices
- Monetary  = total revenue

Scoring: Quartile-based (1–4), combined into 11-segment map.
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
from config.settings import RFM as RFM_CFG, CUSTOMER_SEGMENTS_PATH

logger = get_logger("mitra.analytics.rfm")

# ─── Segment Map (RFM score → segment name) ──────────────────────────────────
# Score = R_score (1–4) + F_score (1–4) + M_score (1–4)
# We map on combined RF pattern first, then M for tie-breaking.

SEGMENT_MAP: dict[tuple, str] = {
    # (R_score, F_score) → segment
    (4, 4): "Champions",
    (4, 3): "Champions",
    (3, 4): "Loyal Customers",
    (3, 3): "Loyal Customers",
    (4, 2): "Potential Loyalists",
    (3, 2): "Potential Loyalists",
    (4, 1): "New Customers",
    (3, 1): "New Customers",
    (2, 4): "At Risk",
    (2, 3): "At Risk",
    (2, 2): "Needs Attention",
    (1, 4): "Cant Lose Them",
    (1, 3): "Cant Lose Them",
    (2, 1): "About To Sleep",
    (1, 2): "Hibernating",
    (1, 1): "Lost Customers",
}

SEGMENT_DESCRIPTION: dict[str, str] = {
    "Champions": "Bought recently, buy often, spend the most",
    "Loyal Customers": "Buy regularly with good spending",
    "Potential Loyalists": "Recent customers with average frequency",
    "New Customers": "Bought recently but not yet frequent",
    "At Risk": "Were good customers but haven't bought recently",
    "Cant Lose Them": "Used to buy often but haven't returned",
    "Needs Attention": "Above average recency/frequency/monetary but not consistent",
    "About To Sleep": "Below average recency, frequency, and monetary",
    "Hibernating": "Last purchase was long ago with low frequency",
    "Lost Customers": "Lowest recency, frequency, and monetary values",
}


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM scores for all identified customers.

    Excludes 'UNKNOWN' Customer ID rows.

    Returns
    -------
    pd.DataFrame with columns:
        Customer ID, recency_days, frequency, monetary,
        R_score, F_score, M_score, RFM_score, segment, segment_description
    """
    logger.info("Computing RFM scores…")

    # Exclude unknown customers
    cdf = df[df["Customer ID"] != "UNKNOWN"].copy()

    snapshot = cdf["InvoiceDate"].max() + pd.Timedelta(days=RFM_CFG["snapshot_date_offset_days"])
    logger.info("  Snapshot date: %s", snapshot.date())

    # Aggregate
    rfm = (
        cdf.groupby("Customer ID")
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("Invoice", "nunique"),
            monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (snapshot - rfm["last_purchase"]).dt.days

    # ── Quartile Scoring ──────────────────────────────────────────────────────
    # Recency: lower recency_days = higher score (inverted)
    rfm["R_score"] = pd.qcut(
        rfm["recency_days"],
        q=RFM_CFG["recency_quantiles"],
        labels=[4, 3, 2, 1],   # inverted: low days → score 4
        duplicates="drop",
    ).astype(int)

    rfm["F_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        q=RFM_CFG["frequency_quantiles"],
        labels=[1, 2, 3, 4],
        duplicates="drop",
    ).astype(int)

    rfm["M_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"),
        q=RFM_CFG["monetary_quantiles"],
        labels=[1, 2, 3, 4],
        duplicates="drop",
    ).astype(int)

    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

    # ── Segment Assignment ────────────────────────────────────────────────────
    def _assign_segment(row) -> str:
        key = (int(row["R_score"]), int(row["F_score"]))
        return SEGMENT_MAP.get(key, "Needs Attention")

    rfm["segment"] = rfm.apply(_assign_segment, axis=1)
    rfm["segment_description"] = rfm["segment"].map(SEGMENT_DESCRIPTION).fillna("")

    rfm = rfm.sort_values("monetary", ascending=False).reset_index(drop=True)
    logger.info(
        "RFM complete: %d customers, %d segments",
        len(rfm), rfm["segment"].nunique(),
    )
    return rfm


def summarise_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Segment-level summary: count, avg RFM, total revenue per segment.

    Returns
    -------
    pd.DataFrame
    """
    summary = (
        rfm.groupby("segment")
        .agg(
            n_customers=("Customer ID", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    summary["revenue_pct"] = round(
        100 * summary["total_revenue"] / summary["total_revenue"].sum(), 2
    )
    for col in ["avg_recency_days", "avg_frequency", "avg_monetary"]:
        summary[col] = summary[col].round(1)
    return summary


def save_segments(rfm: pd.DataFrame) -> None:
    """Save customer segments to CSV."""
    rfm.to_csv(CUSTOMER_SEGMENTS_PATH, index=False)
    logger.info("Saved customer segments → %s", CUSTOMER_SEGMENTS_PATH)
