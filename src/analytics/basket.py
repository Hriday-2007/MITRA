"""
MITRA — Market Basket Analysis (Association Rule Mining)
Discovers product co-purchase relationships using the Apriori algorithm.

Uses mlxtend for association rule mining.
Filters rules by support, confidence, lift, and minimum transaction count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from config.settings import ASSOCIATION_RULES as AR_CFG, PRODUCT_RELATIONSHIPS_PATH

logger = get_logger("mitra.analytics.basket")


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary invoice × product matrix for Apriori.

    Rows    = Invoice
    Columns = StockCode
    Values  = 1 (purchased) / 0 (not purchased)

    Only invoices with >= 2 distinct products are kept
    (single-item baskets add noise but no co-purchase signal).

    Returns
    -------
    pd.DataFrame (binary, bool dtype for mlxtend)
    """
    logger.info("Building basket matrix…")

    # Count distinct products per invoice
    basket_size = df.groupby("Invoice")["StockCode"].nunique()
    valid_invoices = basket_size[basket_size >= 2].index

    logger.info(
        "  Valid baskets (>= 2 products): %d / %d total invoices",
        len(valid_invoices),
        df["Invoice"].nunique(),
    )

    filtered = df[df["Invoice"].isin(valid_invoices)]

    # Pivot to binary matrix
    basket = (
        filtered.groupby(["Invoice", "StockCode"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
        .clip(upper=1)   # binary: 1 if purchased, 0 otherwise
        .astype(bool)
    )
    logger.info(
        "  Basket matrix shape: %d invoices × %d products",
        *basket.shape,
    )
    return basket


def mine_association_rules(
    df: pd.DataFrame,
    min_support: float | None = None,
    min_confidence: float | None = None,
    min_lift: float | None = None,
) -> pd.DataFrame:
    """
    Mine association rules from the transaction data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.
    min_support, min_confidence, min_lift : float, optional
        Override config defaults.

    Returns
    -------
    pd.DataFrame
        Association rules with support, confidence, lift, and product labels.
    """
    from mlxtend.frequent_patterns import apriori, association_rules

    min_support = min_support or AR_CFG["min_support"]
    min_confidence = min_confidence or AR_CFG["min_confidence"]
    min_lift = min_lift or AR_CFG["min_lift"]
    min_txns = AR_CFG["min_transactions"]

    logger.info(
        "Mining rules: support≥%.3f, confidence≥%.2f, lift≥%.2f",
        min_support, min_confidence, min_lift,
    )

    basket = build_basket_matrix(df)
    n_baskets = len(basket)

    # Frequent itemsets via Apriori
    logger.info("Running Apriori (this may take ~1–2 min)…")
    freq_items = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=AR_CFG.get("max_antecedents", 2) + 1,
    )
    logger.info("  Frequent itemsets found: %d", len(freq_items))

    if len(freq_items) == 0:
        logger.warning("No frequent itemsets found. Lower min_support.")
        return pd.DataFrame()

    # Generate rules
    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    rules = rules[
        (rules["confidence"] >= min_confidence) &
        (rules["lift"] >= min_lift)
    ].copy()

    logger.info("  Rules before transaction filter: %d", len(rules))

    # Filter by minimum transaction count
    rules["antecedent_support_count"] = (rules["antecedent support"] * n_baskets).astype(int)
    rules["consequent_support_count"] = (rules["consequent support"] * n_baskets).astype(int)
    rules["rule_transaction_count"] = (rules["support"] * n_baskets).astype(int)
    rules = rules[rules["rule_transaction_count"] >= min_txns].copy()

    logger.info("  Rules after transaction filter (>= %d txns): %d", min_txns, len(rules))

    # ── Add human-readable product labels ────────────────────────────────────
    stock_to_desc = (
        df.groupby("StockCode")["Description"].first().to_dict()
    )

    def _label(itemset) -> str:
        return " + ".join(
            f"{code} ({stock_to_desc.get(code, code)})"
            for code in sorted(itemset)
        )

    rules["antecedent_label"] = rules["antecedents"].apply(_label)
    rules["consequent_label"] = rules["consequents"].apply(_label)

    # ── Add revenue relevance ─────────────────────────────────────────────────
    prod_revenue = df.groupby("StockCode")["Revenue"].sum()

    def _revenue_score(itemset) -> float:
        return float(sum(prod_revenue.get(code, 0) for code in itemset))

    rules["antecedent_revenue"] = rules["antecedents"].apply(_revenue_score)
    rules["consequent_revenue"] = rules["consequents"].apply(_revenue_score)
    rules["combined_revenue"] = rules["antecedent_revenue"] + rules["consequent_revenue"]

    # Final sort: by lift × confidence × revenue relevance
    rules["score"] = rules["lift"] * rules["confidence"] * (
        rules["combined_revenue"] / rules["combined_revenue"].max()
    )
    rules = rules.sort_values("score", ascending=False).reset_index(drop=True)

    # Clean up frozenset columns for serialisation
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: list(x))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: list(x))

    logger.info("Final association rules: %d", len(rules))
    return rules


def format_rules_for_display(rules: pd.DataFrame, top_n: int = 30) -> list[dict]:
    """
    Format rules into a clean list of dicts for the API / recommendation engine.

    Returns
    -------
    list of dicts:
        antecedents, consequents, support, confidence, lift,
        transaction_count, revenue_relevance
    """
    if rules.empty:
        return []

    top = rules.head(top_n)
    out = []
    for _, row in top.iterrows():
        out.append({
            "antecedents": list(row["antecedents"]),
            "antecedent_label": row["antecedent_label"],
            "consequents": list(row["consequents"]),
            "consequent_label": row["consequent_label"],
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
            "transaction_count": int(row["rule_transaction_count"]),
            "combined_revenue": round(float(row["combined_revenue"]), 2),
        })
    return out


def save_relationships(rules: pd.DataFrame) -> None:
    """Save association rules to CSV (excluding frozenset columns)."""
    save_cols = [
        "antecedent_label", "consequent_label",
        "support", "confidence", "lift",
        "rule_transaction_count", "combined_revenue", "score",
        "antecedents_str", "consequents_str",
    ]
    available = [c for c in save_cols if c in rules.columns]
    rules[available].to_csv(PRODUCT_RELATIONSHIPS_PATH, index=False)
    logger.info("Saved product relationships → %s", PRODUCT_RELATIONSHIPS_PATH)
