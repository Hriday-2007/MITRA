"""
MITRA — Master Pipeline Runner
Runs all phases of the MITRA data pipeline in order.

Usage:
    python run_pipeline.py

This script:
1. Loads raw dataset
2. Generates data quality report
3. Cleans data
4. Computes core business metrics
5. Computes product intelligence
6. Mines association rules
7. Computes RFM customer segments
8. Computes seasonality
9. Computes synthetic economics
10. Builds demo inventory model
11. Runs demand forecast
12. Assembles merchant business profile

All intermediate outputs are saved to /data for reuse.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from config.settings import DATA_DIR

logger = get_logger("mitra.pipeline")


def run_pipeline(skip_basket: bool = False, skip_forecast: bool = False) -> dict:
    """
    Run the full MITRA analytics pipeline.

    Parameters
    ----------
    skip_basket : bool
        Skip market basket analysis (slow for large datasets).
    skip_forecast : bool
        Skip demand forecasting.

    Returns
    -------
    dict: summary of all results
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("MITRA PIPELINE STARTING")
    logger.info("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Load raw data ────────────────────────────────────────────────
    logger.info("[Phase 1] Loading raw dataset…")
    from src.data.loader import load_raw, inspect_raw, print_report
    df_raw = load_raw()
    raw_report = inspect_raw(df_raw)
    print_report(raw_report)

    # Save quality report
    report_path = DATA_DIR / "data_quality_report.json"
    with open(report_path, "w") as f:
        # Convert non-serialisable items
        serialisable = {
            k: (v if not hasattr(v, 'items') else dict(v))
            for k, v in raw_report.items()
        }
        json.dump(serialisable, f, indent=2, default=str)
    logger.info("Data quality report → %s", report_path)

    # ── Phase 2: Clean data ───────────────────────────────────────────────────
    logger.info("[Phase 2] Cleaning dataset…")
    from src.data.cleaner import clean, save_clean, print_cleaning_log
    df_clean, cleaning_log = clean(df_raw, verbose=True)
    print_cleaning_log(cleaning_log)
    save_clean(df_clean)

    # ── Phase 3: Core business metrics ───────────────────────────────────────
    logger.info("[Phase 3] Computing core business metrics…")
    from src.analytics.metrics import (
        calculate_summary,
        calculate_product_performance,
        calculate_customer_metrics,
        calculate_time_metrics,
        calculate_country_metrics,
        calculate_repeat_rate,
    )
    summary = calculate_summary(df_clean)
    logger.info(
        "  Revenue: £{:,.0f} | Orders: {:,} | Customers: {:,} | Products: {:,}".format(
            summary["total_revenue"], summary["total_orders"],
            summary["total_customers"], summary["total_products"],
        )
    )

    # ── Phase 4: Product Intelligence ────────────────────────────────────────
    logger.info("[Phase 4] Computing product intelligence…")
    from src.analytics.product_intel import (
        calculate_product_intelligence,
        classify_products,
        save_product_intelligence,
    )
    prod_intel = calculate_product_intelligence(df_clean)
    prod_classifications = classify_products(prod_intel)

    # ── Phase 5: Market Basket Analysis ──────────────────────────────────────
    rules_formatted = []
    import pandas as pd
    rules_df = pd.DataFrame()

    if not skip_basket:
        logger.info("[Phase 5] Mining association rules…")
        from src.analytics.basket import (
            mine_association_rules,
            format_rules_for_display,
            save_relationships,
        )
        try:
            rules_df = mine_association_rules(df_clean)
            if not rules_df.empty:
                rules_formatted = format_rules_for_display(rules_df, top_n=30)
                save_relationships(rules_df)
                logger.info("  Found %d association rules.", len(rules_df))
            else:
                logger.warning("  No association rules found.")
        except Exception as e:
            logger.error("  Basket analysis failed: %s", e)
    else:
        logger.info("[Phase 5] Skipping basket analysis.")

    # ── Phase 6: RFM Customer Segmentation ───────────────────────────────────
    logger.info("[Phase 6] Computing RFM customer segments…")
    from src.analytics.rfm import calculate_rfm, summarise_segments, save_segments
    rfm = calculate_rfm(df_clean)
    rfm_segments = summarise_segments(rfm)
    save_segments(rfm)
    logger.info(
        "  Segments:\n%s",
        rfm_segments[["segment", "n_customers", "revenue_pct"]].to_string(index=False),
    )

    # ── Phase 7: Seasonality ──────────────────────────────────────────────────
    logger.info("[Phase 7] Computing seasonality analysis…")
    from src.analytics.seasonality import calculate_seasonality, save_seasonality
    season = calculate_seasonality(df_clean)
    save_seasonality(season)
    peak = season["peak_periods"]
    logger.info(
        "  Busiest day: %s | Peak month: %s | Q4 uplift: %.1f%%",
        peak["busiest_day"], peak["busiest_month"], peak.get("q4_uplift_pct", 0),
    )

    # ── Phase 8: Synthetic Economics ─────────────────────────────────────────
    logger.info("[Phase 8] Computing synthetic COGS model…")
    from src.analytics.economics import (
        calculate_product_economics,
        calculate_portfolio_economics,
        save_product_economics,
    )
    prod_economics = calculate_product_economics(prod_intel)
    portfolio_economics = calculate_portfolio_economics(prod_economics)
    save_product_economics(prod_economics)
    save_product_intelligence(prod_economics)   # overwrite with economics columns
    logger.info(
        "  Portfolio gross margin (synthetic): %.1f%%",
        portfolio_economics["portfolio_gross_margin_pct"],
    )

    # ── Phase 9: Inventory Model ──────────────────────────────────────────────
    logger.info("[Phase 9] Building demo inventory model…")
    from src.ml.inventory import build_inventory_model, get_inventory_risks
    inv_df = build_inventory_model(df_clean)
    inv_risks = get_inventory_risks(inv_df)
    inv_path = DATA_DIR / "inventory_model.csv"
    inv_df.to_csv(inv_path, index=False)
    logger.info(
        "  High stockout risk products: %d",
        int((inv_df["stockout_risk"] == "HIGH").sum()),
    )

    # ── Phase 10: Demand Forecast ─────────────────────────────────────────────
    forecast = {}
    if not skip_forecast:
        logger.info("[Phase 10] Running demand forecasting…")
        from src.ml.forecasting import forecast_total_sales
        try:
            forecast = forecast_total_sales(df_clean)
            logger.info(
                "  Best model: %s | MAPE: %s",
                forecast.get("best_model"),
                forecast.get("mape"),
            )
        except Exception as e:
            logger.error("  Forecasting failed: %s", e)
            forecast = {"error": str(e)}
    else:
        logger.info("[Phase 10] Skipping forecasting.")

    # ── Phase 11: Recommendation Engine ───────────────────────────────────────
    logger.info("[Phase 11] Generating recommendations…")
    from src.recommendations.engine import generate_all_recommendations, save_recommendations
    recommendations = generate_all_recommendations(
        rules=rules_formatted,
        inv_df=inv_df,
        prod_economics=prod_economics,
        rfm_segments=rfm_segments,
        seasonality=season,
        total_revenue=summary["total_revenue"]
    )
    save_recommendations(recommendations)

    # ── Phase 12: Business Brain ──────────────────────────────────────────────
    logger.info("[Phase 12] Assembling merchant business profile…")
    from src.agent.brain import build_business_profile, save_business_profile
    profile = build_business_profile(
        df_clean=df_clean,
        summary=summary,
        prod_intel=prod_economics,
        prod_classifications=prod_classifications,
        rules=rules_formatted,
        rfm=rfm,
        rfm_segments=rfm_segments,
        season=season,
        forecast=forecast,
        inv_risks=inv_risks,
        portfolio_economics=portfolio_economics,
        recommendations=recommendations,
    )
    save_business_profile(profile)

    elapsed = round(time.time() - t0, 1)
    logger.info("=" * 60)
    logger.info("MITRA PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("=" * 60)

    # ── Final Summary Print ───────────────────────────────────────────────────
    _print_final_summary(
        raw_report, cleaning_log, summary, prod_intel,
        prod_classifications, rules_formatted, rfm_segments,
        season, forecast, recommendations, elapsed
    )

    return {
        "summary": summary,
        "cleaning_log": cleaning_log,
        "raw_report": raw_report,
        "prod_intel": prod_intel,
        "rules": rules_formatted,
        "rfm_segments": rfm_segments,
        "season": season,
        "forecast": forecast,
        "recommendations": recommendations,
        "profile": profile,
    }


def _print_final_summary(
    raw_report, cleaning_log, summary, prod_intel,
    prod_classifications, rules, rfm_segments, season, forecast, recommendations, elapsed
):
    """Print the final results summary."""
    import pandas as pd

    print("\n" + "=" * 70)
    print("  MITRA — PIPELINE RESULTS SUMMARY")
    print("=" * 70)

    print("\n📊 DATASET")
    print(f"  Raw rows              : {raw_report['n_rows']:,}")
    print(f"  Cleaned rows          : {cleaning_log['8_final_rows']:,}")
    print(f"  Rows removed          : {cleaning_log['total_removed']:,}")
    print(f"  Rows retained         : {cleaning_log['pct_retained']}%")
    print(f"  Date range            : {summary['analysis_period']['start']} → {summary['analysis_period']['end']}")
    print(f"  Span                  : {summary['analysis_period']['days']} days")

    print("\n💰 BUSINESS METRICS")
    print(f"  Total Revenue         : £{summary['total_revenue']:,.2f}")
    print(f"  Total Orders          : {summary['total_orders']:,}")
    print(f"  Total Units Sold      : {summary['total_units_sold']:,}")
    print(f"  Unique Customers      : {summary['total_customers']:,}")
    print(f"  Unique Products       : {summary['total_products']:,}")
    print(f"  Average Order Value   : £{summary['average_order_value']:,.2f}")
    print(f"  Average Basket Size   : {summary['average_basket_size']} items")
    print(f"  Customer Repeat Rate  : {summary['repeat_rate_pct']}%")
    print(f"  MoM Revenue Growth    : {summary['revenue_growth_mom_pct']}%")
    print(f"  Top Country           : {summary['top_country']}")

    print("\n🏆 TOP 10 PRODUCTS BY REVENUE")
    top10 = prod_intel.head(10)
    for _, row in top10.iterrows():
        gm = f" | GM: {row.get('gross_margin_pct', '?')}%" if 'gross_margin_pct' in row else ""
        print(f"  {row.get('rank', '?'):>3}. [{row['StockCode']}] {row['Description'][:40]:<40} "
              f"£{row['total_revenue']:>10,.0f}{gm}")

    print("\n🔗 STRONGEST PRODUCT RELATIONSHIPS")
    for i, rule in enumerate(rules[:5], 1):
        print(f"  {i}. {rule['antecedent_label'][:35]} → {rule['consequent_label'][:35]}")
        print(f"     Lift: {rule['lift']:.2f} | Conf: {rule['confidence']:.2f} | "
              f"Txns: {rule['transaction_count']}")

    print("\n👥 CUSTOMER SEGMENTS")
    for _, seg in rfm_segments.iterrows():
        print(f"  {seg['segment']:<25}: {seg['n_customers']:>5,} customers | "
              f"Rev: £{seg['total_revenue']:>10,.0f} ({seg['revenue_pct']}%)")

    print("\n📅 STRONGEST SEASONAL PATTERNS")
    peak = season.get("peak_periods", {})
    print(f"  Busiest Day           : {peak.get('busiest_day', 'N/A')}")
    print(f"  Peak Hour             : {peak.get('peak_hour', 'N/A')}:00")
    print(f"  Busiest Month         : Month {peak.get('busiest_month', 'N/A')}")
    print(f"  Q4 Revenue Uplift     : {peak.get('q4_uplift_pct', 'N/A')}%")

    print("\n📈 DEMAND FORECAST")
    if "best_model" in forecast:
        print(f"  Best Model            : {forecast.get('best_model')}")
        mape = forecast.get("mape", {})
        for m, v in mape.items():
            print(f"    {m:<30}: MAPE = {v:.2%}")
    else:
        print("  Forecast not available.")

    print("\n🏗️  PRODUCT LIFECYCLE")
    for label, codes in prod_classifications.items():
        print(f"  {label:<25}: {len(codes):>4} products")

    print("\n💡 RECOMMENDATIONS")
    print(f"  Total Generated       : {len(recommendations)}")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. [{rec['type']}] {rec['title']} (Score: {rec['priority_score']})")

    print("\n⚠️  IMPORTANT LIMITATIONS")
    print("  1. COGS is SYNTHETIC — not real merchant cost data.")
    print("  2. Inventory levels are MODELLED — not real stock counts.")
    print("  3. Demand forecasts are MODEL PROJECTIONS — not guarantees.")
    print("  4. CustomerID missing for ~25% of rows — those excluded from RFM.")
    print("  5. Dataset is historical (2009–2011) — patterns may not reflect today.")

    print("\n📁 FILES CREATED")
    from config.settings import (
        CLEAN_CSV_PATH, CLEAN_PARQUET_PATH,
        PRODUCT_PERFORMANCE_PATH, PRODUCT_RELATIONSHIPS_PATH,
        CUSTOMER_SEGMENTS_PATH, SEASONALITY_FEATURES_PATH,
        PRODUCT_ECONOMICS_PATH, BUSINESS_PROFILE_PATH, DATA_DIR
    )
    files = [
        CLEAN_CSV_PATH, CLEAN_PARQUET_PATH,
        PRODUCT_PERFORMANCE_PATH, PRODUCT_RELATIONSHIPS_PATH,
        CUSTOMER_SEGMENTS_PATH, SEASONALITY_FEATURES_PATH,
        PRODUCT_ECONOMICS_PATH, BUSINESS_PROFILE_PATH,
        DATA_DIR / "inventory_model.csv",
        DATA_DIR / "data_quality_report.json",
    ]
    for f in files:
        exists = "✓" if Path(f).exists() else "✗"
        print(f"  {exists} {f}")

    print(f"\n⏱️  Total pipeline time: {elapsed}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MITRA Analytics Pipeline")
    parser.add_argument("--skip-basket", action="store_true",
                        help="Skip market basket analysis (faster)")
    parser.add_argument("--skip-forecast", action="store_true",
                        help="Skip demand forecasting (faster)")
    args = parser.parse_args()

    run_pipeline(
        skip_basket=args.skip_basket,
        skip_forecast=args.skip_forecast,
    )
