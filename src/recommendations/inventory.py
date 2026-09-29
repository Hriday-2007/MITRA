import uuid
import pandas as pd
from typing import List
from src.recommendations.models import Recommendation, Evidence
from src.recommendations.scoring import calculate_priority_score

def generate_inventory_recommendations(inv_df: pd.DataFrame, prod_economics: pd.DataFrame) -> List[Recommendation]:
    recommendations = []
    
    # Merge economics for revenue impact estimation
    if prod_economics is not None and not prod_economics.empty:
        merge_cols = ['StockCode', 'total_revenue', 'gross_margin_pct']
        available = [c for c in merge_cols if c in prod_economics.columns] + ['StockCode']
        merged = inv_df.merge(prod_economics[list(set(available))], on='StockCode', how='left')
    else:
        merged = inv_df.copy()
        merged['total_revenue'] = 0.0
        merged['gross_margin_pct'] = 0.45
    
    for _, row in merged.iterrows():
        # INVENTORY_REPLENISHMENT
        if row.get('stockout_risk') == 'HIGH' and row.get('velocity_label') in ['FAST', 'STEADY']:
            revenue = row.get('total_revenue', 0.0)
            est_impact = revenue * 0.10 # Assuming 10% revenue lost due to stockouts
            
            evidence = [
                Evidence("Modeled Stockout Risk", "HIGH", "High risk of stockout based on modeled inventory"),
                Evidence("Avg Daily Units", round(row.get('avg_daily_units', 0.0), 2), "PROJECTED average daily sales"),
                Evidence("Modeled Days Remaining", round(row.get('modelled_stock_days', 0.0), 1), "ESTIMATED days of stock remaining")
            ]
            
            rec = Recommendation(
                id=f"INV-{uuid.uuid4().hex[:6]}",
                type="INVENTORY_REPLENISHMENT",
                title=f"Replenish {row['StockCode']}",
                description=f"Product {row['StockCode']} has increasing demand and low estimated inventory.",
                evidence=evidence,
                recommended_action=f"Increase stock of {row['StockCode']}.",
                estimated_revenue_impact=round(est_impact, 2),
                estimated_profit_impact=round(est_impact * (row.get('gross_margin_pct', 45)/100.0), 2),
                confidence=0.7, # Moderate confidence as inventory is modeled
                priority_score=0.0,
                assumptions=["Inventory is MODELED because original dataset lacks stock counts.", "Assumes 10% of revenue is lost without replenishment (ESTIMATED)."],
                source_modules=["src.ml.inventory", "src.ml.forecasting"]
            )
            rec.priority_score = calculate_priority_score(rec.estimated_revenue_impact, rec.confidence, 0.9)
            recommendations.append(rec)
            
        # SLOW_MOVING_STOCK
        elif row.get('velocity_label') == 'SLOW' and row.get('modelled_stock_days', 0.0) > 90:
            revenue = row.get('total_revenue', 0.0)
            est_impact = revenue * 0.05
            
            evidence = [
                Evidence("Velocity", "SLOW", "OBSERVED sales velocity is slow"),
                Evidence("Modeled Days Remaining", round(row.get('modelled_stock_days', 0.0), 1), "ESTIMATED days of stock remaining")
            ]
            
            rec = Recommendation(
                id=f"SLW-{uuid.uuid4().hex[:6]}",
                type="SLOW_MOVING_STOCK",
                title=f"Clear slow stock: {row['StockCode']}",
                description=f"Product {row['StockCode']} has low velocity and high modeled inventory.",
                evidence=evidence,
                recommended_action=f"Bundle or promote {row['StockCode']} to clear stock. Reduce future purchasing.",
                estimated_revenue_impact=round(est_impact, 2),
                estimated_profit_impact=0.0, # Clearance often neutral or loss
                confidence=0.8,
                priority_score=0.0,
                assumptions=["Inventory is MODELED.", "Assumes clearing stock frees up 5% of product revenue capital (ESTIMATED)."],
                source_modules=["src.ml.inventory"]
            )
            rec.priority_score = calculate_priority_score(rec.estimated_revenue_impact, rec.confidence, 0.8)
            recommendations.append(rec)
            
    return recommendations
