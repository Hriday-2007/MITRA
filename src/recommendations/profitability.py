import uuid
import pandas as pd
from typing import List
from src.recommendations.models import Recommendation, Evidence
from src.recommendations.scoring import calculate_priority_score

def generate_profit_recommendations(prod_economics: pd.DataFrame) -> List[Recommendation]:
    recommendations = []
    
    for _, row in prod_economics.iterrows():
        margin = row.get('gross_margin_pct', 0.0)
        revenue = row.get('total_revenue', 0.0)
        
        # PROFIT_OPPORTUNITY
        if margin > 60 and revenue > 500:
            est_impact = revenue * 0.15 # 15% boost
            
            evidence = [
                Evidence("Gross Margin", f"{margin}%", "ESTIMATED synthetic margin"),
                Evidence("Revenue", round(revenue, 2), "OBSERVED historical revenue")
            ]
            
            rec = Recommendation(
                id=f"PRF-{uuid.uuid4().hex[:6]}",
                type="PROFIT_OPPORTUNITY",
                title=f"Boost high-margin product: {row['StockCode']}",
                description=f"Product {row['StockCode']} has excellent modeled margins and solid revenue.",
                evidence=evidence,
                recommended_action=f"Feature {row['StockCode']} prominently in marketing to improve overall profit mix.",
                estimated_revenue_impact=round(est_impact, 2),
                estimated_profit_impact=round(est_impact * (margin/100.0), 2),
                confidence=0.6,
                priority_score=0.0,
                assumptions=["COGS and Margin are SYNTHETIC/MODELED.", "Assumes 15% revenue growth with promotion (PROJECTED)."],
                source_modules=["src.analytics.economics"]
            )
            rec.priority_score = calculate_priority_score(rec.estimated_revenue_impact, rec.confidence, 0.8)
            recommendations.append(rec)
            
    return recommendations
