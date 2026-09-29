import uuid
from typing import List, Dict
from src.recommendations.models import Recommendation, Evidence
from src.recommendations.scoring import calculate_priority_score

def generate_seasonal_recommendations(seasonality: Dict, total_revenue: float) -> List[Recommendation]:
    recommendations = []
    
    peak = seasonality.get("peak_periods", {})
    busiest_month = peak.get("busiest_month")
    q4_uplift = peak.get("q4_uplift_pct", 0)
    
    if q4_uplift > 15:
        est_impact = total_revenue * (q4_uplift / 100.0) * 0.1 # 10% capture of the uplift
        
        evidence = [
            Evidence("Q4 Uplift", f"{q4_uplift}%", "OBSERVED historical Q4 revenue surge"),
            Evidence("Peak Month", str(busiest_month), "OBSERVED busiest historical month")
        ]
        
        rec = Recommendation(
            id=f"SSN-{uuid.uuid4().hex[:6]}",
            type="SEASONAL_OPPORTUNITY",
            title="Prepare for Q4 Seasonal Peak",
            description=f"Significant seasonal demand observed, with a {q4_uplift}% uplift in Q4.",
            evidence=evidence,
            recommended_action=f"Increase inventory and launch targeted promotions before Month {busiest_month}.",
            estimated_revenue_impact=round(est_impact, 2),
            estimated_profit_impact=round(est_impact * 0.40, 2),
            confidence=0.85,
            priority_score=0.0,
            assumptions=["Assumes capturing 10% of the historical Q4 uplift through preparation (PROJECTED)."],
            source_modules=["src.analytics.seasonality"]
        )
        rec.priority_score = calculate_priority_score(rec.estimated_revenue_impact, rec.confidence, 0.9)
        recommendations.append(rec)
        
    return recommendations
