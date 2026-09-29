import uuid
import pandas as pd
from typing import List
from src.recommendations.models import Recommendation, Evidence
from src.recommendations.scoring import calculate_priority_score

def generate_customer_recommendations(rfm_segments: pd.DataFrame) -> List[Recommendation]:
    recommendations = []
    
    for _, row in rfm_segments.iterrows():
        segment = row.get('segment', '')
        if segment in ['At Risk', 'Need Attention', 'Hibernating']:
            n_customers = row.get('n_customers', 0)
            segment_revenue = row.get('total_revenue', 0.0)
            
            if segment_revenue > 0 and n_customers > 10:
                est_impact = segment_revenue * 0.05 # Reclaim 5% of segment revenue
                
                evidence = [
                    Evidence("Segment", segment, "OBSERVED RFM categorization"),
                    Evidence("Customers in Segment", n_customers, "OBSERVED count"),
                    Evidence("Historical Segment Revenue", round(segment_revenue, 2), "OBSERVED revenue from these customers")
                ]
                
                rec = Recommendation(
                    id=f"RET-{uuid.uuid4().hex[:6]}",
                    type="CUSTOMER_RETENTION",
                    title=f"Retain {segment} customers",
                    description=f"High-value {segment.lower()} customers are declining in frequency.",
                    evidence=evidence,
                    recommended_action=f"Launch a targeted retention offer for {segment.lower()} customers.",
                    estimated_revenue_impact=round(est_impact, 2),
                    estimated_profit_impact=round(est_impact * 0.30, 2), # 30% margin after promo costs
                    confidence=0.75,
                    priority_score=0.0,
                    assumptions=["Assumes retention campaign recovers 5% of segment's historical revenue (PROJECTED)."],
                    source_modules=["src.analytics.rfm"]
                )
                rec.priority_score = calculate_priority_score(rec.estimated_revenue_impact, rec.confidence, 0.85)
                recommendations.append(rec)
                
    return recommendations
