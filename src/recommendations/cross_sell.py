import uuid
import pandas as pd
from typing import List
from src.recommendations.models import Recommendation, Evidence
from src.recommendations.scoring import calculate_priority_score

def generate_cross_sell_recommendations(rules: List[dict]) -> List[Recommendation]:
    """Generate CROSS_SELL recommendations from association rules."""
    recommendations = []
    
    for rule in rules:
        # Cross-sell requires high confidence but slightly lower lift is okay
        if rule["confidence"] >= 0.5 and rule["transaction_count"] >= 20:
            revenue = rule.get("combined_revenue", 0.0)
            
            # Assumptions
            conversion_rate = 0.10 # assume 10% cross-sell conversion
            est_impact = revenue * conversion_rate
            
            evidence = [
                Evidence(metric="Confidence", value=rule["confidence"], interpretation=f"{rule['confidence']*100:.1f}% probability of buying {rule['consequent_label']}"),
                Evidence(metric="Qualifying baskets", value=rule["transaction_count"], interpretation="OBSERVED transaction count")
            ]
            
            rec = Recommendation(
                id=f"CRS-{uuid.uuid4().hex[:6]}",
                type="CROSS_SELL",
                title=f"Cross-sell {rule['consequent_label']}",
                description=f"Customers purchasing {rule['antecedent_label']} have a high probability of purchasing {rule['consequent_label']}.",
                evidence=evidence,
                recommended_action=f"Recommend {rule['consequent_label']} when a customer purchases {rule['antecedent_label']}.",
                estimated_revenue_impact=round(est_impact, 2),
                estimated_profit_impact=round(est_impact * 0.45, 2), 
                confidence=min(1.0, rule["confidence"]),
                priority_score=0.0,
                assumptions=[f"If {conversion_rate*100}% of {rule['antecedent_label']} buyers accept the cross-sell (ESTIMATED)."],
                source_modules=["src.analytics.basket"]
            )
            
            rec.priority_score = calculate_priority_score(rec.estimated_revenue_impact, rec.confidence, 0.8)
            recommendations.append(rec)
            
    return recommendations
