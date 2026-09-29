import uuid
import pandas as pd
from typing import List
from src.recommendations.models import Recommendation, Evidence
from src.recommendations.scoring import calculate_priority_score

def generate_bundle_recommendations(rules: List[dict]) -> List[Recommendation]:
    """Generate PRODUCT_BUNDLE recommendations from association rules."""
    recommendations = []
    
    for rule in rules:
        # Sufficient evidence threshold
        if rule["lift"] >= 2.0 and rule["confidence"] >= 0.3 and rule["transaction_count"] >= 30:
            revenue = rule.get("combined_revenue", 0.0)
            
            # Assumptions
            conversion_rate = 0.05 # assume 5% of related transactions convert to bundle
            est_impact = revenue * conversion_rate
            
            evidence = [
                Evidence(metric="Lift", value=rule["lift"], interpretation="Items are frequently bought together"),
                Evidence(metric="Confidence", value=rule["confidence"], interpretation=f"{rule['confidence']*100:.1f}% probability of consequent purchase"),
                Evidence(metric="Qualifying baskets", value=rule["transaction_count"], interpretation="OBSERVED co-purchases")
            ]
            
            rec = Recommendation(
                id=f"BNDL-{uuid.uuid4().hex[:6]}",
                type="PRODUCT_BUNDLE",
                title=f"Bundle: {rule['antecedent_label']} + {rule['consequent_label']}",
                description=f"Customers purchasing {rule['antecedent_label']} frequently purchase {rule['consequent_label']}.",
                evidence=evidence,
                recommended_action=f"Create a specific product bundle combining {rule['antecedent_label']} and {rule['consequent_label']}.",
                estimated_revenue_impact=round(est_impact, 2),
                estimated_profit_impact=round(est_impact * 0.45, 2), # synthetic margin assumption
                confidence=min(1.0, rule["confidence"] + 0.2), # scaled confidence
                priority_score=0.0, # calculated later
                assumptions=[f"If {conversion_rate*100}% of qualifying purchases adopt the bundle (ESTIMATED)."],
                source_modules=["src.analytics.basket"]
            )
            
            rec.priority_score = calculate_priority_score(rec.estimated_revenue_impact, rec.confidence, 0.9)
            recommendations.append(rec)
            
    return recommendations
