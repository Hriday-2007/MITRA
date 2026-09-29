def calculate_priority_score(
    estimated_revenue_impact: float,
    confidence: float,
    actionability_score: float,
    max_impact_reference: float = 1000.0,
) -> float:
    """
    Calculate a transparent priority score 0-100.
    
    Weights:
    - Financial Impact (normalized): 50%
    - Confidence (0-1): 30%
    - Actionability (0-1): 20%
    """
    impact_norm = min(1.0, max(0.0, estimated_revenue_impact / max_impact_reference))
    score = (impact_norm * 50) + (confidence * 30) + (actionability_score * 20)
    return round(score, 2)
