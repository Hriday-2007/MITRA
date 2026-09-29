from dataclasses import dataclass
from typing import List, Any

@dataclass
class Evidence:
    metric: str
    value: Any
    interpretation: str

    def to_dict(self):
        return {
            "metric": self.metric,
            "value": self.value,
            "interpretation": self.interpretation,
        }

@dataclass
class Recommendation:
    id: str
    type: str
    title: str
    description: str
    evidence: List[Evidence]
    recommended_action: str
    estimated_revenue_impact: float
    estimated_profit_impact: float
    confidence: float
    priority_score: float
    assumptions: List[str]
    source_modules: List[str]

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "evidence": [e.to_dict() for e in self.evidence],
            "recommended_action": self.recommended_action,
            "estimated_revenue_impact": self.estimated_revenue_impact,
            "estimated_profit_impact": self.estimated_profit_impact,
            "confidence": self.confidence,
            "priority_score": self.priority_score,
            "assumptions": self.assumptions,
            "source_modules": self.source_modules,
        }
