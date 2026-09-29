from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ProductChange:
    product: str
    volume_change_pct: float = 0.0
    price_change_pct: float = 0.0
    discount_pct: float = 0.0

@dataclass
class ScenarioInput:
    product_changes: List[ProductChange]

@dataclass
class Metrics:
    revenue: float
    cost: float
    profit: float
    margin_pct: float

    def to_dict(self):
        return {
            "revenue": round(self.revenue, 2),
            "cost": round(self.cost, 2),
            "profit": round(self.profit, 2),
            "margin_pct": round(self.margin_pct, 2)
        }

@dataclass
class ScenarioOutput:
    baseline: Metrics
    scenario: Metrics
    delta: Metrics
    assumptions: List[str]
    confidence: float

    def to_dict(self):
        return {
            "baseline": self.baseline.to_dict(),
            "scenario": self.scenario.to_dict(),
            "delta": self.delta.to_dict(),
            "assumptions": self.assumptions,
            "confidence": self.confidence
        }

@dataclass
class StrategyCandidate:
    description: str
    projected_profit_impact: float

    def to_dict(self):
        return {
            "description": self.description,
            "projected_profit_impact": round(self.projected_profit_impact, 2)
        }

@dataclass
class OptimizationResult:
    target_profit: float
    achieved_profit: float
    candidates: List[StrategyCandidate]

    def to_dict(self):
        return {
            "target_profit": round(self.target_profit, 2),
            "achieved_profit": round(self.achieved_profit, 2),
            "candidates": [c.to_dict() for c in self.candidates]
        }
