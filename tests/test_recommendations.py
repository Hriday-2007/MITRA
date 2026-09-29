import pytest
import pandas as pd
from src.recommendations.engine import generate_all_recommendations
from src.recommendations.bundle import generate_bundle_recommendations
from src.recommendations.cross_sell import generate_cross_sell_recommendations
from src.recommendations.inventory import generate_inventory_recommendations
from src.recommendations.customers import generate_customer_recommendations
from src.recommendations.seasonality import generate_seasonal_recommendations
from src.recommendations.scoring import calculate_priority_score

def test_scoring():
    score = calculate_priority_score(estimated_revenue_impact=500.0, confidence=0.8, actionability_score=0.9, max_impact_reference=1000.0)
    # impact: 500/1000 = 0.5 * 50 = 25
    # confidence: 0.8 * 30 = 24
    # actionability: 0.9 * 20 = 18
    # total = 25 + 24 + 18 = 67
    assert score == 67.0

def test_bundle_recommendations():
    strong_rule = {
        "antecedent_label": "A", "consequent_label": "B",
        "lift": 2.5, "confidence": 0.4, "transaction_count": 50,
        "combined_revenue": 1000.0
    }
    weak_rule = {
        "antecedent_label": "C", "consequent_label": "D",
        "lift": 1.5, "confidence": 0.2, "transaction_count": 10,
        "combined_revenue": 100.0
    }
    
    # Strong bundle accepted
    recs = generate_bundle_recommendations([strong_rule])
    assert len(recs) == 1
    assert recs[0].type == "PRODUCT_BUNDLE"
    assert recs[0].estimated_revenue_impact == 50.0 # 5% conversion of 1000
    
    # Weak bundle rejected
    recs = generate_bundle_recommendations([weak_rule])
    assert len(recs) == 0

def test_cross_sell_recommendations():
    rule = {
        "antecedent_label": "A", "consequent_label": "B",
        "lift": 1.2, "confidence": 0.6, "transaction_count": 25,
        "combined_revenue": 2000.0
    }
    recs = generate_cross_sell_recommendations([rule])
    assert len(recs) == 1
    assert recs[0].type == "CROSS_SELL"

def test_inventory_recommendations():
    inv_df = pd.DataFrame([{
        "StockCode": "P1", "stockout_risk": "HIGH", "velocity_label": "FAST", "avg_daily_units": 10, "modelled_stock_days": 2
    }, {
        "StockCode": "P2", "stockout_risk": "LOW", "velocity_label": "SLOW", "modelled_stock_days": 100
    }])
    prod_economics = pd.DataFrame([
        {"StockCode": "P1", "total_revenue": 5000, "gross_margin_pct": 50},
        {"StockCode": "P2", "total_revenue": 500, "gross_margin_pct": 40}
    ])
    
    recs = generate_inventory_recommendations(inv_df, prod_economics)
    assert len(recs) == 2
    types = [r.type for r in recs]
    assert "INVENTORY_REPLENISHMENT" in types
    assert "SLOW_MOVING_STOCK" in types

def test_customer_recommendations():
    rfm = pd.DataFrame([
        {"segment": "At Risk", "n_customers": 50, "total_revenue": 10000},
        {"segment": "Champions", "n_customers": 10, "total_revenue": 5000}
    ])
    recs = generate_customer_recommendations(rfm)
    assert len(recs) == 1
    assert recs[0].type == "CUSTOMER_RETENTION"
    assert recs[0].estimated_revenue_impact == 500.0

def test_seasonal_recommendations():
    seasonality = {
        "peak_periods": {
            "busiest_month": 11,
            "q4_uplift_pct": 25
        }
    }
    recs = generate_seasonal_recommendations(seasonality, total_revenue=100000)
    assert len(recs) == 1
    assert recs[0].type == "SEASONAL_OPPORTUNITY"

def test_missing_insufficient_data():
    recs = generate_all_recommendations([], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, 0)
    assert len(recs) == 0
