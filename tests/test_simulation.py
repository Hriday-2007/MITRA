import pytest
import pandas as pd
from src.simulation.models import ScenarioInput, ProductChange
from src.simulation.engine import simulate_scenario
from src.simulation.optimizer import optimize_profit_goal

@pytest.fixture
def sample_economics():
    return pd.DataFrame([
        {
            "StockCode": "P1",
            "total_revenue": 1000.0,
            "estimated_cogs": 400.0,
            "gross_profit": 600.0,
            "gross_margin_pct": 60.0,
            "total_units": 100
        },
        {
            "StockCode": "P2",
            "total_revenue": 2000.0,
            "estimated_cogs": 1500.0,
            "gross_profit": 500.0,
            "gross_margin_pct": 25.0,
            "total_units": 200
        }
    ])

def test_baseline_matches(sample_economics):
    input_data = ScenarioInput(product_changes=[])
    result = simulate_scenario(input_data, sample_economics)
    
    assert result.baseline.revenue == 3000.0
    assert result.baseline.cost == 1900.0
    assert result.baseline.profit == 1100.0
    
    assert result.scenario.revenue == 3000.0
    assert result.delta.revenue == 0.0

def test_volume_increase(sample_economics):
    # +20% volume on P1
    input_data = ScenarioInput(product_changes=[
        ProductChange(product="P1", volume_change_pct=20.0)
    ])
    result = simulate_scenario(input_data, sample_economics)
    
    # Old P1: Rev 1000, Cost 400. New: Rev 1200, Cost 480
    assert result.scenario.revenue == 3200.0
    assert result.scenario.cost == 1980.0
    assert result.scenario.profit == 1220.0
    assert result.delta.profit == 120.0

def test_price_increase(sample_economics):
    # +5% price on P1
    input_data = ScenarioInput(product_changes=[
        ProductChange(product="P1", price_change_pct=5.0)
    ])
    result = simulate_scenario(input_data, sample_economics)
    
    # Old P1: Rev 1000. New: Rev 1050, Cost 400
    assert result.scenario.revenue == 3050.0
    assert result.scenario.cost == 1900.0 # Cost doesn't change
    assert result.scenario.profit == 1150.0
    assert result.delta.profit == 50.0

def test_discount(sample_economics):
    # 5% discount on P2
    input_data = ScenarioInput(product_changes=[
        ProductChange(product="P2", discount_pct=5.0)
    ])
    result = simulate_scenario(input_data, sample_economics)
    
    # Old P2: Rev 2000. New: Rev 1900, Cost 1500
    assert result.scenario.revenue == 2900.0
    assert result.scenario.cost == 1900.0 # Cost doesn't change
    assert result.scenario.profit == 1000.0
    assert result.delta.profit == -100.0

def test_optimize_profit_goal(sample_economics):
    result = optimize_profit_goal(50000.0, sample_economics)
    assert result.target_profit == 50000.0
    assert result.achieved_profit == 50000.0
    assert len(result.candidates) > 0
