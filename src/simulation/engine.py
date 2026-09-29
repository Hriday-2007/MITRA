import pandas as pd
from typing import Dict, List
from src.simulation.models import ScenarioInput, ScenarioOutput, Metrics

def simulate_scenario(input_data: ScenarioInput, prod_economics: pd.DataFrame) -> ScenarioOutput:
    """
    Simulate quantitative changes to product volumes and pricing.
    """
    # Calculate baseline
    baseline_revenue = prod_economics['total_revenue'].sum()
    baseline_cost = prod_economics['estimated_cogs'].sum()
    baseline_profit = prod_economics['gross_profit'].sum()
    baseline_margin = (baseline_profit / baseline_revenue * 100) if baseline_revenue > 0 else 0.0

    baseline_metrics = Metrics(baseline_revenue, baseline_cost, baseline_profit, baseline_margin)

    # Calculate scenario
    scenario_revenue = baseline_revenue
    scenario_cost = baseline_cost

    assumptions = ["COGS is assumed constant per unit (MODEL PROJECTION).", "Price elasticity is ignored unless explicitly modeled."]
    
    changes_by_product = {c.product: c for c in input_data.product_changes}
    
    for _, row in prod_economics.iterrows():
        stock_code = str(row.get('StockCode', ''))
        change = changes_by_product.get(stock_code)
        
        if change:
            orig_rev = row.get('total_revenue', 0.0)
            orig_cost = row.get('estimated_cogs', 0.0)
            orig_units = row.get('total_units', 1)
            
            if orig_units == 0:
                continue
                
            orig_price_avg = orig_rev / orig_units
            unit_cost = orig_cost / orig_units
            
            # Apply changes
            new_units = orig_units * (1.0 + change.volume_change_pct / 100.0)
            new_price = orig_price_avg * (1.0 + change.price_change_pct / 100.0) * (1.0 - change.discount_pct / 100.0)
            
            new_rev = new_units * new_price
            new_cost = new_units * unit_cost
            
            # Subtract old from total, add new
            scenario_revenue = scenario_revenue - orig_rev + new_rev
            scenario_cost = scenario_cost - orig_cost + new_cost

    scenario_profit = scenario_revenue - scenario_cost
    scenario_margin = (scenario_profit / scenario_revenue * 100) if scenario_revenue > 0 else 0.0
    scenario_metrics = Metrics(scenario_revenue, scenario_cost, scenario_profit, scenario_margin)

    # Calculate delta
    delta_metrics = Metrics(
        scenario_revenue - baseline_revenue,
        scenario_cost - baseline_cost,
        scenario_profit - baseline_profit,
        scenario_margin - baseline_margin
    )

    return ScenarioOutput(
        baseline=baseline_metrics,
        scenario=scenario_metrics,
        delta=delta_metrics,
        assumptions=assumptions,
        confidence=0.7 # Moderate confidence, assumes linear cost scaling
    )
