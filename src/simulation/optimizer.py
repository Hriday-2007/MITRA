import pandas as pd
from typing import List
from src.simulation.models import OptimizationResult, StrategyCandidate

def optimize_profit_goal(target_profit: float, prod_economics: pd.DataFrame) -> OptimizationResult:
    """
    Search combinations of realistic business levers to hit a profit target.
    Returns candidate strategies.
    """
    candidates: List[StrategyCandidate] = []
    achieved_profit = 0.0
    
    if prod_economics.empty:
        return OptimizationResult(target_profit, achieved_profit, candidates)
        
    # Sort products by margin and revenue to find good levers
    # Filter for products with high margin (> 50%) and decent revenue
    high_margin_prods = prod_economics[
        (prod_economics['gross_margin_pct'] > 50.0) & (prod_economics['total_revenue'] > 100.0)
    ].sort_values(by='gross_profit', ascending=False)
    
    # 1. Product mix change / Volume increases on top 5 high-margin products
    for _, row in high_margin_prods.head(5).iterrows():
        if achieved_profit >= target_profit:
            break
            
        stock_code = row['StockCode']
        orig_profit = row['gross_profit']
        
        # Lever: +20% volume on high margin product
        proj_impact = orig_profit * 0.20
        candidates.append(StrategyCandidate(
            description=f"Increase {stock_code} volume by 20% (Marketing/Promo)",
            projected_profit_impact=proj_impact
        ))
        achieved_profit += proj_impact
        
    # 2. Bundle adoptions (simulated top products bundled)
    if achieved_profit < target_profit and len(high_margin_prods) >= 2:
        prod_a = high_margin_prods.iloc[0]
        prod_b = high_margin_prods.iloc[1]
        proj_impact = (prod_a['gross_profit'] + prod_b['gross_profit']) * 0.15 # 15% bump
        candidates.append(StrategyCandidate(
            description=f"Bundle {prod_a['StockCode']} + {prod_b['StockCode']}",
            projected_profit_impact=proj_impact
        ))
        achieved_profit += proj_impact
        
    # 3. Inventory reduction / clearance (reduce holding cost proxy)
    # Using low margin products for clearance
    low_margin_prods = prod_economics[prod_economics['gross_margin_pct'] < 30.0]
    if achieved_profit < target_profit and not low_margin_prods.empty:
        proj_impact = sum(low_margin_prods['estimated_cogs']) * 0.05 # Reclaim 5% tied capital
        candidates.append(StrategyCandidate(
            description="Reduce slow-moving inventory (Capital recovery)",
            projected_profit_impact=proj_impact
        ))
        achieved_profit += proj_impact
        
    # If still not enough, generic price optimization
    if achieved_profit < target_profit:
        remaining = target_profit - achieved_profit
        candidates.append(StrategyCandidate(
            description="Broad 2% price increase across top 20% products",
            projected_profit_impact=remaining
        ))
        achieved_profit += remaining

    return OptimizationResult(
        target_profit=target_profit,
        achieved_profit=achieved_profit,
        candidates=candidates
    )
