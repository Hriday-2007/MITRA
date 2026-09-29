import json
from pathlib import Path
import pandas as pd
from typing import List, Dict

from config.logging_config import get_logger
from config.settings import DATA_DIR
from src.recommendations.models import Recommendation
from src.recommendations.bundle import generate_bundle_recommendations
from src.recommendations.cross_sell import generate_cross_sell_recommendations
from src.recommendations.inventory import generate_inventory_recommendations
from src.recommendations.customers import generate_customer_recommendations
from src.recommendations.seasonality import generate_seasonal_recommendations
from src.recommendations.profitability import generate_profit_recommendations

logger = get_logger("mitra.recommendations.engine")

def generate_all_recommendations(
    rules: List[dict],
    inv_df: pd.DataFrame,
    prod_economics: pd.DataFrame,
    rfm_segments: pd.DataFrame,
    seasonality: Dict,
    total_revenue: float
) -> List[Dict]:
    """Orchestrate all recommendation generators."""
    logger.info("Generating recommendations...")
    
    recs: List[Recommendation] = []
    
    if rules:
        recs.extend(generate_bundle_recommendations(rules))
        recs.extend(generate_cross_sell_recommendations(rules))
        
    if inv_df is not None and not inv_df.empty:
        recs.extend(generate_inventory_recommendations(inv_df, prod_economics))
        
    if rfm_segments is not None and not rfm_segments.empty:
        recs.extend(generate_customer_recommendations(rfm_segments))
        
    if seasonality:
        recs.extend(generate_seasonal_recommendations(seasonality, total_revenue))
        
    if prod_economics is not None and not prod_economics.empty:
        recs.extend(generate_profit_recommendations(prod_economics))
        
    # Sort by priority score
    recs.sort(key=lambda x: x.priority_score, reverse=True)
    
    logger.info("Generated %d recommendations.", len(recs))
    
    return [r.to_dict() for r in recs]

def save_recommendations(recs: List[Dict]) -> None:
    path = DATA_DIR / "recommendations.json"
    with open(path, "w") as f:
        json.dump(recs, f, indent=2)
    logger.info("Saved recommendations -> %s", path)
