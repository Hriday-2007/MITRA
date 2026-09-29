"""
MITRA Configuration Settings
All configurable parameters for the pipeline.
"""
import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Raw data (do NOT modify)
RAW_DATA_PATH = DATA_DIR / "online_retail_II.csv"

# Cleaned outputs
CLEAN_CSV_PATH = DATA_DIR / "clean_transactions.csv"
CLEAN_PARQUET_PATH = DATA_DIR / "clean_transactions.parquet"

# Derived datasets
PRODUCT_PERFORMANCE_PATH = DATA_DIR / "product_performance.csv"
PRODUCT_RELATIONSHIPS_PATH = DATA_DIR / "product_relationships.csv"
CUSTOMER_SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"
SEASONALITY_FEATURES_PATH = DATA_DIR / "seasonality_features.csv"
PRODUCT_ECONOMICS_PATH = DATA_DIR / "product_economics.csv"
BUSINESS_PROFILE_PATH = DATA_DIR / "merchant_business_profile.json"

# ─── Data Cleaning Settings ──────────────────────────────────────────────────
CLEANING = {
    "min_unit_price": 0.01,           # Drop rows where UnitPrice < this
    "min_quantity": 1,                 # Drop rows where Quantity < this (for sales)
    "cancelled_invoice_prefix": "C",   # Invoices starting with C are cancellations
    "drop_missing_description": True,  # Drop rows with no product description
    # CustomerID missing: keep for aggregate analytics, exclude from customer models
}

# ─── Synthetic COGS Model ─────────────────────────────────────────────────────
# IMPORTANT: These are ESTIMATED cost ratios for demonstration only.
# They are NOT real merchant COGS. Clearly disclosed as synthetic.
COST_RATIO_DEFAULT = 0.55   # Default: 55% of revenue = COGS → 45% gross margin
COST_RATIO_BY_KEYWORD = {
    # keyword (lowercase) → cost_ratio
    "glass":     0.60,
    "metal":     0.58,
    "ceramic":   0.62,
    "paper":     0.40,
    "card":      0.35,
    "bag":       0.45,
    "tin":       0.52,
    "candle":    0.50,
    "light":     0.55,
    "frame":     0.58,
    "holder":    0.55,
    "set":       0.53,
    "jumbo":     0.60,
    "vintage":   0.48,
    "christmas": 0.50,
    "cake":      0.40,
    "clock":     0.60,
}

# ─── Association Rule Mining ─────────────────────────────────────────────────
ASSOCIATION_RULES = {
    "min_support": 0.01,         # At least 1% of baskets
    "min_confidence": 0.20,      # At least 20% confidence
    "min_lift": 1.5,             # Lift must exceed 1.5×
    "min_transactions": 20,      # Basket must appear in >= 20 invoices
    "max_antecedents": 2,        # A+B → C (max 2 items on left side)
}

# ─── RFM Settings ────────────────────────────────────────────────────────────
RFM = {
    "recency_quantiles": 4,      # Number of R bins
    "frequency_quantiles": 4,    # Number of F bins
    "monetary_quantiles": 4,     # Number of M bins
    "snapshot_date_offset_days": 1,   # Days after last transaction for snapshot date
}

# ─── Forecasting ─────────────────────────────────────────────────────────────
FORECASTING = {
    "horizon_weeks": 8,          # Forecast 8 weeks ahead
    "train_ratio": 0.80,         # 80% train / 20% validation
    "seasonality_period": 52,    # Weekly seasonality (52 weeks)
}

# ─── Inventory Model (Demo) ──────────────────────────────────────────────────
INVENTORY = {
    "initial_stock_days": 30,    # Assume 30 days of stock on hand at snapshot
    "safety_stock_days": 7,      # Safety stock = 7 days of demand
    "lead_time_days": 14,        # Supplier lead time = 14 days
    "reorder_point_days": 21,    # Reorder when stock falls to 21 days
}

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# ─── Database Settings ───────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://mitra:mitra@localhost:5432/mitra_db"
)

# ─── LLM Settings ────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "mitra.log"
