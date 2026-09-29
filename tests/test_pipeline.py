"""
MITRA — Unit Tests
Tests all analytics functions using small known datasets.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small synthetic dataset for unit tests."""
    rows = [
        # Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
        ("INV001", "P001", "RED MUG", 2, "2023-01-10 10:00", 5.0, "C001", "United Kingdom"),
        ("INV001", "P002", "BLUE PEN", 5, "2023-01-10 10:00", 1.5, "C001", "United Kingdom"),
        ("INV002", "P001", "RED MUG", 1, "2023-01-15 11:00", 5.0, "C002", "United Kingdom"),
        ("INV002", "P003", "GREEN BAG", 3, "2023-01-15 11:00", 3.0, "C002", "United Kingdom"),
        ("INV003", "P001", "RED MUG", 4, "2023-02-01 09:00", 5.0, "C003", "Germany"),
        ("INV003", "P002", "BLUE PEN", 10, "2023-02-01 09:00", 1.5, "C003", "Germany"),
        ("INV003", "P003", "GREEN BAG", 2, "2023-02-01 09:00", 3.0, "C003", "Germany"),
        ("INV004", "P001", "RED MUG", 1, "2023-03-05 14:00", 5.0, "UNKNOWN", "United Kingdom"),
    ]
    df = pd.DataFrame(rows, columns=[
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country"
    ])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Week"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df


@pytest.fixture
def raw_sample_df():
    """Sample raw data with issues for cleaning tests."""
    rows = [
        # Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
        ("INV001", "P001", "MUG", 2, "2023-01-10", 5.0, "C001", "UK"),      # valid
        ("CINV001", "P001", "MUG", -2, "2023-01-10", 5.0, "C001", "UK"),    # cancelled
        ("INV002", "P002", "PEN", -1, "2023-01-10", 2.0, "C001", "UK"),     # negative qty
        ("INV003", "P003", "BAG", 5, "2023-01-10", 0.0, "C002", "UK"),      # zero price
        ("INV004", None, "GLASS", 3, "2023-01-10", 4.0, "C002", "UK"),      # null description (StockCode)
        ("INV005", "P005", None, 3, "2023-01-10", 4.0, "C002", "UK"),       # null description
        ("INV006", "P006", "BOX", 2, "2023-01-10", 3.0, None, "UK"),        # no CustomerID
        ("INV001", "P001", "MUG", 2, "2023-01-10", 5.0, "C001", "UK"),      # duplicate
    ]
    df = pd.DataFrame(rows, columns=[
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country"
    ])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df


# ── Test: Data Cleaning ───────────────────────────────────────────────────────

class TestCleaner:
    def test_removes_duplicates(self, raw_sample_df):
        from src.data.cleaner import clean
        clean_df, log = clean(raw_sample_df, verbose=False)
        assert log["1_removed_duplicates"] >= 1

    def test_removes_cancelled(self, raw_sample_df):
        from src.data.cleaner import clean
        clean_df, log = clean(raw_sample_df, verbose=False)
        assert not clean_df["Invoice"].str.upper().str.startswith("C").any()
        assert log["2_removed_cancelled"] >= 1

    def test_removes_negative_quantity(self, raw_sample_df):
        from src.data.cleaner import clean
        clean_df, log = clean(raw_sample_df, verbose=False)
        assert (clean_df["Quantity"] > 0).all()
        assert log["3_removed_nonpositive_quantity"] >= 1

    def test_removes_zero_price(self, raw_sample_df):
        from src.data.cleaner import clean
        clean_df, log = clean(raw_sample_df, verbose=False)
        assert (clean_df["Price"] > 0).all()

    def test_missing_customer_id_filled(self, raw_sample_df):
        from src.data.cleaner import clean
        clean_df, log = clean(raw_sample_df, verbose=False)
        assert clean_df["Customer ID"].isnull().sum() == 0
        assert "UNKNOWN" in clean_df["Customer ID"].values

    def test_revenue_column_added(self, raw_sample_df):
        from src.data.cleaner import clean
        clean_df, log = clean(raw_sample_df, verbose=False)
        assert "Revenue" in clean_df.columns
        assert (clean_df["Revenue"] > 0).all()


# ── Test: Revenue Calculations ────────────────────────────────────────────────

class TestMetrics:
    def test_calculate_revenue(self, sample_df):
        from src.analytics.metrics import calculate_revenue
        # 2×5 + 5×1.5 + 1×5 + 3×3 + 4×5 + 10×1.5 + 2×3 + 1×5 = 10+7.5+5+9+20+15+6+5 = 77.5
        expected = (2*5.0 + 5*1.5 + 1*5.0 + 3*3.0 + 4*5.0 + 10*1.5 + 2*3.0 + 1*5.0)
        assert abs(calculate_revenue(sample_df) - expected) < 0.01

    def test_calculate_aov(self, sample_df):
        from src.analytics.metrics import calculate_aov
        total_rev = (2*5.0 + 5*1.5 + 1*5.0 + 3*3.0 + 4*5.0 + 10*1.5 + 2*3.0 + 1*5.0)
        n_orders = 4
        expected = round(total_rev / n_orders, 2)
        assert abs(calculate_aov(sample_df) - expected) < 0.01

    def test_calculate_order_count(self, sample_df):
        from src.analytics.metrics import calculate_order_count
        assert calculate_order_count(sample_df) == 4

    def test_calculate_units_sold(self, sample_df):
        from src.analytics.metrics import calculate_units_sold
        expected = 2 + 5 + 1 + 3 + 4 + 10 + 2 + 1
        assert calculate_units_sold(sample_df) == expected

    def test_calculate_basket_size(self, sample_df):
        from src.analytics.metrics import calculate_basket_size
        # INV001: 2 products, INV002: 2, INV003: 3, INV004: 1 → avg = 2.0
        result = calculate_basket_size(sample_df)
        assert result == 2.0

    def test_repeat_rate(self, sample_df):
        from src.analytics.metrics import calculate_repeat_rate
        # C001 bought in INV001 only (1 order), C002 in INV002 only (1 order),
        # C003 in INV003 only (1 order) → 0 repeats
        result = calculate_repeat_rate(sample_df)
        assert result["total_customers"] == 3
        assert result["repeat_rate_pct"] == 0.0


# ── Test: Product Performance ─────────────────────────────────────────────────

class TestProductPerformance:
    def test_product_table_shape(self, sample_df):
        from src.analytics.metrics import calculate_product_performance
        result = calculate_product_performance(sample_df)
        assert len(result) == 3  # P001, P002, P003
        assert "total_revenue" in result.columns
        assert "revenue_contribution_pct" in result.columns

    def test_top_product_is_correct(self, sample_df):
        from src.analytics.metrics import calculate_product_performance
        result = calculate_product_performance(sample_df)
        # P001 (RED MUG): (2+1+4+1)×5 = 40
        # P002 (BLUE PEN): (5+10)×1.5 = 22.5
        # P003 (GREEN BAG): (3+2)×3 = 15
        assert result.iloc[0]["StockCode"] == "P001"

    def test_revenue_pct_sums_to_100(self, sample_df):
        from src.analytics.metrics import calculate_product_performance
        result = calculate_product_performance(sample_df)
        total_pct = result["revenue_contribution_pct"].sum()
        assert abs(total_pct - 100.0) < 0.1


# ── Test: RFM Segmentation ────────────────────────────────────────────────────

class TestRFM:
    def test_rfm_excludes_unknown(self, sample_df):
        from src.analytics.rfm import calculate_rfm
        rfm = calculate_rfm(sample_df)
        assert "UNKNOWN" not in rfm["Customer ID"].values

    def test_rfm_scores_in_range(self, sample_df):
        from src.analytics.rfm import calculate_rfm
        rfm = calculate_rfm(sample_df)
        for col in ["R_score", "F_score", "M_score"]:
            assert rfm[col].between(1, 4).all(), f"{col} out of range"

    def test_rfm_has_segment_column(self, sample_df):
        from src.analytics.rfm import calculate_rfm
        rfm = calculate_rfm(sample_df)
        assert "segment" in rfm.columns
        assert rfm["segment"].notnull().all()


# ── Test: Basket Analysis ─────────────────────────────────────────────────────

class TestBasket:
    def test_basket_matrix_shape(self, sample_df):
        from src.analytics.basket import build_basket_matrix
        basket = build_basket_matrix(sample_df)
        # INV001 (2 products), INV002 (2 products), INV003 (3 products)
        # INV004 has only 1 product → excluded
        assert basket.shape[0] == 3
        assert basket.shape[1] == 3   # P001, P002, P003

    def test_basket_is_binary(self, sample_df):
        from src.analytics.basket import build_basket_matrix
        basket = build_basket_matrix(sample_df)
        assert basket.dtypes.unique()[0] == bool


# ── Test: Synthetic Economics ─────────────────────────────────────────────────

class TestEconomics:
    def test_economics_adds_columns(self, sample_df):
        from src.analytics.metrics import calculate_product_performance
        from src.analytics.economics import calculate_product_economics
        prod = calculate_product_performance(sample_df)
        econ = calculate_product_economics(prod)
        assert "estimated_cogs" in econ.columns
        assert "gross_profit" in econ.columns
        assert "gross_margin_pct" in econ.columns
        assert "is_synthetic_cogs" in econ.columns
        assert econ["is_synthetic_cogs"].all()

    def test_gross_profit_positive(self, sample_df):
        from src.analytics.metrics import calculate_product_performance
        from src.analytics.economics import calculate_product_economics
        prod = calculate_product_performance(sample_df)
        econ = calculate_product_economics(prod)
        assert (econ["gross_profit"] >= 0).all()

    def test_margin_between_0_and_100(self, sample_df):
        from src.analytics.metrics import calculate_product_performance
        from src.analytics.economics import calculate_product_economics
        prod = calculate_product_performance(sample_df)
        econ = calculate_product_economics(prod)
        assert econ["gross_margin_pct"].between(0, 100).all()


# ── Test: Inventory Model ─────────────────────────────────────────────────────

class TestInventory:
    def test_inventory_columns_present(self, sample_df):
        from src.ml.inventory import build_inventory_model
        inv = build_inventory_model(sample_df)
        for col in ["avg_daily_units", "modelled_stock_units", "stockout_risk",
                    "velocity_label", "is_demo_inventory"]:
            assert col in inv.columns

    def test_stockout_risk_values(self, sample_df):
        from src.ml.inventory import build_inventory_model
        inv = build_inventory_model(sample_df)
        assert inv["stockout_risk"].isin(["LOW", "MEDIUM", "HIGH"]).all()

    def test_all_demo_flagged(self, sample_df):
        from src.ml.inventory import build_inventory_model
        inv = build_inventory_model(sample_df)
        assert inv["is_demo_inventory"].all()
