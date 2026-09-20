"""
Sanity checks on the ETL output. Run with: pytest tests/

"""

import sqlite3
from pathlib import Path
import src.etl


import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "olist.db"


@pytest.fixture
def conn():
    if not DB_PATH.exists():
        pytest.skip("data/olist.db not found - run `python src/etl.py` first")
    connection = sqlite3.connect(DB_PATH)
    yield connection
    connection.close()


def test_orders_table_not_empty(conn):
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert count > 0
def test_order_items_table_not_empty(conn):
    count = conn.execute("SELECT COUNT(*) FROM order_items").fetchone()[0]
    assert count > 0
def test_customers_table_not_empty(conn):
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    assert count > 0
def test_payments_table_not_empty(conn):
    count = conn.execute("SELECT COUNT(*) FROM order_payments").fetchone()[0]
    assert count > 0
def test_geolocation_table_not_empty(conn):
    count = conn.execute("SELECT COUNT(*) FROM geolocation").fetchone()[0]
    assert count > 0
def test_sellers_table_not_empty(conn):
    count = conn.execute("SELECT COUNT(*) FROM sellers").fetchone()[0]
    assert count > 0

def test_load_orders_no_null_order_id():
    df = src.etl.load_orders()
    assert df["order_id"].isnull().sum() == 0

def test_load_customers_no_null_customer_id():
    df = src.etl.load_customers()
    assert df["customer_id"].isnull().sum() == 0

def test_load_geolocation_one_row_per_zip_prefix():
    # raw file has many lat/lng readings per prefix - load_geolocation() collapses
    # them so the geo join in queries/repeat_purchase_features.sql is 1:1
    df = src.etl.load_geolocation()
    assert df["geolocation_zip_code_prefix"].is_unique

def test_load_sellers_no_null_seller_id():
    df = src.etl.load_sellers()
    assert df["seller_id"].isnull().sum() == 0

# every order_id in order_items must also exist in orders 
def test_order_id_integrity():
    orders_df = src.etl.load_orders()
    order_items_df = src.etl.load_order_items()
    assert order_items_df['order_id'].isin(orders_df['order_id']).all() == True

# every customer_id in orders must also exist in customers 
def test_order_customer_id_integrity(conn):
    cust_ord_id = conn.execute("""
        SELECT COUNT(*) FROM orders o
        WHERE NOT EXISTS (
            SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id   
        )
    """).fetchone()[0]
    assert cust_ord_id == 0

