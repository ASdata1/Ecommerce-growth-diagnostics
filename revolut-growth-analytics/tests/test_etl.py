"""
Sanity checks on the ETL output. Run with: pytest tests/

These are intentionally minimal - the point is to have SOME automated check that the pipeline
didn't silently break, not to test every edge case.
"""

import sqlite3
from pathlib import Path

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


# TODO: add a test that order_id has no nulls in the orders table
# TODO: add a test that customer_id has no nulls in the customers table
# TODO: add a test that every order_id in order_items also exists in orders (referential
#       integrity - use a LEFT JOIN / NOT IN check)
