"""
Push the local SQLite tables (data/olist.db) to a cloud Postgres database.

"""

import sqlite3

import pandas as pd
from sqlalchemy import text

from db import DEFAULT_SQLITE_PATH, get_engine

TABLES = ["customers", "orders", "order_items", "order_payments", "order_reviews", "products"]

# Same join-key indexes src/etl.py builds on the local SQLite DB. to_sql(...) only
# copies rows, not indexes, so without this every downstream join/subquery on the
# cloud DB is a full table scan.
INDEXES = {
    "idx_orders_customer_id": "orders(customer_id)",
    "idx_orders_order_id": "orders(order_id)",
    "idx_order_items_order_id": "order_items(order_id)",
    "idx_order_payments_order_id": "order_payments(order_id)",
    "idx_customers_customer_id": "customers(customer_id)",
    "idx_customers_customer_unique_id": "customers(customer_unique_id)",
    "idx_order_reviews_order_id": "order_reviews(order_id)",
    "idx_products_product_id": "products(product_id)",
}


def main() -> None:
    if not DEFAULT_SQLITE_PATH.exists():
        raise SystemExit(f"{DEFAULT_SQLITE_PATH} not found - run src/etl.py first.")

    local = sqlite3.connect(DEFAULT_SQLITE_PATH)
    cloud_engine = get_engine()

    if cloud_engine.url.get_backend_name() == "sqlite":
        raise SystemExit(
            "DATABASE_URL is not set (or .env wasn't found) - migrate_to_cloud.py "
            "needs a cloud Postgres URL to migrate TO. Set DATABASE_URL in .env first "
            
        )

    for table in TABLES:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", local)
        except pd.errors.DatabaseError:
            print(f"skipping {table}: not found in local DB (run the updated etl.py first)")
            continue
        df.to_sql(table, cloud_engine, if_exists="replace", index=False, chunksize=1000, method = "multi") # inserts 1000 rows per group
        print(f"migrated {table}: {len(df):,} rows -> cloud")

    with cloud_engine.begin() as conn:
        for name, target in INDEXES.items():
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {target}"))
    print(f"created {len(INDEXES)} join-key indexes")



if __name__ == "__main__":
    main()
