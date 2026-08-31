"""
ETL: raw Olist CSVs -> cleaned SQLite database (data/olist.db)

"""

import sqlite3
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "olist.db"


def load_orders() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    # order_id and customer_id are the join keys used everywhere downstream - never null
    df = df.dropna(subset=["order_id", "customer_id"])
    date_cols = [c for c in df.columns if "timestamp" in c or "date" in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.drop_duplicates(subset=["order_id"])


def load_order_items() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    df = df.dropna(subset=["order_id"])
    return df


def load_order_payments() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv")
    df = df.dropna(subset=["order_id"])
    return df


def load_customers() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    df = df.dropna(subset=["customer_id"])
    return df.drop_duplicates(subset=["customer_id"])


def load_order_reviews() -> pd.DataFrame:
    """Added for the repeat-purchase model: review_score is a candidate driver
    of whether a first-time customer comes back."""
    df = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    df = df.dropna(subset=["order_id"])
    # a small number of orders have more than one review row (resubmitted
    # review) - keep the most recent one per order
    date_cols = [c for c in df.columns if "date" in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.sort_values("review_creation_date").drop_duplicates(subset=["order_id"], keep="last")
    return df


def load_products() -> pd.DataFrame:
    """Added for the repeat-purchase model: product_category_name is a candidate
    feature (some categories may retain customers better than others)."""
    df = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
    df = df.dropna(subset=["product_id"])
    return df.drop_duplicates(subset=["product_id"])


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "orders": load_orders(),
        "order_items": load_order_items(),
        "order_payments": load_order_payments(),
        "customers": load_customers(),
        "order_reviews": load_order_reviews(),
        "products": load_products(),
    }

    with sqlite3.connect(DB_PATH) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
            print(f"loaded {name}: {len(df):,} rows")

        # join columns get hit by every downstream query (and the referential
        # integrity tests) - without these, SQLite falls back to full table
        # scans per row on the joins/subqueries below
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_payments_order_id ON order_payments(order_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_customer_unique_id ON customers(customer_unique_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_reviews_order_id ON order_reviews(order_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_product_id ON products(product_id)")

    print(f"\ndone -> {DB_PATH}")


if __name__ == "__main__":
    main()
