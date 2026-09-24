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


def load_geolocation() -> pd.DataFrame:
    """Added for geo features (src/geo_features.py): customer/seller distance and
    same-state match. The raw file has many lat/lng readings per zip prefix
    (repeat geocodes of addresses sharing that prefix), so it's collapsed here to
    one row per geolocation_zip_code_prefix (mean lat/lng) - that way
    queries/repeat_purchase_features.sql can join it with a plain equality join
    instead of fanning out first-order rows."""
    df = pd.read_csv(RAW_DIR / "olist_geolocation_dataset.csv")
    df = df.dropna(subset=["geolocation_zip_code_prefix"])
    return df.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        geolocation_lat=("geolocation_lat", "mean"),
        geolocation_lng=("geolocation_lng", "mean"),
    )


def load_sellers() -> pd.DataFrame:
    """Added for geo features: seller_state and (via geolocation) seller lat/lng
    are candidate drivers of customer-seller distance."""
    df = pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv")
    df = df.dropna(subset=["seller_id"])
    return df.drop_duplicates(subset=["seller_id"])


# join columns get hit by every downstream query (and the referential
# integrity tests) - without these, SQLite falls back to full table scans
# per row on the joins/subqueries below
INDEXES = {
    "idx_orders_customer_id": "orders(customer_id)",
    "idx_orders_order_id": "orders(order_id)",
    "idx_order_items_order_id": "order_items(order_id)",
    "idx_order_payments_order_id": "order_payments(order_id)",
    "idx_customers_customer_id": "customers(customer_id)",
    "idx_customers_customer_unique_id": "customers(customer_unique_id)",
    "idx_order_reviews_order_id": "order_reviews(order_id)",
    "idx_products_product_id": "products(product_id)",
    "idx_geolocation_zip": "geolocation(geolocation_zip_code_prefix)",
    "idx_sellers_seller_id": "sellers(seller_id)",
    "idx_sellers_zip": "sellers(seller_zip_code_prefix)",
    "idx_customers_zip": "customers(customer_zip_code_prefix)",
}


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "orders": load_orders(),
        "order_items": load_order_items(),
        "order_payments": load_order_payments(),
        "customers": load_customers(),
        "order_reviews": load_order_reviews(),
        "products": load_products(),
        "geolocation": load_geolocation(),
        "sellers": load_sellers(),
    }

    with sqlite3.connect(DB_PATH) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
            print(f"loaded {name}: {len(df):,} rows")

        for name, target in INDEXES.items():
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {target}")

    print(f"\ndone -> {DB_PATH}")


if __name__ == "__main__":
    main()
