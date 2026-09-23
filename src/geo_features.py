"""
Pure-Python geo feature engineering shared by notebooks/geolocation_analysis.ipynb
and src/repeat_purchase_analysis.py - one implementation, not two copies that
drift apart. Exists because SQLite (this project's local DB) has no trig
functions, so haversine distance can't be computed in queries/*.sql (see
repeat_purchase_features.sql's first_order_geo CTE comment). SQL only joins
and exposes raw lat/lng + seller_state + seller_state_seller_count; this
module derives model-ready features from them in numpy.
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds customer_seller_distance_km and same_state to a features DataFrame
    that already has customer_lat/lng, seller_lat/lng, customer_state,
    seller_state (from either repeat_purchase_features.sql or
    repeat_purchase_scoring_candidates.sql, either dialect). Missing
    coordinates/state produce NaN, not a dropped row - same treatment any other
    missing numeric/categorical feature gets from the model pipeline's imputer.
    """
    df = df.copy()
    df["customer_seller_distance_km"] = haversine_km(
        df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"]
    )
    same_state = (df["customer_state"] == df["seller_state"]).astype(float)
    same_state[df["customer_state"].isna() | df["seller_state"].isna()] = np.nan
    df["same_state"] = same_state
    return df
