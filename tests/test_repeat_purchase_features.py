"""
Sanity checks on the repeat-purchase feature query - same spirit as
test_etl.py (assert against the real database, not mocks), scoped to the
new feature-engineering query rather than re-testing the base ETL.

Requires data/olist.db to exist (run src/etl.py first) - these tests are
skipped, not failed, if it's missing, since a fresh clone won't have the
data downloaded yet.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "olist.db"
QUERY_PATH = Path(__file__).resolve().parent.parent / "queries" / "repeat_purchase_features.sql"
CANDIDATES_QUERY_PATH = (
    Path(__file__).resolve().parent.parent / "queries" / "repeat_purchase_scoring_candidates.sql"
)

# repeat_purchase_analysis.py is meant to be run as `python src/repeat_purchase_analysis.py`
# and its own imports do `from db import ...`, so put src/ on sys.path the same way before
# importing the pieces under test. (Needs the project requirements installed - a bare env
# without sklearn/statsmodels will error at collection here, not skip.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repeat_purchase_analysis import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    _rank_and_decile,
    load_features,
    score_scoring_candidates,
)

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="data/olist.db not built yet - run src/etl.py")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(QUERY_PATH.read_text(), conn, parse_dates=["first_order_date"])
    conn.close()
    return df


def test_one_row_per_customer(features: pd.DataFrame) -> None:
    assert features["customer_unique_id"].is_unique


def test_target_is_binary(features: pd.DataFrame) -> None:
    assert set(features["repeat_purchase"].unique()) <= {0, 1}


def test_target_is_rare_but_present(features: pd.DataFrame) -> None:
    # this is the class-imbalance check that motivates class_weight="balanced"
    # and PR-AUC over accuracy in repeat_purchase_analysis.py - if this ever
    # fails because the positive class vanished entirely, the model script
    # will silently be meaningless, so catch it here instead.
    repeat_rate = features["repeat_purchase"].mean()
    assert 0 < repeat_rate < 0.5, f"expected a rare positive class, got repeat_rate={repeat_rate:.4f}"


def test_no_negative_delivery_time(features: pd.DataFrame) -> None:
    # delivered before purchased would indicate a data/join bug upstream
    assert (features["delivery_time_days"].dropna() >= 0).all()


def test_right_censoring_cutoff_applied(features: pd.DataFrame) -> None:
    # no customer's first order should be within ~3 months of the dataset's
    # latest order - this is the query's protection against counting someone
    # as "not a repeat customer" just because they haven't had time yet
    conn = sqlite3.connect(DB_PATH)
    dataset_max_date = pd.read_sql("SELECT MAX(order_purchase_timestamp) AS m FROM orders", conn)["m"].iloc[0]
    conn.close()
    dataset_max_date = pd.to_datetime(dataset_max_date)

    latest_included_first_order = features["first_order_date"].max()
    gap_days = (dataset_max_date - latest_included_first_order).days
    # ~90 days expected; allow a few days of slack for month-length variation
    # in SQLite's '-3 months' date arithmetic
    assert gap_days >= 85, f"expected at least ~3 months of buffer, got {gap_days} days"


def test_review_delay_days_bounded_where_score_known(features: pd.DataFrame) -> None:
    # every review that survived the "known" gate was created within 30 days of
    # delivery (the gate's no-lower-bound window), so review_delay_days must be
    # present and <= 30 wherever review_score is
    known = features["review_score"].notnull()
    assert features.loc[known, "review_delay_days"].notnull().all()
    assert (features.loc[known, "review_delay_days"] <= 30).all()



@pytest.fixture(scope="module")
def candidates() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(CANDIDATES_QUERY_PATH.read_text(), conn)
    conn.close()
    return df


def test_candidates_one_row_per_customer(candidates: pd.DataFrame) -> None:
    assert candidates["customer_unique_id"].is_unique


def test_candidates_non_empty(candidates: pd.DataFrame) -> None:
    # the dataset spans ~2 years, so the last 3 months always hold some delivered
    # first orders - if this query returns nothing, score_scoring_candidates()
    # silently no-ops and there is no outreach list to act on.
    assert len(candidates) > 0


def test_candidates_have_no_target_column(candidates: pd.DataFrame) -> None:
    # the whole reason this query exists separately: repeat_purchase is not
    # knowable yet for these too-recent customers, so it must not be emitted.
    assert "repeat_purchase" not in candidates.columns


def test_candidates_columns_are_exactly_the_model_features(candidates: pd.DataFrame) -> None:
    # score_scoring_candidates() does candidates[NUMERIC_FEATURES + CATEGORICAL_FEATURES];
    # a renamed or missing column would KeyError at predict time.
    expected = {"customer_unique_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    assert set(candidates.columns) == expected


def test_candidates_disjoint_from_training_set(
    candidates: pd.DataFrame, features: pd.DataFrame
) -> None:
    # the two queries partition the delivered first-time customers by the
    # right-censoring cutoff - no customer should appear in both.
    overlap = set(features["customer_unique_id"]) & set(candidates["customer_unique_id"])
    assert not overlap, f"{len(overlap)} customers are in both the training and candidate queries"


def test_candidates_are_inside_the_censoring_window(candidates: pd.DataFrame) -> None:
    # exact complement of test_right_censoring_cutoff_applied: every candidate's
    # first order is *within* ~3 months of the dataset's latest order.
    conn = sqlite3.connect(DB_PATH)
    first_dates = pd.read_sql(
        """
        WITH ranked AS (
            SELECT c.customer_unique_id,
                   o.order_purchase_timestamp,
                   ROW_NUMBER() OVER (PARTITION BY c.customer_unique_id
                                      ORDER BY o.order_purchase_timestamp) AS rk
            FROM orders o JOIN customers c ON o.customer_id = c.customer_id
        )
        SELECT customer_unique_id, order_purchase_timestamp AS first_order_date
        FROM ranked WHERE rk = 1
        """,
        conn,
        parse_dates=["first_order_date"],
    )
    dataset_max = pd.to_datetime(
        pd.read_sql("SELECT MAX(order_purchase_timestamp) AS m FROM orders", conn)["m"].iloc[0]
    )
    conn.close()

    cand_first = first_dates[first_dates["customer_unique_id"].isin(candidates["customer_unique_id"])]
    gap_days = (dataset_max - cand_first["first_order_date"]).dt.days
    assert (gap_days >= 0).all()
    assert (gap_days <= 95).all(), f"a candidate's first order is {gap_days.max()} days back - outside the window"


def test_candidates_no_negative_delivery_time(candidates: pd.DataFrame) -> None:
    # the candidate query re-implements the delivery-feature CTEs, so re-check
    # the same join-bug guard the training query has.
    assert (candidates["delivery_time_days"].dropna() >= 0).all()


# --------------------------------------------------------------------------------------
# score_scoring_candidates() itself - the _rank_and_decile helper (pure) and one
# end-to-end run against a throwaway copy of the local DB.
# --------------------------------------------------------------------------------------


def test_rank_and_decile_basic() -> None:
    n = 100
    df = pd.DataFrame({"repeat_probability": np.linspace(1.0, 0.0, n)})  # already sorted desc
    out = _rank_and_decile(df)

    assert out["rank"].tolist() == list(range(1, n + 1))
    assert out["decile"].tolist()[:10] == [1] * 10   # top 10% -> decile 1
    assert out["decile"].tolist()[-10:] == [10] * 10  # bottom 10% -> decile 10
    assert out["decile"].is_monotonic_increasing
    # does not mutate the caller's frame
    assert "rank" not in df.columns and "decile" not in df.columns


def test_rank_and_decile_small_n_stays_in_bounds() -> None:
    # guards the `arange(n) * 10 // n` arithmetic when n < 10
    out = _rank_and_decile(pd.DataFrame({"repeat_probability": [0.9, 0.5, 0.1]}))
    assert out["rank"].tolist() == [1, 2, 3]
    assert out["decile"].between(1, 10).all()


def test_scored_candidates_schema(scored_candidates_table: pd.DataFrame) -> None:
    assert set(scored_candidates_table.columns) == {
        "customer_unique_id",
        "repeat_probability",
        "rank",
        "decile",
    }


def test_scored_candidates_one_row_per_customer(scored_candidates_table: pd.DataFrame) -> None:
    assert scored_candidates_table["customer_unique_id"].is_unique


def test_scored_candidates_probability_in_unit_interval(scored_candidates_table: pd.DataFrame) -> None:
    assert scored_candidates_table["repeat_probability"].between(0.0, 1.0).all()


def test_scored_candidates_rank_is_dense_and_orders_by_probability(
    scored_candidates_table: pd.DataFrame,
) -> None:
    t = scored_candidates_table.sort_values("rank")
    assert t["rank"].tolist() == list(range(1, len(t) + 1))
    assert t["repeat_probability"].is_monotonic_decreasing  # rank 1 = highest probability


def test_scored_candidates_deciles_track_probability(scored_candidates_table: pd.DataFrame) -> None:
    d = scored_candidates_table["decile"]
    assert d.between(1, 10).all()
    top = scored_candidates_table.loc[d == 1, "repeat_probability"]
    rest = scored_candidates_table.loc[d > 1, "repeat_probability"]
    assert top.min() >= rest.max()  # the top decile really is the highest-scored slice


def test_scored_candidates_cover_exactly_the_query_population(
    scored_candidates_table: pd.DataFrame, candidates: pd.DataFrame
) -> None:
    assert set(scored_candidates_table["customer_unique_id"]) == set(candidates["customer_unique_id"])
