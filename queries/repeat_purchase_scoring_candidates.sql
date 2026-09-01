-- SCORING CANDIDATES (SQLite) - outreach list the repeat-purchase model scores.
-- Keep in sync with repeat_purchase_scoring_candidates.postgres.sql (used when
-- DATABASE_URL points at the cloud DB).
--
-- Complement of repeat_purchase_features.sql: first-time customers INSIDE the
-- 3-month right-censoring window, too recent to know whether they'll reorder -
-- so no repeat_purchase target column (calling "no second order yet" a 0 would
-- reintroduce the bias the training cutoff avoids). Same first-order/feature
-- logic as that file. Needs a delivered first order for the delivery/review
-- features; undelivered first orders are dropped, not guessed.
--
-- Output: one row per customer_unique_id, feature columns only (matches
-- NUMERIC_FEATURES/CATEGORICAL_FEATURES in repeat_purchase_analysis.py). Scored
-- by score_scoring_candidates() there.

WITH dataset_bounds AS (
    SELECT
        date(MAX(order_purchase_timestamp), '-3 months') AS cutoff_date
    FROM orders
),
orders_ranked AS (
    SELECT
        c.customer_unique_id,
        c.customer_state,
        o.order_id,
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date,
        o.order_delivered_customer_date,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp
        ) AS order_rank
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
),
first_orders AS (
    SELECT orders_ranked.*
    FROM orders_ranked, dataset_bounds
    WHERE order_rank = 1
      AND order_delivered_customer_date IS NOT NULL       -- delivery features need a delivery date
      AND order_purchase_timestamp > dataset_bounds.cutoff_date  -- inside the buffer: complement of training query's "<="
),
first_order_items AS (
    SELECT
        oi.order_id,
        COUNT(*) AS num_items,
        SUM(oi.price) AS items_price,
        SUM(oi.freight_value) AS freight_value,
        -- main category for multi-item orders: first category alphabetically (matches training query)
        MIN(p.product_category_name) AS product_category
    FROM order_items oi
    LEFT JOIN products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
),
first_order_payments AS (
    SELECT
        order_id,
        SUM(payment_value) AS payment_value,
        MAX(payment_installments) AS payment_installments,
        (
            SELECT payment_type FROM order_payments op2
            WHERE op2.order_id = op.order_id
            ORDER BY payment_value DESC LIMIT 1
        ) AS payment_type
    FROM order_payments op
    GROUP BY order_id
),
first_order_review AS (
    -- only reviews in within 30 days of delivery, i.e. knowable at first-order time (matches training query)
    SELECT
        f.order_id,
        CASE
            WHEN r.review_score IS NOT NULL
             AND julianday(r.review_creation_date) - julianday(f.order_delivered_customer_date) <= 30
            THEN r.review_score
        END AS review_score
    FROM first_orders f
    LEFT JOIN order_reviews r ON r.order_id = f.order_id
)

SELECT
    f.customer_unique_id,
    f.customer_state,
    fi.num_items,
    fi.items_price,
    fi.freight_value,
    fi.product_category,
    fp.payment_value,
    fp.payment_installments,
    fp.payment_type,
    fr.review_score,
    CAST(julianday(f.order_delivered_customer_date) - julianday(f.order_purchase_timestamp) AS REAL)
        AS delivery_time_days,
    CAST(julianday(f.order_delivered_customer_date) - julianday(f.order_estimated_delivery_date) AS REAL)
        AS delivery_delay_days
FROM first_orders f
LEFT JOIN first_order_items fi ON fi.order_id = f.order_id
LEFT JOIN first_order_payments fp ON fp.order_id = f.order_id
LEFT JOIN first_order_review fr ON fr.order_id = f.order_id;
