-- FEATURES: REPEAT-PURCHASE PROPENSITY  (PostgreSQL variant)
--
-- Identical logic to repeat_purchase_features.sql - keep the two in sync.
-- This file exists only because the date handling differs by dialect:
--   * the migrated cloud tables store every timestamp as TEXT (pandas ->
--     SQLite -> Postgres round-trip never assigned a real type), so every
--     date column is cast ::timestamp before arithmetic;
--   * SQLite's  date(x, '-3 months')     -> Postgres  x - INTERVAL '3 months'
--   * SQLite's  julianday(a) - julianday(b)  (fractional days)
--       -> Postgres  EXTRACT(EPOCH FROM (a - b)) / 86400.0
--   * day-difference outputs are cast ::double precision so psycopg2 returns
--     float (not Decimal), matching what the SQLite path feeds the model.
--
-- See repeat_purchase_features.sql for the full rationale behind each CTE
-- (first-order-only features, the right-censoring cutoff, and the 30-day-after-
-- delivery review window - which no longer looks at the second order at all).

WITH dataset_bounds AS (
    SELECT
        MAX(order_purchase_timestamp::timestamp) AS max_order_date,
        -- 3-month buffer before the latest order; ::date to match SQLite's
        -- date() dropping the time part, so the same first orders are excluded
        (MAX(order_purchase_timestamp::timestamp) - INTERVAL '3 months')::date AS cutoff_date
    FROM orders
),
orders_ranked AS (
    SELECT
        c.customer_unique_id,
        c.customer_state,
        o.order_id,
        o.order_purchase_timestamp::timestamp      AS order_purchase_timestamp,
        o.order_estimated_delivery_date::timestamp AS order_estimated_delivery_date,
        o.order_delivered_customer_date::timestamp AS order_delivered_customer_date,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp::timestamp
        ) AS order_rank,
        COUNT(*) OVER (PARTITION BY c.customer_unique_id) AS num_orders
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
),
first_orders AS (
    SELECT orders_ranked.*
    FROM orders_ranked, dataset_bounds
    WHERE order_rank = 1
      AND order_delivered_customer_date IS NOT NULL       -- delivery features need a delivery date
      AND order_purchase_timestamp < dataset_bounds.cutoff_date  -- right-censoring: exclude
        -- customers who haven't had the 3-month buffer yet to place a second order.
        -- strict '<' (not '<=') against the midnight-valued date matches the SQLite
        -- file, where date() vs full-timestamp string comparison excludes the boundary
),
first_order_items AS (
    SELECT
        oi.order_id,
        COUNT(*) AS num_items,
        SUM(oi.price) AS items_price,
        SUM(oi.freight_value) AS freight_value,
        -- crude "main category" pick for a multi-item order: first product's
        -- category alphabetically. Good enough for a portfolio feature; a
        -- production version would pick by highest item value instead.
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
        -- payment type tied to the largest single payment line, ties broken arbitrarily
        (
            SELECT payment_type FROM order_payments op2
            WHERE op2.order_id = op.order_id
            ORDER BY payment_value DESC LIMIT 1
        ) AS payment_type
    FROM order_payments op
    GROUP BY order_id
),
first_order_review AS (
    -- A review only counts as "known" if the business could have acted on it at
    -- first-order time: created within 30 days of delivery. Later (or absent)
    -- leaves review_score null. This gate depends only on delivery date and review
    -- date, not on the repeat_purchase outcome. review_delay_days is kept raw (even
    -- for reviews past the 30-day window) so the EDA can size the signal it drops.
    SELECT
        f.order_id,
        CASE
            WHEN r.review_score IS NOT NULL
             AND EXTRACT(EPOCH FROM (r.review_creation_date::timestamp - f.order_delivered_customer_date)) / 86400.0 <= 30
            THEN r.review_score::double precision
        END AS review_score,
        (EXTRACT(EPOCH FROM (r.review_creation_date::timestamp - f.order_delivered_customer_date)) / 86400.0)::double precision
            AS review_delay_days
    FROM first_orders f
    LEFT JOIN order_reviews r ON r.order_id = f.order_id
)

SELECT
    f.customer_unique_id,
    f.customer_state,
    f.order_purchase_timestamp AS first_order_date,
    fi.num_items,
    fi.items_price,
    fi.freight_value,
    fi.product_category,
    fp.payment_value,
    fp.payment_installments,
    fp.payment_type,
    fr.review_score,
    fr.review_delay_days,  -- delivery -> review in days, raw (kept even when the
        -- review failed the "known" gate above); EDA-only, not a model feature
    (EXTRACT(EPOCH FROM (f.order_delivered_customer_date - f.order_purchase_timestamp)) / 86400.0)::double precision
        AS delivery_time_days,
    (EXTRACT(EPOCH FROM (f.order_delivered_customer_date - f.order_estimated_delivery_date)) / 86400.0)::double precision
        AS delivery_delay_days,  -- positive = delivered later than promised
    CASE WHEN f.num_orders > 1 THEN 1 ELSE 0 END AS repeat_purchase
FROM first_orders f
LEFT JOIN first_order_items fi ON fi.order_id = f.order_id
LEFT JOIN first_order_payments fp ON fp.order_id = f.order_id
LEFT JOIN first_order_review fr ON fr.order_id = f.order_id;
