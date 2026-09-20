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
-- delivery review window - which no longer looks at the second order at all -
-- plus first_order_geo, which is dialect-identical: no date/trig arithmetic).

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
        c.customer_zip_code_prefix,
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
        MIN(p.product_category_name) AS product_category,
        -- same crude pick, applied to seller_id, so first_order_geo has exactly
        -- one seller per order to compute distance/same-state against
        MIN(oi.seller_id) AS seller_id
    FROM order_items oi
    LEFT JOIN products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
),
seller_state_counts AS (
    -- market-density feature: how many sellers operate out of this seller's
    -- state - a crude proxy for how competitive/established that state's seller
    -- base is, independent of any one seller's distance to this customer
    SELECT seller_state, COUNT(*) AS seller_state_seller_count
    FROM sellers
    GROUP BY seller_state
),
first_order_geo AS (
    -- Raw geo inputs for src/geo_features.py's add_geo_features(): customer
    -- coordinates from their zip prefix, seller_state from the sellers table
    -- directly, and the seller's coordinates from their zip prefix. geolocation
    -- is pre-aggregated to one row per zip prefix in src/etl.py's
    -- load_geolocation(), so these are plain equality joins, not fan-outs. A zip
    -- prefix with no geolocation match (rare - see load_geolocation) leaves
    -- lat/lng null, same as any other missing feature.
    SELECT
        f.order_id,
        cg.geolocation_lat AS customer_lat,
        cg.geolocation_lng AS customer_lng,
        s.seller_state,
        sg.geolocation_lat AS seller_lat,
        sg.geolocation_lng AS seller_lng
    FROM first_orders f
    LEFT JOIN geolocation cg ON cg.geolocation_zip_code_prefix = f.customer_zip_code_prefix
    LEFT JOIN first_order_items fi ON fi.order_id = f.order_id
    LEFT JOIN sellers s ON s.seller_id = fi.seller_id
    LEFT JOIN geolocation sg ON sg.geolocation_zip_code_prefix = s.seller_zip_code_prefix
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
    fg.customer_lat,
    fg.customer_lng,
    fg.seller_state,
    fg.seller_lat,
    fg.seller_lng,
    ssc.seller_state_seller_count,
    CASE WHEN f.num_orders > 1 THEN 1 ELSE 0 END AS repeat_purchase
FROM first_orders f
LEFT JOIN first_order_items fi ON fi.order_id = f.order_id
LEFT JOIN first_order_payments fp ON fp.order_id = f.order_id
LEFT JOIN first_order_review fr ON fr.order_id = f.order_id
LEFT JOIN first_order_geo fg ON fg.order_id = f.order_id
LEFT JOIN seller_state_counts ssc ON ssc.seller_state = fg.seller_state;
