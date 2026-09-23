-- FEATURES: REPEAT-PURCHASE PROPENSITY  (SQLite dialect - used by the tests
-- and any local olist.db run.)
--
-- One row per customer, built ONLY from their FIRST order. This matters:
-- if we pulled features from ALL of a customer's orders (e.g. total spend
-- across every order), repeat customers would trivially look different
-- from one-time customers just because they have more rows to sum -
-- that's leakage, not signal. Everything here is knowable at the moment
-- the first order completes, which is the only point a real business
-- could actually act on this prediction.
--
-- Target: repeat_purchase = 1 if the customer placed more than one order
-- (any time in the dataset), 0 if their first order was their only order.
--
-- Right-censoring cutoff: a customer whose first order happened near the
-- end of the dataset's date range simply hasn't had time to place a
-- second order yet - that's not the same as "won't repeat", and left in,
-- it would quietly bias repeat_purchase downward for recent cohorts and
-- confound any feature (like review_score's availability, see below) that
-- happens to correlate with order recency. dataset_bounds computes a cutoff 3 months
-- before the latest order in the data - a buffer at least as long as the
-- longest retention window this project already tracks (cohort_retention.sql
-- goes out to month 2) - and first_orders excludes anyone whose first
-- order is more recent than that cutoff.
--
-- review_score: a review is only usable here if it existed in time to act on.
-- Olist's review request goes out days after delivery and not everyone responds,
-- so many first orders have no review at all. review_score counts as "known" only
-- when review_creation_date is within 30 days of delivery; reviews that arrive
-- later, or never arrive, leave review_score null. That 30-day window depends only
-- on delivery timing and whether the customer responded - NOT on whether or when
-- they placed a second order - so review_score's null-ness is not a function of
-- the target. (An earlier version also required the review to predate the
-- customer's second order. That arm fired only when a second order existed, which
-- made ~38% of repeat purchasers' reviews disappear vs <1% of one-timers' -
-- null-ness became partly the label itself - so it was dropped.) review_delay_days
-- (delivery -> review, kept raw even for reviews past the 30-day window) is exposed
-- for the EDA; it is not fed to the model. There is no review_missing flag - see
-- repeat_purchase_analysis.py for that call.
--
-- Uses window functions (ROW_NUMBER, COUNT ... OVER PARTITION BY) rather
-- than a correlated subquery to find each customer's first order - faster,
-- and the natural next SQL skill up from CTEs/subqueries.
--
-- Geo features (first_order_geo below): SQLite has no trig functions, so
-- distance can't be computed here. This query only joins and exposes raw
-- customer_lat/lng, seller_lat/lng, seller_state and seller_state_seller_count;
-- src/geo_features.py's add_geo_features() turns those into
-- customer_seller_distance_km and same_state in pandas instead.
--
-- Tables: orders, order_items, order_payments, order_reviews, products,
-- customers, geolocation, sellers
-- Expected output: one row per customer_unique_id with features + repeat_purchase label

WITH dataset_bounds AS (
    SELECT
        MAX(order_purchase_timestamp) AS max_order_date,
        date(MAX(order_purchase_timestamp), '-3 months') AS cutoff_date
    FROM orders
),
orders_ranked AS (
    SELECT
        c.customer_unique_id,
        c.customer_state,
        c.customer_zip_code_prefix,
        o.order_id,
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date,
        o.order_delivered_customer_date,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp
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
      AND order_purchase_timestamp <= dataset_bounds.cutoff_date  -- right-censoring: exclude
        -- customers who haven't had the 3-month buffer yet to place a second order
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
    -- first-order time: created within 30 days of delivery. One written weeks later
    -- simply wasn't available yet; that (and no review at all) leaves review_score
    -- null. This gate depends only on delivery date and review date, not on the
    -- repeat_purchase outcome. review_delay_days is kept raw (even for reviews past
    -- the 30-day window) so the EDA can size how much signal the gate drops.
    SELECT
        f.order_id,
        CASE
            WHEN r.review_score IS NOT NULL
             AND julianday(r.review_creation_date) - julianday(f.order_delivered_customer_date) <= 30
            THEN r.review_score
        END AS review_score,
        CAST(julianday(r.review_creation_date) - julianday(f.order_delivered_customer_date) AS REAL)
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
    CAST(julianday(f.order_delivered_customer_date) - julianday(f.order_purchase_timestamp) AS REAL)
        AS delivery_time_days,
    CAST(julianday(f.order_delivered_customer_date) - julianday(f.order_estimated_delivery_date) AS REAL)
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

