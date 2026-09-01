-- SCORING CANDIDATES (PostgreSQL) - outreach list the repeat-purchase model scores.
-- Postgres variant of repeat_purchase_scoring_candidates.sql; see that file for
-- the rationale. Timestamp columns arrive as TEXT from the migrated cloud tables,
-- hence the ::timestamp casts (same as repeat_purchase_features.postgres.sql).

WITH dataset_bounds AS (
    SELECT
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
        ) AS order_rank
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
),
first_orders AS (
    SELECT orders_ranked.*
    FROM orders_ranked, dataset_bounds
    WHERE order_rank = 1
      AND order_delivered_customer_date IS NOT NULL
      AND order_purchase_timestamp >= dataset_bounds.cutoff_date  -- inside the buffer: complement of training query's strict "<"
),
first_order_items AS (
    SELECT
        oi.order_id,
        COUNT(*) AS num_items,
        SUM(oi.price) AS items_price,
        SUM(oi.freight_value) AS freight_value,
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
    SELECT
        f.order_id,
        CASE
            WHEN r.review_score IS NOT NULL
             AND EXTRACT(EPOCH FROM (r.review_creation_date::timestamp - f.order_delivered_customer_date)) / 86400.0 <= 30
            THEN r.review_score::double precision
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
    (EXTRACT(EPOCH FROM (f.order_delivered_customer_date - f.order_purchase_timestamp)) / 86400.0)::double precision
        AS delivery_time_days,
    (EXTRACT(EPOCH FROM (f.order_delivered_customer_date - f.order_estimated_delivery_date)) / 86400.0)::double precision
        AS delivery_delay_days
FROM first_orders f
LEFT JOIN first_order_items fi ON fi.order_id = f.order_id
LEFT JOIN first_order_payments fp ON fp.order_id = f.order_id
LEFT JOIN first_order_review fr ON fr.order_id = f.order_id;
