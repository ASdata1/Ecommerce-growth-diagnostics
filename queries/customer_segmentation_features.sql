-- Order-level rows for customer segmentation. One row per order_id;
-- rolled up to one row per customer_unique_id in the notebook (pandas),
-- since the distance feature needs per-order haversine before averaging.
WITH item_agg AS (
    SELECT
        order_id,
        SUM(freight_value) AS freight_value,
        MIN(seller_id) AS seller_id
    FROM order_items
    GROUP BY order_id
),
payment_agg AS (
    SELECT
        order_id,
        SUM(payment_value) AS payment_value,
        MAX(payment_installments) AS payment_installments
    FROM order_payments
    GROUP BY order_id
)
SELECT
    c.customer_unique_id,
    o.order_id,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    ia.freight_value,
    pa.payment_value,
    pa.payment_installments,
    r.review_score,
    cg.geolocation_lat AS customer_lat,
    cg.geolocation_lng AS customer_lng,
    sg.geolocation_lat AS seller_lat,
    sg.geolocation_lng AS seller_lng
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN item_agg ia ON ia.order_id = o.order_id
LEFT JOIN payment_agg pa ON pa.order_id = o.order_id
LEFT JOIN order_reviews r ON r.order_id = o.order_id
LEFT JOIN geolocation cg ON cg.geolocation_zip_code_prefix = c.customer_zip_code_prefix
LEFT JOIN sellers s ON s.seller_id = ia.seller_id
LEFT JOIN geolocation sg ON sg.geolocation_zip_code_prefix = s.seller_zip_code_prefix;
