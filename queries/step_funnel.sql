-- Step-over-step funnel conversion: for each adjacent pair of stages, what % of orders
-- that reached the earlier stage also reached the next one (from_count -> to_count).
-- Stages are defined by non-null timestamp columns on orders: purchase, approved,
-- delivered_carrier (shipped), delivered_customer (delivered).

WITH stages AS (
    SELECT 'created'   AS stage, 1 AS stage_order, COUNT(*) AS orders_reached FROM orders WHERE order_purchase_timestamp IS NOT NULL
    UNION ALL
    SELECT 'approved',  2, COUNT(*) FROM orders WHERE order_approved_at IS NOT NULL
    UNION ALL
    SELECT 'shipped',   3, COUNT(*) FROM orders WHERE order_delivered_carrier_date IS NOT NULL
    UNION ALL
    SELECT 'delivered', 4, COUNT(*) FROM orders WHERE order_delivered_customer_date IS NOT NULL
)
SELECT
    s1.stage          AS from_stage,
    s2.stage          AS to_stage,
    s1.orders_reached AS from_count,
    s2.orders_reached AS to_count,
    ROUND(100.0 * s2.orders_reached / s1.orders_reached, 2) AS pct_moved_to_next
FROM stages s1
JOIN stages s2 ON s2.stage_order = s1.stage_order + 1
ORDER BY s1.stage_order;