-- Q1: FUNNEL
-- Of all orders, what % reach each status: created -> approved -> invoiced -> shipped -> delivered?
--
-- Table: orders
--   order_id, customer_id, order_status,
--   order_purchase_timestamp, order_approved_at,
--   order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
--
-- Hint: order_status already tells you the LATEST stage an order reached, but that alone won't
-- give you a clean funnel (statuses aren't strictly ordered in the raw data). A simpler and more
-- honest approach: use the timestamp columns directly - an order "reached" a stage if that
-- stage's timestamp/date column is not null. Count non-null timestamps per stage, divide by
-- total orders.
--
-- TODO: write the funnel query here.
-- Expected output columns: stage, orders_reached, pct_of_total

WITH totals AS (
    SELECT COUNT(*) AS total_orders FROM orders
),
stages AS (
    SELECT 'purchased'   AS stage, 1 AS stage_order, COUNT(*) AS orders_reached FROM orders WHERE order_purchase_timestamp IS NOT NULL
    UNION ALL
    SELECT 'approved',  2, COUNT(*) FROM orders WHERE order_approved_at IS NOT NULL
    UNION ALL
    SELECT 'shipped',   3, COUNT(*) FROM orders WHERE order_delivered_carrier_date IS NOT NULL
    UNION ALL
    SELECT 'delivered', 4, COUNT(*) FROM orders WHERE order_delivered_customer_date IS NOT NULL
)
SELECT
    stage,
    orders_reached,
    ROUND(100.0 * orders_reached / total_orders, 2) AS pct_of_total
FROM stages, totals
ORDER BY stage_order;