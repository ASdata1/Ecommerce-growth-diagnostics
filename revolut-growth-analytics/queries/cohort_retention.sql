-- Q2: COHORT RETENTION
-- Group customers by the month of their FIRST order (their "cohort").
-- For each cohort, what % of customers placed another order in month 1, 2, 3... after that?
--
-- Tables: orders (order_id, customer_id, order_purchase_timestamp)
--
-- This is the hardest query in the project. Rough shape:
--   1. Find each customer's first order month  -> a CTE, using MIN() OVER (PARTITION BY customer_id)
--      or GROUP BY customer_id
--   2. For every order, compute months_since_first_order = difference between that order's
--      month and the customer's cohort month
--   3. Count distinct customers per (cohort_month, months_since_first_order)
--   4. Divide by the cohort's total size to get a retention %
--
-- Note: this Olist dataset is known to have very few repeat customers - a low/flat retention
-- curve is a real, correct finding here, not a bug. Say so plainly in the write-up.
--
-- TODO: write the cohort retention query here.
-- Expected output columns: cohort_month, months_since_first_order, customers_active, cohort_size, retention_pct


WITH customer_first_dates AS (
    SELECT
        c.customer_unique_id,
        MIN(o.order_purchase_timestamp) AS first_order_date
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
),
orders_with_cohort AS (
    SELECT
        c.customer_unique_id,
        f.first_order_date,
        o.order_purchase_timestamp,
        (CAST(strftime('%Y', o.order_purchase_timestamp) AS INTEGER) - CAST(strftime('%Y', f.first_order_date) AS INTEGER)) * 12
        + (CAST(strftime('%m', o.order_purchase_timestamp) AS INTEGER) - CAST(strftime('%m', f.first_order_date) AS INTEGER))
        AS months_since_first_order
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    JOIN customer_first_dates f
        ON c.customer_unique_id = f.customer_unique_id
),
cohort_sizes AS (
    SELECT
        strftime('%Y-%m', first_order_date) AS cohort_month,
        COUNT(DISTINCT customer_unique_id) AS cohort_size
    FROM customer_first_dates
    GROUP BY cohort_month
),
activity AS (
    SELECT
        strftime('%Y-%m', first_order_date) AS cohort_month,
        months_since_first_order,
        COUNT(DISTINCT customer_unique_id) AS customers_active
    FROM orders_with_cohort
    WHERE months_since_first_order IN (1, 2)
    GROUP BY cohort_month, months_since_first_order
)

SELECT
    s.cohort_month,
    s.cohort_size,
    COALESCE(a1.customers_active, 0) AS customers_active_month_1,
    ROUND(100.0 * COALESCE(a1.customers_active, 0) / s.cohort_size, 2) AS retention_month_1_pct,
    COALESCE(a2.customers_active, 0) AS customers_active_month_2,
    ROUND(100.0 * COALESCE(a2.customers_active, 0) / s.cohort_size, 2) AS retention_month_2_pct
FROM cohort_sizes s
LEFT JOIN activity a1
    ON a1.cohort_month = s.cohort_month AND a1.months_since_first_order = 1
LEFT JOIN activity a2
    ON a2.cohort_month = s.cohort_month AND a2.months_since_first_order = 2
WHERE s.cohort_month NOT IN ('2018-09', '2018-10')
ORDER BY s.cohort_month;




