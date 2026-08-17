-- Q3: PAYMENT / SEGMENT CUT
-- How does average order value and average payment installments vary by customer state?
--

-- TODO: write the aggregation query here.
-- Expected output columns: customer_state, num_orders, avg_order_value, avg_installments
-- Sort by avg_order_value descending. Consider filtering out states with very few orders
-- (e.g. HAVING num_orders >= 30) so the average isn't noise from 1-2 customers.


SELECT
    c.customer_state,
    ROUND(AVG(p.payment_value), 2) AS avg_payment_val,
    COUNT(o.order_id) AS num_orders,
    ROUND(AVG(p.payment_installments)) AS avg_payment_instals
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    JOIN order_payments p 
        ON o.order_id = p.order_id
    GROUP BY c.customer_state
    HAVING COUNT(o.order_id)>= 100
    ORDER BY avg_payment_val ASC
