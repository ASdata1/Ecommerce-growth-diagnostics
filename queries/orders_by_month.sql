SELECT strftime('%Y-%m',
  order_purchase_timestamp) AS month, COUNT(*) FROM orders GROUP BY month ORDER BY month