-- 1. Table structure: run `\d orders` in psql (it is a psql command, not SQL)

-- 2. Orders per status
SELECT order_status, COUNT(*) AS n_orders
FROM orders
GROUP BY order_status
ORDER BY n_orders DESC;

-- 3. Date range (the column is stored as text, so we cast it)
SELECT
    MIN(order_purchase_timestamp::timestamp) AS first_order,
    MAX(order_purchase_timestamp::timestamp) AS last_order
FROM orders;

-- 4. Orders per month
SELECT
    DATE_TRUNC('month', order_purchase_timestamp::timestamp) AS month,
    COUNT(*) AS n_orders
FROM orders
GROUP BY month
ORDER BY month;

-- 5. Top 10 categories by items sold
SELECT t.product_category_name_english AS category, COUNT(*) AS items_sold
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
JOIN product_category_name_translation t
    ON t.product_category_name = p.product_category_name
GROUP BY category
ORDER BY items_sold DESC
LIMIT 10;