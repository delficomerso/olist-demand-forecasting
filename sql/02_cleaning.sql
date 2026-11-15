-- Convert text date columns to proper timestamps
ALTER TABLE orders
    ALTER COLUMN order_purchase_timestamp TYPE timestamp USING order_purchase_timestamp::timestamp,
    ALTER COLUMN order_approved_at TYPE timestamp USING order_approved_at::timestamp,
    ALTER COLUMN order_delivered_carrier_date TYPE timestamp USING order_delivered_carrier_date::timestamp,
    ALTER COLUMN order_delivered_customer_date TYPE timestamp USING order_delivered_customer_date::timestamp,
    ALTER COLUMN order_estimated_delivery_date TYPE timestamp USING order_estimated_delivery_date::timestamp;

ALTER TABLE order_items
    ALTER COLUMN shipping_limit_date TYPE timestamp USING shipping_limit_date::timestamp;