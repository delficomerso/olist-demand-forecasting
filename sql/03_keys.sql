-- Primary keys
ALTER TABLE orders ADD PRIMARY KEY (order_id);
ALTER TABLE products ADD PRIMARY KEY (product_id);
ALTER TABLE product_category_name_translation ADD PRIMARY KEY (product_category_name);
ALTER TABLE order_items ADD PRIMARY KEY (order_id, order_item_id);

-- Foreign keys
ALTER TABLE order_items ADD FOREIGN KEY (order_id) REFERENCES orders (order_id);
ALTER TABLE order_items ADD FOREIGN KEY (product_id) REFERENCES products (product_id);