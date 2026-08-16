-- Text-to-SQL project schema
-- Order matters: parents before children (FK dependencies)

DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS categories CASCADE;

CREATE TABLE categories (
    category_id     INTEGER PRIMARY KEY,
    category_name   VARCHAR(100) NOT NULL
);

CREATE TABLE customers (
    customer_id         INTEGER PRIMARY KEY,
    first_name          VARCHAR(50) NOT NULL,
    last_name           VARCHAR(50) NOT NULL,
    city                VARCHAR(50),
    phone               VARCHAR(11),
    email               VARCHAR(150),
    registration_date   DATE
);

CREATE TABLE products (
    product_id       INTEGER PRIMARY KEY,
    product_name     VARCHAR(150) NOT NULL,
    category_id      INTEGER NOT NULL REFERENCES categories(category_id),
    brand            VARCHAR(100),
    price            BIGINT NOT NULL CHECK (price >= 0),
    stock_quantity   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    order_id         INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date       TIMESTAMP NOT NULL,
    status           VARCHAR(30) NOT NULL,
    shipping_city    VARCHAR(50),
    payment_method   VARCHAR(50)
);

CREATE TABLE order_items (
    order_item_id   INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      BIGINT NOT NULL CHECK (unit_price >= 0)
);

CREATE TABLE reviews (
    review_id      INTEGER PRIMARY KEY,
    product_id     INTEGER NOT NULL REFERENCES products(product_id),
    customer_id    INTEGER NOT NULL REFERENCES customers(customer_id),
    order_id       INTEGER NOT NULL REFERENCES orders(order_id),
    rating         SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment        TEXT,
    review_date    DATE
);

-- helpful indexes for join-heavy text-to-SQL queries
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_reviews_customer ON reviews(customer_id);
CREATE INDEX idx_reviews_order ON reviews(order_id);