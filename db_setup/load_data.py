import argparse
import csv
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_FILE = SCRIPT_DIR / "schema.sql"
CSV_DIR = PROJECT_ROOT / "data" / "csv-data" / "csv"
# order matters: parents before children (FK dependencies)
TABLES = [
    ("categories", ["category_id", "category_name"]),
    (
        "customers",
        [
            "customer_id",
            "first_name",
            "last_name",
            "city",
            "phone",
            "email",
            "registration_date",
        ],
    ),
    (
        "products",
        [
            "product_id",
            "product_name",
            "category_id",
            "brand",
            "price",
            "stock_quantity",
        ],
    ),
    (
        "orders",
        [
            "order_id",
            "customer_id",
            "order_date",
            "status",
            "shipping_city",
            "payment_method",
        ],
    ),
    (
        "order_items",
        ["order_item_id", "order_id", "product_id", "quantity", "unit_price"],
    ),
    (
        "reviews",
        [
            "review_id",
            "product_id",
            "customer_id",
            "order_id",
            "rating",
            "comment",
            "review_date",
        ],
    ),
]


def get_connection(args):
    return psycopg2.connect(
        host=args.host or os.environ.get("PGHOST", "localhost"),
        port=args.port or os.environ.get("PGPORT", "5432"),
        dbname=args.db or os.environ.get("PGDATABASE"),
        user=args.user or os.environ.get("PGUSER"),
        password=args.password or os.environ.get("PGPASSWORD"),
    )


def run_schema(conn):
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        ddl = f.read()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print("schema.sql applied.")


def read_csv_rows(table_name, columns):
    path = CSV_DIR / f"{table_name}.csv"
    if not path.exists():
        sys.exit(f"CSV not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"{table_name}.csv is missing columns: {missing}")
        rows = [
            tuple(row[c] if row[c] != "" else None for c in columns) for row in reader
        ]
    return rows


def load_table(conn, table_name, columns):
    rows = read_csv_rows(table_name, columns)
    if not rows:
        print(f"{table_name}: no rows found, skipped.")
        return
    cols_sql = ", ".join(columns)
    query = f"INSERT INTO {table_name} ({cols_sql}) VALUES %s"
    with conn.cursor() as cur:
        execute_values(cur, query, rows, page_size=500)
    conn.commit()
    print(f"{table_name}: {len(rows)} rows loaded.")


def main():
    parser = argparse.ArgumentParser(description="Load CSV dataset into PostgreSQL")
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--db")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="don't run schema.sql, just load data",
    )
    args = parser.parse_args()

    conn = get_connection(args)
    try:
        if not args.skip_schema:
            run_schema(conn)
        for table_name, columns in TABLES:
            load_table(conn, table_name, columns)
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
