# -*- coding: utf-8 -*-
"""
Integration tests against a REAL PostgreSQL loaded with db_setup/load_data.py.
Skipped automatically when Postgres isn't reachable (connection settings come
from .env / app/core/config.py: PG_HOST, PG_PORT, PG_DATABASE, PG_USER,
PG_PASSWORD), so they never break a machine/CI without a database.

Run with: python -m unittest tests.test_postgres_integration -v
"""

from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

import psycopg2

from app.core.config import setting
from app.db.executor import UnsafeSQLError, close_pool, execute_sql

CSV_DIR = Path(__file__).resolve().parents[1] / "data" / "csv-data" / "csv"


def _postgres_ready() -> bool:
    try:
        conn = psycopg2.connect(
            host=setting.PG_HOST,
            port=setting.PG_PORT,
            dbname=setting.PG_DATABASE,
            user=setting.PG_USER,
            password=setting.PG_PASSWORD,
            connect_timeout=2,
        )
    except Exception:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM products")
            return cur.fetchone()[0] > 0
    except Exception:
        return False  # connected, but data not loaded
    finally:
        conn.close()


@unittest.skipUnless(_postgres_ready(), "PostgreSQL with loaded data not reachable")
class TestPostgresIntegration(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        close_pool()

    def test_group_by_matches_the_source_csv(self):
        columns, rows = execute_sql(
            "SELECT city, COUNT(*) AS n FROM customers GROUP BY city"
        )
        with open(CSV_DIR / "customers.csv", encoding="utf-8-sig") as f:
            expected = Counter(r["city"] for r in csv.DictReader(f))
        self.assertEqual(columns, ["city", "n"])
        self.assertEqual({c: n for c, n in rows}, dict(expected))

    def test_join_query_runs(self):
        _, rows = execute_sql(
            "SELECT c.category_name, COUNT(*) FROM products p "
            "JOIN categories c ON p.category_id = c.category_id "
            "GROUP BY c.category_name"
        )
        self.assertTrue(rows)

    def test_row_limit_is_applied(self):
        _, rows = execute_sql("SELECT * FROM order_items", row_limit=10)
        self.assertEqual(len(rows), 10)

    def test_write_is_blocked_and_data_untouched(self):
        _, before = execute_sql("SELECT COUNT(*) FROM products")
        with self.assertRaises(UnsafeSQLError):
            execute_sql("DELETE FROM products")
        _, after = execute_sql("SELECT COUNT(*) FROM products")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
