# -*- coding: utf-8 -*-
"""
Executes LLM-generated SQL against PostgreSQL, defensively:
  1. must be a single statement (no ';' hiding a second command)
  2. must start with SELECT or WITH
  3. must not contain any data/schema-mutating keyword anywhere in the
     string (this also blocks the "WITH x AS (INSERT ... RETURNING ...)
     SELECT * FROM x" data-modifying-CTE trick)
  4. runs inside an explicit READ ONLY transaction as a second layer of
     defense, in case a future edit ever loosens the checks above

This does NOT make arbitrary SQL execution "safe" in a hostile
multi-tenant sense -- it's scoped to "don't let a hallucinated or
malicious-looking LLM completion silently mutate the database in a
single-user local project."
"""

import re
import threading
from typing import Any

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

from app.core.config import setting

_DISALLOWED_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|vacuum|copy)\b",
    flags=re.IGNORECASE,
)

# A single process-wide connection pool. Without this, every question asked
# through the API opened (and never explicitly closed on error paths) a brand
# new TCP connection to Postgres - fine for one person testing with scripts/
# ask.py, but under a few concurrent Telegram users it either exhausts
# Postgres' max_connections or serializes requests behind connection setup
# latency. ThreadedConnectionPool is safe to share across the worker threads
# FastAPI runs blocking endpoints in (see api/deps.py).
_pool_lock = threading.Lock()
_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadedConnectionPool(
                    setting.PG_POOL_MIN,
                    setting.PG_POOL_MAX,
                    host=setting.PG_HOST,
                    port=setting.PG_PORT,
                    dbname=setting.PG_DATABASE,
                    user=setting.PG_USER,
                    password=setting.PG_PASSWORD,
                )
    return _pool


def close_pool() -> None:
    """Call on application shutdown to release all pooled connections."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


class UnsafeSQLError(Exception):
    pass


def validate_sql(sql: str) -> None:
    stripped = sql.strip()
    if not stripped:
        raise UnsafeSQLError("کوئری خالی است.")
    if stripped.rstrip(";").count(";") > 0:
        raise UnsafeSQLError("چند دستور SQL در یک کوئری مجاز نیست.")
    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise UnsafeSQLError("فقط کوئری‌های SELECT (یا WITH ... SELECT) مجاز هستند.")
    if _DISALLOWED_RE.search(stripped):
        raise UnsafeSQLError(
            "این کوئری شامل دستور غیرمجاز است (insert/update/delete/ddl)."
        )


def execute_sql(
    sql: str, row_limit: int = 200
) -> tuple[list[str], list[tuple[Any, ...]]]:
    validate_sql(sql)
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(sql)
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(row_limit)
        return columns, rows
    except psycopg2.Error:
        # connection may be in a bad transaction state; don't hand it back
        conn.close()
        raise
    finally:
        if not conn.closed:
            pool.putconn(conn)
