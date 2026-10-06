# -*- coding: utf-8 -*-
"""
Generic synchronous key/value cache backed by SQLite.

Deliberately plain sqlite3, not sqlalchemy/async like api/database.py's
request log: the pipeline itself (app/pipeline.py) is fully synchronous --
network calls to Ollama, Postgres, and FAISS all block -- and it's invoked
both from a FastAPI worker thread via asyncio.to_thread (api/main.py) and
directly from CLI scripts (scripts/ask.py, benchmarks/run_benchmark.py). A
sync store keeps the cache usable from both without needing an event loop.

One short-lived connection per operation, not a shared handle -- sqlite3
connections aren't safe to share across threads, but SQLite itself handles
concurrent readers/writers fine once WAL mode is on (the same trade-off
api/database.py already makes for the request log).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import decimal as _decimal
import json
import sqlite3
import time
import uuid as _uuid
from pathlib import Path
from typing import Any, Iterator


def _json_default(obj: Any) -> Any:
    """Handles the psycopg2 row types json.dumps can't serialize natively
    (Decimal, date/datetime/time, UUID) -- only ever hit by the result
    cache, whose values come straight out of execute_sql()."""
    if isinstance(obj, _decimal.Decimal):
        return float(obj)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, _uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class SqliteKVStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:  # commits on clean exit, rolls back on exception
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace  TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )

    def get(self, namespace: str, key: str) -> Any | None:
        """Returns the deserialized value, or None on a miss or an expired
        entry. Expired rows are lazily deleted on the read that finds them
        -- no background sweeper needed at this project's traffic."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache_entries "
                "WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and expires_at < time.time():
                conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                return None
            return json.loads(value)

    def set(
        self, namespace: str, key: str, value: Any, ttl_seconds: int | None
    ) -> None:
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        payload = json.dumps(value, ensure_ascii=False, default=_json_default)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (namespace, key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (namespace, key, payload, now, expires_at),
            )

    def delete(self, namespace: str, key: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            )

    def clear_namespace(self, namespace: str) -> int:
        """Wipes every entry in one namespace (e.g. 'result' after
        db_setup/load_data.py reloads Postgres) -- returns rows removed."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ?", (namespace,)
            )
            return cur.rowcount

    def purge_expired(self) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                (time.time(),),
            )
            return cur.rowcount
