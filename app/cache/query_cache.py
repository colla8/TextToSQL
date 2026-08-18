# -*- coding: utf-8 -*-
"""
QueryCache -- the three caches in the pipeline's caching architecture, each
a thin, typed wrapper around a shared SqliteKVStore:

  sql        question_hash -> generated SQL
             hit skips retrieval + SQL generation + validation entirely
  retrieval  question_hash -> (schema_context, allowed_schema)
             hit skips embedding + FAISS; only consulted on a sql-cache miss
  result     sql_hash      -> (columns, rows)
             hit skips the Postgres round trip

See app/pipeline.py for how these compose into the request flow:

    question -> [SQL cache] --hit--------------------------+
                     | miss                                |
                [retrieval cache] --hit--+                 |
                     | miss              |                 |
                [embed + FAISS]          |                 |
                     +------------------>|                 |
                [SQL generation + validation]               |
                     | (store into SQL cache)               |
                     +--------------------------------------+
                                         v
                                  [result cache] --hit--+
                                         | miss          |
                                    [PostgreSQL]         |
                                         | (store)        |
                                         +----------------+
                                         v
                                 [answer generation]  (never cached --
                                 depends on exact question phrasing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.cache.keys import question_cache_key, sql_cache_key
from app.cache.store import SqliteKVStore
from app.core.config import setting

NS_SQL = "sql"
NS_RETRIEVAL = "retrieval"
NS_RESULT = "result"


class QueryCache:
    def __init__(self, db_path: Path | None = None, enabled: bool | None = None):
        # enabled=None -> defer to setting.CACHE_ENABLED. An explicit True/
        # False lets a caller override that regardless of config, which
        # benchmarks/instrumented_pipeline.py relies on (see its docstring)
        # to keep repeated benchmark runs measuring real, uncached cost.
        self.enabled = setting.CACHE_ENABLED if enabled is None else enabled
        self.store = SqliteKVStore(db_path or setting.CACHE_DB_PATH)

    # -- SQL cache --------------------------------------------------------
    def get_sql(self, question: str) -> str | None:
        if not self.enabled:
            return None
        hit = self.store.get(NS_SQL, question_cache_key(question))
        return hit["sql"] if hit else None

    def set_sql(self, question: str, sql: str) -> None:
        if not self.enabled:
            return
        self.store.set(
            NS_SQL,
            question_cache_key(question),
            {"sql": sql},
            setting.SQL_CACHE_TTL_SECONDS,
        )

    # -- Retrieval cache ----------------------------------------------------
    def get_retrieval(self, question: str) -> tuple[str, dict[str, list[str]]] | None:
        if not self.enabled:
            return None
        hit = self.store.get(NS_RETRIEVAL, question_cache_key(question))
        if hit is None:
            return None
        return hit["schema_context"], hit["allowed_schema"]

    def set_retrieval(
        self, question: str, schema_context: str, allowed_schema: dict[str, list[str]]
    ) -> None:
        if not self.enabled:
            return
        self.store.set(
            NS_RETRIEVAL,
            question_cache_key(question),
            {"schema_context": schema_context, "allowed_schema": allowed_schema},
            setting.RETRIEVAL_CACHE_TTL_SECONDS,
        )

    # -- Result cache -------------------------------------------------------
    def get_result(self, sql: str) -> tuple[list[str], list[list[Any]]] | None:
        if not self.enabled:
            return None
        hit = self.store.get(NS_RESULT, sql_cache_key(sql))
        if hit is None:
            return None
        return hit["columns"], hit["rows"]

    def set_result(self, sql: str, columns: list[str], rows: list[tuple]) -> None:
        if not self.enabled:
            return
        self.store.set(
            NS_RESULT,
            sql_cache_key(sql),
            # rows come back from psycopg2 as tuples -- json can't encode
            # tuples natively, and round-tripping through JSON turns them
            # into lists anyway, so normalize that up front.
            {"columns": columns, "rows": [list(r) for r in rows]},
            setting.RESULT_CACHE_TTL_SECONDS,
        )

    # -- Maintenance --------------------------------------------------------
    def invalidate_results(self) -> int:
        """Call after loading new data into Postgres (see
        db_setup/load_data.py) so stale cached rows don't outlive the data
        they came from. SQL/retrieval caches don't need this -- they only
        depend on the schema and the question, not on row contents."""
        return self.store.clear_namespace(NS_RESULT)

    def invalidate_all(self) -> int:
        return (
            self.store.clear_namespace(NS_SQL)
            + self.store.clear_namespace(NS_RETRIEVAL)
            + self.store.clear_namespace(NS_RESULT)
        )
