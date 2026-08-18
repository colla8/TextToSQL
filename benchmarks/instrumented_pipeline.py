# -*- coding: utf-8 -*-
"""
Mirrors app/pipeline.py's TextToSQLPipeline.ask() exactly, stage by stage,
so benchmark numbers map 1:1 onto real code paths -- this is deliberately a
copy, not a monkeypatch of the real pipeline, so a bug in benchmarking code
can never change production behavior. If you change app/pipeline.py's
control flow, update this to match or the benchmark silently drifts from
what's actually deployed.

Stages measured:
  retrieval          -- SchemaRetriever.get_relevant_schema (embed + FAISS),
                         skipped entirely on a retrieval-cache hit, and never
                         entered at all on a sql-cache hit
  sql_generation      -- SQLGenerator.generate (1..N Ollama calls: initial +
                          repair attempts; all captured, not just the last),
                          never entered on a sql-cache hit
  sql_execution       -- execute_sql against Postgres, skipped on a
                          result-cache hit
  answer_generation   -- AnswerGenerator.generate (1 Ollama call) -- never
                          cached, always runs

`not_found` and `validation_failed` short-circuit exactly like the real
pipeline (no execution/answer stage runs), which itself is useful signal:
if your benchmark question set is triggering these paths more than
expected, that's a retrieval-quality problem, not a speed problem, and no
amount of GPU tuning fixes it.

Caching defaults OFF here (see QueryCache(enabled=False) below): the whole
point of run_benchmark.py's `--repeats` is to measure the *uncached* cost
of each stage several times over so report.py can compute a median/p95
that means something (see run_benchmark.py's docstring). If the cache
silently kicked in from repeat_idx=1 onward, every repeat after the first
would just be measuring a cache hit, not the pipeline -- pass
`cache=QueryCache(enabled=True)` in (or use `--with-cache`, see
run_benchmark.py) to benchmark the cache itself instead.
"""

from __future__ import annotations

from typing import Any

from app.cache.query_cache import QueryCache
from app.embeddings.retriever import SchemaRetriever
from app.llm.sql_generator import (
    SQLGenerator,
    SchemaInsufficientError,
    extract_sql,
)
from app.llm.sql_validator import SchemaValidationError, validate_against_schema
from app.llm.prompt import SQL_SYSTEM_PROMPT, build_sql_prompt, build_repair_prompt
from app.llm.answer_generator import AnswerGenerator
from app.db.executor import execute_sql, UnsafeSQLError

from benchmarks.instrumented_ollama_client import InstrumentedOllamaClient
from benchmarks.resource_monitor import ResourceSampler
from benchmarks.stage_timer import Stage, StageRecord


class InstrumentedPipeline:
    def __init__(
        self,
        sampler: ResourceSampler,
        max_repair_attempts: int = 1,
        cache: QueryCache | None = None,
    ):
        self.sampler = sampler
        self.max_repair_attempts = max_repair_attempts
        self.cache = cache or QueryCache(enabled=False)

        # Separate clients per role, exactly like the real pipeline
        # (SQLGenerator: think=True; AnswerGenerator: think=False,
        # max_tokens=128) -- reusing one client with mismatched settings
        # would silently change what's being benchmarked.
        self.retriever = SchemaRetriever()
        self.sql_client = InstrumentedOllamaClient(think=False)
        self.answer_client = InstrumentedOllamaClient(think=False, max_tokens=128)

    def ask(self, question: str) -> tuple[dict[str, Any], list[StageRecord]]:
        records: list[StageRecord] = []
        cache_info = {
            "sql_cache": "miss",
            "retrieval_cache": "skipped",
            "result_cache": "miss",
        }

        schema_context: str | None = None
        allowed_schema: dict[str, list[str]] | None = None
        cached_sql = self.cache.get_sql(question)

        if cached_sql is not None:
            cache_info["sql_cache"] = "hit"
            sql = cached_sql
        else:
            with Stage("retrieval", self.sampler, records) as st:
                cached_retrieval = self.cache.get_retrieval(question)
                if cached_retrieval is not None:
                    cache_info["retrieval_cache"] = "hit"
                    schema_context, allowed_schema = cached_retrieval
                else:
                    cache_info["retrieval_cache"] = "miss"
                    schema_context, allowed_schema = self.retriever.get_relevant_schema(
                        question
                    )
                    self.cache.set_retrieval(question, schema_context, allowed_schema)
                st.meta["schema_context_chars"] = len(schema_context)
                st.meta["n_tables_retrieved"] = len(allowed_schema)
                st.meta["retrieval_cache"] = cache_info["retrieval_cache"]

            with Stage("sql_generation", self.sampler, records) as st:
                before = len(self.sql_client.call_history)
                try:
                    sql = self._generate_sql(question, schema_context, allowed_schema)
                except SchemaInsufficientError as e:
                    st.ollama_calls = self.sql_client.call_history[before:]
                    return (
                        {
                            "question": question,
                            "schema_context": schema_context,
                            "sql": None,
                            "not_found": True,
                            "reason": e.reason,
                            "cache": cache_info,
                        },
                        records,
                    )
                except SchemaValidationError as e:
                    st.ollama_calls = self.sql_client.call_history[before:]
                    return (
                        {
                            "question": question,
                            "schema_context": schema_context,
                            "sql": None,
                            "error": f"نتوانستم کوئری معتبری بسازم: {e.violations}",
                            "cache": cache_info,
                        },
                        records,
                    )
                st.ollama_calls = self.sql_client.call_history[before:]
                st.meta["repair_attempts"] = len(st.ollama_calls) - 1
                self.cache.set_sql(question, sql)

        with Stage("sql_execution", self.sampler, records) as st:
            st.meta["sql"] = sql
            cached_result = self.cache.get_result(sql)
            if cached_result is not None:
                cache_info["result_cache"] = "hit"
                columns, rows = cached_result
            else:
                try:
                    columns, rows = execute_sql(sql)
                except UnsafeSQLError as e:
                    return (
                        {
                            "question": question,
                            "schema_context": schema_context,
                            "sql": sql,
                            "error": str(e),
                            "cache": cache_info,
                        },
                        records,
                    )
                except Exception as e:
                    return (
                        {
                            "question": question,
                            "schema_context": schema_context,
                            "sql": sql,
                            "error": f"خطا در اجرای کوئری: {e}",
                            "cache": cache_info,
                        },
                        records,
                    )
                self.cache.set_result(sql, columns, rows)
            st.meta["n_rows"] = len(rows)
            st.meta["result_cache"] = cache_info["result_cache"]

        with Stage("answer_generation", self.sampler, records) as st:
            before = len(self.answer_client.call_history)
            answer = AnswerGenerator(client=self.answer_client).generate(
                question, sql, columns, rows
            )
            st.ollama_calls = self.answer_client.call_history[before:]

        return (
            {
                "question": question,
                "schema_context": schema_context,
                "sql": sql,
                "columns": columns,
                "rows": rows,
                "answer": answer,
                "cache": cache_info,
            },
            records,
        )

    def _generate_sql(
        self, question: str, schema_context: str, allowed_schema: dict
    ) -> str:
        """Same control flow as SQLGenerator.generate(), reimplemented here
        only so the instrumented client instance (with call_history) is the
        one actually used, and so every attempt -- not just the last -- is
        visible to the stage."""
        prompt = build_sql_prompt(question, schema_context)
        raw = self.sql_client.generate(prompt, system=SQL_SYSTEM_PROMPT)
        sql = extract_sql(raw)

        attempts = 0
        while attempts < self.max_repair_attempts:
            try:
                validate_against_schema(sql, allowed_schema)
                return sql
            except SchemaValidationError as e:
                attempts += 1
                repair_prompt = build_repair_prompt(
                    question, schema_context, sql, e.violations
                )
                raw = self.sql_client.generate(repair_prompt, system=SQL_SYSTEM_PROMPT)
                sql = extract_sql(raw)

        validate_against_schema(sql, allowed_schema)
        return sql
