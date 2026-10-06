# -*- coding: utf-8 -*-
"""
Subclasses app/pipeline.py's TextToSQLPipeline instead of duplicating it.
super().__init__() wires up the exact same real components the production
pipeline uses -- SchemaRetriever, SQLGenerator (with its own repair loop),
AnswerGenerator, QueryCache -- just with InstrumentedOllamaClient instances
passed in as their client=, so every call the pipeline was already going to
make also records Ollama's raw per-call timing. Nothing about retrieval,
SQL generation/repair, execution, or answering is reimplemented here; it's
delegated straight to the inherited self.retriever / self.sql_generator /
self.cache / self.answer_generator.

ask() is still overridden (not inherited as-is): app/pipeline.py's ask()
doesn't expose per-stage hook points, and this project deliberately keeps
app/ behavior-unchanged (see benchmarks/README.md), so there's no seam to
attach Stage() timing to without an override. This override mirrors
app/pipeline.py's control flow (same cache checks, same short-circuits) but
contains no independent business logic -- if app/pipeline.py's control flow
changes, update this to match, or the two can silently drift apart.

Stages measured:
  retrieval          -- SchemaRetriever.get_relevant_schema (lexical BM25 +
                         fuzzy first, embedding + FAISS only on a lexical
                         miss -- see app/embeddings/retriever.py), skipped
                         entirely on a retrieval-cache hit, and never
                         entered at all on a sql-cache hit. On an actual
                         retriever call, this stage's meta also records
                         which path answered it (retrieval_method:
                         "lexical" | "embedding_fallback") and the
                         lexical stage's confidence score
                         (retrieval_confidence), read straight off
                         self.retriever.last_retrieval_method /
                         .last_retrieval_confidence right after the call --
                         see benchmarks/aggregate.py and benchmarks/report.py
                         for how these roll up into the report.
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

benchmarks/run_cache_benchmark.py also builds one of these directly (with
its own isolated QueryCache) to benchmark the caches themselves -- that
script is untouched by this refactor and doesn't need to be.
"""

from __future__ import annotations

from typing import Any

from app.cache.query_cache import QueryCache
from app.embeddings.retriever import SchemaRetriever
from app.llm.sql_generator import SQLGenerator, SchemaInsufficientError
from app.llm.sql_validator import SchemaValidationError
from app.llm.answer_generator import AnswerGenerator
from app.db.executor import execute_sql, UnsafeSQLError
from app.pipeline import TextToSQLPipeline

from benchmarks.instrumented_ollama_client import InstrumentedOllamaClient
from benchmarks.resource_monitor import ResourceSampler
from benchmarks.stage_timer import Stage, StageRecord


class InstrumentedPipeline(TextToSQLPipeline):
    def __init__(
        self,
        sampler: ResourceSampler,
        max_repair_attempts: int = 1,
        cache: QueryCache | None = None,
    ):
        self.sampler = sampler

        # Both clients now default to the same model (setting.OLLAMA_MODEL)
        # -- app/llm/answer_generator.py no longer passes a model= override,
        # so AnswerGenerator runs on the same qwen3:8b as SQLGenerator
        # instead of a separate smaller model. Just InstrumentedOllamaClient
        # instead of OllamaClient, so call_history is available below.
        self.sql_client = InstrumentedOllamaClient(think=False)
        self.answer_client = InstrumentedOllamaClient(think=False, max_tokens=128)

        # Real components, real wiring -- identical to how app/pipeline.py's
        # own __init__ builds its defaults, except the LLM-backed ones get
        # the instrumented client instead of AnswerGenerator/SQLGenerator's
        # own default OllamaClient. SQLGenerator.generate (repair loop
        # included) and AnswerGenerator.generate run completely unmodified
        # from here on.
        super().__init__(
            retriever=SchemaRetriever(),
            sql_generator=SQLGenerator(
                client=self.sql_client, max_repair_attempts=max_repair_attempts
            ),
            answer_generator=AnswerGenerator(client=self.answer_client),
            cache=cache or QueryCache(enabled=False),
        )

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
                    schema_context, allowed_schema = (
                        self.retriever.get_relevant_schema(question)
                    )
                    self.cache.set_retrieval(question, schema_context, allowed_schema)
                    # Only meaningful right after an actual retriever call
                    # -- a retrieval-cache hit above skips the retriever
                    # entirely, so there's no fresh method/confidence to
                    # report for this question (left unset rather than
                    # showing a stale value from some earlier question).
                    st.meta["retrieval_method"] = self.retriever.last_retrieval_method
                    st.meta["retrieval_confidence"] = (
                        self.retriever.last_retrieval_confidence
                    )
                st.meta["schema_context_chars"] = len(schema_context)
                st.meta["n_tables_retrieved"] = len(allowed_schema)
                st.meta["retrieval_cache"] = cache_info["retrieval_cache"]

            with Stage("sql_generation", self.sampler, records) as st:
                before = len(self.sql_client.call_history)
                try:
                    # Delegates to the real SQLGenerator.generate() --
                    # including its repair loop -- inherited via
                    # self.sql_generator. Nothing reimplemented here.
                    sql = self.sql_generator.generate(
                        question, schema_context, allowed_schema
                    )
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
            # Delegates to the real AnswerGenerator instance built in
            # __init__ (inherited as self.answer_generator), instead of
            # constructing a fresh one on every call.
            answer = self.answer_generator.generate(question, sql, columns, rows)
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