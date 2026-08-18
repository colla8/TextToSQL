# -*- coding: utf-8 -*-
from typing import Any

from app.cache.query_cache import QueryCache
from app.embeddings.retriever import SchemaRetriever
from app.llm.sql_generator import SQLGenerator, SchemaInsufficientError
from app.llm.sql_validator import SchemaValidationError
from app.llm.answer_generator import AnswerGenerator
from app.db.executor import execute_sql, UnsafeSQLError


class TextToSQLPipeline:
    def __init__(
        self,
        retriever: SchemaRetriever | None = None,
        sql_generator: SQLGenerator | None = None,
        answer_generator: AnswerGenerator | None = None,
        cache: QueryCache | None = None,
    ):
        self.retriever = retriever or SchemaRetriever()
        self.sql_generator = sql_generator or SQLGenerator()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.cache = cache or QueryCache()

    def ask(self, question: str) -> dict[str, Any]:
        # "cache" in the returned dict is debug/observability info only --
        # api/schemas.py's QuestionResponse doesn't surface it to the
        # Telegram user, but it's handy in logs/scripts/ask.py output to
        # see why a question came back fast.
        cache_info = {
            "sql_cache": "miss",
            "retrieval_cache": "skipped",  # only consulted on a sql-cache miss
            "result_cache": "miss",
        }

        schema_context: str | None = None
        cached_sql = self.cache.get_sql(question)

        if cached_sql is not None:
            # SQL cache HIT -- shortcut straight past retrieval, generation
            # and validation. This exact (normalized) question already
            # mapped to this SQL, and re-deriving it would cost the whole
            # embed+FAISS+LLM chain for a query we're confident is valid
            # (it only ever gets cached after passing sql_validator.py).
            cache_info["sql_cache"] = "hit"
            sql = cached_sql
        else:
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

            try:
                sql = self.sql_generator.generate(
                    question, schema_context, allowed_schema
                )
            except SchemaInsufficientError as e:
                # the model correctly determined it can't answer from the
                # retrieved schema -- surface that honestly, no fabricated SQL
                return {
                    "question": question,
                    "schema_context": schema_context,
                    "sql": None,
                    "not_found": True,
                    "reason": e.reason,
                    "cache": cache_info,
                }
            except SchemaValidationError as e:
                # exhausted repair attempts and the model still hallucinated
                # an identifier -- refuse rather than run something wrong
                return {
                    "question": question,
                    "schema_context": schema_context,
                    "sql": None,
                    "error": f"نتوانستم کوئری معتبری بسازم: {e.violations}",
                    "cache": cache_info,
                }

            # only cache SQL that made it past validation -- never a query
            # that later errors out at execution
            self.cache.set_sql(question, sql)

        cached_result = self.cache.get_result(sql)
        if cached_result is not None:
            cache_info["result_cache"] = "hit"
            columns, rows = cached_result
        else:
            try:
                columns, rows = execute_sql(sql)
            except UnsafeSQLError as e:
                return {
                    "question": question,
                    "schema_context": schema_context,
                    "sql": sql,
                    "error": str(e),
                    "cache": cache_info,
                }
            except Exception as e:
                return {
                    "question": question,
                    "schema_context": schema_context,
                    "sql": sql,
                    "error": f"خطا در اجرای کوئری: {e}",
                    "cache": cache_info,
                }
            self.cache.set_result(sql, columns, rows)

        # Never cached: the answer is phrased for this exact question text
        # (including cases where the SQL was reused from cache for a
        # differently-worded-but-equivalent question), so caching it by SQL
        # hash could hand back a reply that doesn't match how this question
        # was asked.
        answer = self.answer_generator.generate(question, sql, columns, rows)
        return {
            "question": question,
            "schema_context": schema_context,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "answer": answer,
            "cache": cache_info,
        }
