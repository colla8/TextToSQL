# -*- coding: utf-8 -*-
"""
End-to-end tests of app/pipeline.py with only the *external services* faked:
Ollama (scripted replies), PostgreSQL (execute_sql patched) and the answer
LLM. Everything else is the real code: lexical retrieval over data/metadata,
sqlglot schema validation, the repair loop, and the SQLite query cache
(in a temp dir).

Run with: python -m unittest tests.test_pipeline -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.cache.query_cache import QueryCache
from app.db.executor import UnsafeSQLError, validate_sql
from app.embeddings.retriever import SchemaRetriever
from app.llm.sql_generator import SQLGenerator
from app.pipeline import TextToSQLPipeline

QUESTION = "قیمت محصولات چقدر است؟"
GOOD_SQL = "SELECT product_name, price FROM products"


class ScriptedOllama:
    """Returns the queued replies in order and counts calls."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, prompt, system=None, max_tokens=None):
        self.calls += 1
        return self.replies.pop(0)


class FakeAnswerGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, question, sql, columns, rows):
        self.calls += 1
        return f"{len(rows)} ردیف"


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = QueryCache(Path(self._tmp.name) / "cache.db", enabled=True)
        self.answers = FakeAnswerGenerator()

    def make_pipeline(self, *llm_replies: str) -> tuple[TextToSQLPipeline, ScriptedOllama]:
        llm = ScriptedOllama(*llm_replies)
        pipeline = TextToSQLPipeline(
            retriever=SchemaRetriever(),
            sql_generator=SQLGenerator(client=llm, max_repair_attempts=1),
            answer_generator=self.answers,
            cache=self.cache,
        )
        return pipeline, llm


class TestPipeline(PipelineTestCase):
    def test_happy_path_then_full_cache_hit(self):
        pipeline, llm = self.make_pipeline(f"```sql\n{GOOD_SQL};\n```")
        rows = [("لپ‌تاپ", 100.5), ("موبایل", 50)]

        with mock.patch(
            "app.pipeline.execute_sql", return_value=(["product_name", "price"], rows)
        ) as run_sql:
            first = pipeline.ask(QUESTION)
            second = pipeline.ask(QUESTION)

        self.assertEqual(first["sql"], GOOD_SQL)
        self.assertEqual(first["columns"], ["product_name", "price"])
        self.assertEqual(first["answer"], "2 ردیف")
        self.assertEqual(
            first["cache"],
            {"sql_cache": "miss", "retrieval_cache": "miss", "result_cache": "miss"},
        )
        # second ask: SQL + result served from cache -> no LLM, no DB call
        self.assertEqual(second["cache"]["sql_cache"], "hit")
        self.assertEqual(second["cache"]["result_cache"], "hit")
        self.assertEqual(second["sql"], GOOD_SQL)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(run_sql.call_count, 1)
        # answer is deliberately never cached
        self.assertEqual(self.answers.calls, 2)

    def test_retrieval_cache_used_when_sql_cache_missing(self):
        from app.cache.keys import question_cache_key
        from app.cache.query_cache import NS_SQL

        pipeline, llm = self.make_pipeline(
            f"```sql\n{GOOD_SQL}\n```", f"```sql\n{GOOD_SQL}\n```"
        )
        with mock.patch("app.pipeline.execute_sql", return_value=(["a"], [(1,)])):
            pipeline.ask(QUESTION)
            # drop only the SQL entry; the retrieval entry stays
            self.cache.store.delete(NS_SQL, question_cache_key(QUESTION))
            with mock.patch.object(
                pipeline.retriever, "get_relevant_schema",
                side_effect=AssertionError("retrieval cache should have been used"),
            ):
                second = pipeline.ask(QUESTION)

        self.assertEqual(second["cache"]["sql_cache"], "miss")
        self.assertEqual(second["cache"]["retrieval_cache"], "hit")
        self.assertEqual(llm.calls, 2)

    def test_not_found_is_reported_not_fabricated(self):
        pipeline, _ = self.make_pipeline("NOT_FOUND: اطلاعات ایمیل در دسترس نیست")
        with mock.patch("app.pipeline.execute_sql") as run_sql:
            result = pipeline.ask(QUESTION)
        self.assertTrue(result["not_found"])
        self.assertIsNone(result["sql"])
        self.assertIn("ایمیل", result["reason"])
        run_sql.assert_not_called()

    def test_hallucinated_column_is_repaired_once(self):
        pipeline, llm = self.make_pipeline(
            "SELECT product_name, cost FROM products",  # `cost` doesn't exist
            f"```sql\n{GOOD_SQL}\n```",
        )
        with mock.patch("app.pipeline.execute_sql", return_value=(["a"], [(1,)])):
            result = pipeline.ask(QUESTION)
        self.assertEqual(result["sql"], GOOD_SQL)
        self.assertEqual(llm.calls, 2)

    def test_unrepairable_hallucination_is_refused_and_never_executed(self):
        bad = "SELECT cost FROM products"
        pipeline, _ = self.make_pipeline(bad, bad)
        with mock.patch("app.pipeline.execute_sql") as run_sql:
            result = pipeline.ask(QUESTION)
        self.assertIsNone(result["sql"])
        self.assertIn("error", result)
        run_sql.assert_not_called()

    def test_execution_error_is_returned_and_sql_not_result_cached(self):
        pipeline, _ = self.make_pipeline(f"```sql\n{GOOD_SQL}\n```")
        with mock.patch("app.pipeline.execute_sql", side_effect=RuntimeError("db down")):
            result = pipeline.ask(QUESTION)
        self.assertIn("db down", result["error"])
        self.assertIsNone(self.cache.get_result(GOOD_SQL))

    def test_unsafe_sql_from_executor_is_returned_as_error(self):
        pipeline, _ = self.make_pipeline(f"```sql\n{GOOD_SQL}\n```")
        with mock.patch("app.pipeline.execute_sql", side_effect=UnsafeSQLError("no")):
            result = pipeline.ask(QUESTION)
        self.assertEqual(result["error"], "no")


class TestExecutorSafety(unittest.TestCase):
    def test_select_and_with_are_allowed(self):
        validate_sql("SELECT 1")
        validate_sql("WITH x AS (SELECT 1) SELECT * FROM x;")

    def test_writes_and_multi_statements_are_rejected(self):
        for bad in (
            "DELETE FROM products",
            "UPDATE products SET price = 0",
            "DROP TABLE products",
            "SELECT 1; DROP TABLE products",
            "",
        ):
            with self.subTest(sql=bad):
                with self.assertRaises(UnsafeSQLError):
                    validate_sql(bad)


if __name__ == "__main__":
    unittest.main()
