# -*- coding: utf-8 -*-
from typing import Any

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
    ):
        self.retriever = retriever or SchemaRetriever()
        self.sql_generator = sql_generator or SQLGenerator()
        self.answer_generator = answer_generator or AnswerGenerator()

    def ask(self, question: str) -> dict[str, Any]:
        schema_context, allowed_schema = self.retriever.get_relevant_schema(question)

        try:
            sql = self.sql_generator.generate(question, schema_context, allowed_schema)
        except SchemaInsufficientError as e:
            # the model correctly determined it can't answer from the
            # retrieved schema -- surface that honestly, no fabricated SQL
            return {
                "question": question,
                "schema_context": schema_context,
                "sql": None,
                "not_found": True,
                "reason": e.reason,
            }
        except SchemaValidationError as e:
            # exhausted repair attempts and the model still hallucinated
            # an identifier -- refuse rather than run something wrong
            return {
                "question": question,
                "schema_context": schema_context,
                "sql": None,
                "error": f"نتوانستم کوئری معتبری بسازم: {e.violations}",
            }

        try:
            columns, rows = execute_sql(sql)
        except UnsafeSQLError as e:
            return {
                "question": question,
                "schema_context": schema_context,
                "sql": sql,
                "error": str(e),
            }
        except Exception as e:
            return {
                "question": question,
                "schema_context": schema_context,
                "sql": sql,
                "error": f"خطا در اجرای کوئری: {e}",
            }

        answer = self.answer_generator.generate(question, sql, columns, rows)
        return {
            "question": question,
            "schema_context": schema_context,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "answer": answer,
        }
