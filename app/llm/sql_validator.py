# -*- coding: utf-8 -*-
"""
Deterministically checks a generated SQL query against the schema that was
actually retrieved for it, using sqlglot's qualifier (real SQL parsing +
alias resolution -- not a second LLM call, so it can't hallucinate too).

This is the second line of defense against hallucination: the prompt tells
the model not to invent identifiers, but prompts are not guarantees. This
catches it deterministically before the query ever touches PostgreSQL.
"""

import sqlglot
from sqlglot.optimizer.qualify import qualify


class SchemaValidationError(Exception):
    """Raised when the SQL references a table/column not present in the
    allowed schema. `.violations` is a human-readable description usable
    directly in a repair prompt."""

    def __init__(self, violations: str):
        self.violations = violations
        super().__init__(violations)


def build_allowed_schema(
    allowed_schema: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    """sqlglot's qualify() wants {table: {column: type}}; we don't track
    types here so a placeholder type is fine -- only presence is checked."""
    return {
        table: {col: "TEXT" for col in cols} for table, cols in allowed_schema.items()
    }


def validate_against_schema(sql: str, allowed_schema: dict[str, list[str]]) -> None:
    """Raises SchemaValidationError if `sql` references any table or column
    not present in `allowed_schema` ({table_name: [column_name, ...]})."""
    sqlglot_schema = build_allowed_schema(allowed_schema)

    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception as e:
        raise SchemaValidationError(f"کوئری قابل‌پارس نبود (syntax نامعتبر): {e}")

    try:
        qualify(
            tree,
            schema=sqlglot_schema,
            dialect="postgres",
            validate_qualify_columns=True,
        )
    except Exception as e:
        raise SchemaValidationError(
            f"از جدول یا ستونی استفاده شده که در schema موجود نیست: {e}"
        )
