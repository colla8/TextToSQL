# -*- coding: utf-8 -*-
import re

from app.llm.ollama_client import OllamaClient
from app.llm.prompt import (
    SQL_SYSTEM_PROMPT,
    NOT_FOUND_TOKEN,
    build_sql_prompt,
    build_repair_prompt,
)
from app.llm.sql_validator import validate_against_schema, SchemaValidationError

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_NOT_FOUND_RE = re.compile(
    rf"^\s*{NOT_FOUND_TOKEN}\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL
)


class SchemaInsufficientError(Exception):
    """Raised when the model correctly determined the retrieved schema
    can't answer the question. `.reason` is the model's stated reason,
    shown to the user instead of a fabricated SQL query."""

    def __init__(self, reason: str):
        self.reason = reason.strip()
        super().__init__(self.reason)


def strip_think(text: str) -> str:
    """qwen3 (and similar reasoning models) may emit <think>...</think>
    reasoning before the actual answer. Strip it so it never leaks into
    the parsed SQL or the final user-facing answer."""
    return _THINK_RE.sub("", text).strip()


def extract_sql(raw_response: str) -> str:
    """Returns the SQL string. Raises SchemaInsufficientError if the model
    used the NOT_FOUND protocol instead of writing a query."""
    text = strip_think(raw_response)

    not_found = _NOT_FOUND_RE.match(text)
    if not_found:
        raise SchemaInsufficientError(not_found.group(1))

    match = _CODE_FENCE_RE.search(text)
    sql = match.group(1).strip() if match else text.strip()

    # the model might still emit NOT_FOUND *inside* a fenced block
    not_found_in_fence = _NOT_FOUND_RE.match(sql)
    if not_found_in_fence:
        raise SchemaInsufficientError(not_found_in_fence.group(1))

    return sql.rstrip(";").strip()


class SQLGenerator:
    def __init__(
        self, client: OllamaClient | None = None, max_repair_attempts: int = 1
    ):
        # think=True: this prompt's whole strategy (schema linking -> join
        # path -> filters -> fan-out check -> projection -> optimization)
        # is designed to run inside qwen3's native <think> channel, which
        # is stripped before SQL extraction. Explicit, not left to Ollama's
        # per-model default, since the anti-hallucination behavior depends on it.
        self.client = client or OllamaClient(think=True)
        self.max_repair_attempts = max_repair_attempts

    def generate(
        self, question: str, schema_context: str, allowed_schema: dict[str, list[str]]
    ) -> str:
        """Generates SQL, then deterministically validates it against
        allowed_schema (not another LLM call -- see sql_validator.py).
        On a violation, sends the exact error back to the model for a
        bounded number of repair attempts before giving up."""
        prompt = build_sql_prompt(question, schema_context)
        raw = self.client.generate(prompt, system=SQL_SYSTEM_PROMPT)
        sql = extract_sql(raw)  # may raise SchemaInsufficientError -- let it propagate

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
                raw = self.client.generate(repair_prompt, system=SQL_SYSTEM_PROMPT)
                sql = extract_sql(raw)

        # final attempt, let a real failure surface with full detail
        validate_against_schema(sql, allowed_schema)
        return sql
