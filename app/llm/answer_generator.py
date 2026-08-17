# -*- coding: utf-8 -*-
from app.llm.ollama_client import OllamaClient
from app.llm.sql_generator import strip_think
from app.llm.prompt import ANSWER_SYSTEM_PROMPT, build_answer_prompt


class AnswerGenerator:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient(
            think=False,
            max_tokens=128,
        )

    def generate(
        self,
        question: str,
        sql: str,
        columns: list[str],
        rows: list[tuple],
    ) -> str:
        prompt = build_answer_prompt(question, sql, columns, rows)

        raw = self.client.generate(
            prompt,
            system=ANSWER_SYSTEM_PROMPT,
        )

        return strip_think(raw)
