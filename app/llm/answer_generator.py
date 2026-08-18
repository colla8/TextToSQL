# -*- coding: utf-8 -*-
from app.llm.ollama_client import OllamaClient
from app.llm.sql_generator import strip_think
from app.llm.prompt import ANSWER_SYSTEM_PROMPT, build_answer_prompt

# Base budget for a one-line answer ("۴۲ سفارش ثبت شده است.") plus a
# per-row allowance for list-style answers ("۱۰ مشتری برتر را نشان بده").
# A Persian line like "۱. نگار رضایی: ۶۸۷ میلیون تومان" runs ~15-25 tokens,
# so a fixed 128-token cap (the old default) truncates mid-list as soon as
# the result has more than ~4-5 rows -- this was cutting a real 10-row
# answer down to 4 customers. Capped at MAX_ANSWER_TOKENS so a huge
# preview (build_answer_prompt caps at 20 rows anyway) can't blow past
# setting.NUM_CTX.
_BASE_ANSWER_TOKENS = 64
_TOKENS_PER_ROW = 30
_MAX_ANSWER_TOKENS = 1024


def _answer_token_budget(n_rows: int) -> int:
    return min(_MAX_ANSWER_TOKENS, _BASE_ANSWER_TOKENS + n_rows * _TOKENS_PER_ROW)


class AnswerGenerator:
    def __init__(self, client: OllamaClient | None = None):
        # max_tokens=128 here is just the client's fallback default (used
        # if generate() is ever called without going through this class'
        # generate()) -- the real per-request budget is computed in
        # generate() below and passed as an explicit override.
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
            max_tokens=_answer_token_budget(len(rows)),
        )

        return strip_think(raw)
