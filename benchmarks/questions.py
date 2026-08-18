# -*- coding: utf-8 -*-
"""
Fixed benchmark question set over the online_store schema.

The selected questions intentionally cover different workload types:
- simple lookup
- filtering
- join
- aggregation
- out-of-scope / NOT_FOUND

This keeps the benchmark small enough for repeated runs while still
covering the main pipeline behaviors.
"""

QUESTIONS = [
    # -- simple lookup --
    {
        "id": "simple_1",
        "category": "simple_lookup",
        "text": "لیست همه دسته‌بندی‌های محصولات چیست؟",
        "expect": "ok",
    },
    # -- filter --
    {
        "id": "filter_1",
        "category": "filter",
        "text": "چند سفارش در وضعیت لغو شده هستند؟",
        "expect": "ok",
    },
    # -- join --
    {
        "id": "join_1",
        "category": "join",
        "text": "نام محصولاتی که در دسته‌بندی لوازم خانگی هستند چیست؟",
        "expect": "ok",
    },
    # -- aggregation --
    {
        "id": "agg_2",
        "category": "aggregation",
        "text": "مجموع مبلغ خرید هر مشتری چقدر است؟ ۱۰ مشتری برتر را نشان بده.",
        "expect": "ok",
    },
    # -- out of scope --
    {
        "id": "oos_1",
        "category": "out_of_scope",
        "text": "درآمد شرکت در سال گذشته نسبت به سال قبل از آن چند درصد رشد کرده؟",
        "expect": "not_found",
    },
]


def by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for q in QUESTIONS:
        out.setdefault(q["category"], []).append(q)
    return out
