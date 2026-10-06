# -*- coding: utf-8 -*-
"""
Question set for benchmarks/run_cache_benchmark.py -- separate from
benchmarks/questions.py (which drives run_benchmark.py's stage-timing
benchmark) because these questions serve a different purpose: each
`paraphrases` entry is a deliberate *surface* variation of `canonical` --
extra whitespace, missing/duplicated punctuation, Arabic vs Persian
letterforms, Persian vs Western digits -- the exact classes of noise
app/cache/keys.py's normalize_question() exists to collapse.

These are NOT semantic paraphrases ("how many orders" vs "count the
orders"). A cache keyed by exact question hash can never be expected to
hit across genuinely different wording -- that would need semantic
matching, which this project deliberately doesn't do (see the sql cache's
docstring in app/cache/query_cache.py: hit means "this normalized text was
asked before", not "something like this was asked before"). What's being
tested here is narrower and more mechanical: does normalize_question()
correctly treat two renderings of the *same* text as the same cache key.

Only `canonical` and `paraphrases` are used for the SQL-cache /
normalization passes (cold, warm_repeat, warm_paraphrase). The
retrieval-cache and result-cache isolation passes added in
run_cache_benchmark.py (warm_retrieval_only, warm_result_only) reuse just
`canonical` -- they don't need paraphrases, since they isolate a cache
layer by selectively deleting other namespaces, not by varying the
question text.

Categories intentionally mirror benchmarks/questions.py's coverage
(simple_lookup, filter, aggregation, join) so cache behavior can be
compared against the uncached stage-timing numbers for the same kind of
query.
"""

CACHE_QUESTIONS = [
    {
        "id": "cache_agg_1",
        "category": "aggregation",
        "canonical": "مجموع مبلغ خرید هر مشتری چقدر است؟ ۱۰ مشتری برتر را نشان بده.",
        "paraphrases": [
            # extra / doubled whitespace
            "مجموع مبلغ خرید هر مشتری چقدر است؟   ۱۰ مشتری برتر را نشان بده.",
            # Arabic yeh (ي) instead of Persian yeh (ی) in "مشتری"/"چقدر"
            "مجموع مبلغ خريد هر مشتري چقدر است؟ ۱۰ مشتري برتر را نشان بده.",
            # missing final period, doubled question marks
            "مجموع مبلغ خرید هر مشتری چقدر است؟؟ ۱۰ مشتری برتر را نشان بده",
            # Western digits instead of Persian digits
            "مجموع مبلغ خرید هر مشتری چقدر است؟ 10 مشتری برتر را نشان بده.",
            # leading/trailing whitespace only
            "  مجموع مبلغ خرید هر مشتری چقدر است؟ ۱۰ مشتری برتر را نشان بده.  ",
        ],
    },
    {
        "id": "cache_filter_1",
        "category": "filter",
        "canonical": "چند سفارش در وضعیت لغو شده هستند؟",
        "paraphrases": [
            "چند سفارش در وضعیت لغو شده هستند؟ ",
            # Arabic yeh in "وضعيت"
            "چند سفارش در وضعيت لغو شده هستند؟",
            # no trailing question mark
            "چند سفارش در وضعیت لغو شده هستند",
            # ZWNJ (half-space) swapped for a plain space -- unlikely here
            # since this sentence has no compound word, kept for coverage
            # of the substitution rule itself.
            "چند سفارش در وضعیت لغو‌شده هستند؟",
        ],
    },
    {
        "id": "cache_simple_1",
        "category": "simple_lookup",
        "canonical": "لیست همه دسته‌بندی‌های محصولات چیست؟",
        "paraphrases": [
            # ZWNJ inside "دسته‌بندی‌های" replaced by a plain space
            "لیست همه دسته بندی های محصولات چیست؟",
            # Arabic kaf/yeh mixed in
            "ليست همه دسته‌بندي‌هاي محصولات چيست؟",
            "لیست همه دسته‌بندی‌های محصولات چیست",
        ],
    },
    {
        "id": "cache_join_1",
        "category": "join",
        "canonical": "نام محصولاتی که در دسته‌بندی لوازم خانگی هستند چیست؟",
        "paraphrases": [
            # doubled trailing whitespace
            "نام محصولاتی که در دسته‌بندی لوازم خانگی هستند چیست؟  ",
            # Arabic yeh/kaf mixed in
            "نام محصولاتي كه در دسته‌بندي لوازم خانگي هستند چيست؟",
            # missing final question mark
            "نام محصولاتی که در دسته‌بندی لوازم خانگی هستند چیست",
            # ZWNJ inside "دسته‌بندی" replaced by a plain space
            "نام محصولاتی که در دسته بندی لوازم خانگی هستند چیست؟",
        ],
    },
]