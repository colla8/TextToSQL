# -*- coding: utf-8 -*-
"""
AnswerRouter -- decides whether a question needs LLM reasoning/analysis, or
is a plain lookup whose SQL result can be shown directly.

NOTE: not wired into app/pipeline.py yet; it is a standalone, tested helper.
"""

REASONING_KEYWORDS: tuple[str, ...] = (
    "چرا",
    "چگونه",
    "تحلیل",
    "توضیح",
    "مقایسه",
    "بهتر",
    "نتیجه",
    "پیشنهاد",
    "روند",
    "عملکرد",
    "پیش‌بینی",
    "تجزیه و تحلیل",
    "ارزیابی",
    "گزار",
)


class AnswerRouter:
    def needLLMs(self, question: str) -> bool:
        """True if the question asks for reasoning/analysis (any reasoning
        keyword appears in it), False for a direct data lookup."""
        return any(keyword in question for keyword in REASONING_KEYWORDS)
