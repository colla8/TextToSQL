# -*- coding: utf-8 -*-
"""Run with: python -m unittest tests.test_answer_router -v"""

import unittest

from app.routing.answer_router import AnswerRouter


class TestAnswerRouter(unittest.TestCase):
    def setUp(self):
        self.router = AnswerRouter()

    def test_returns_a_bool(self):
        self.assertIsInstance(self.router.needLLMs("قیمت محصولات چقدر است؟"), bool)

    def test_plain_lookup_does_not_need_llm(self):
        self.assertFalse(self.router.needLLMs("قیمت محصولات چقدر است؟"))
        self.assertFalse(self.router.needLLMs("تعداد مشتریان تهران را نشان بده"))

    def test_reasoning_questions_need_llm(self):
        self.assertTrue(self.router.needLLMs("چرا فروش کاهش یافته است؟"))
        self.assertTrue(self.router.needLLMs("روند فروش ماهانه را تحلیل کن"))
        self.assertTrue(self.router.needLLMs("گزارش عملکرد محصولات"))

    def test_empty_question(self):
        self.assertFalse(self.router.needLLMs(""))


if __name__ == "__main__":
    unittest.main()
