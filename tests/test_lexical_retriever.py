# -*- coding: utf-8 -*-
"""
Tests for app/embeddings/lexical_retriever.py -- the BM25 + fuzzy stage on
its own. Deliberately dependency-free (no faiss, no sentence-transformers,
no pydantic): LexicalSchemaRetriever only touches stdlib + the plain-JSON
metadata loader in schema_documents.py, so these tests run anywhere,
instantly, with no model/index to load.

Run with:
    python -m pytest tests/test_lexical_retriever.py -v
or:
    python -m unittest tests.test_lexical_retriever -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

from app.embeddings.lexical_retriever import LexicalSchemaRetriever

META_DIR = Path(__file__).resolve().parents[1] / "data" / "metadata"


class TestLexicalSchemaRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexical = LexicalSchemaRetriever(META_DIR)

    def test_clear_on_topic_question_ranks_expected_table_first(self):
        tables, confidence, debug = self.lexical.relevant_tables(
            "قیمت محصولات چقدر است؟", top_k_columns=5, top_k_tables=3
        )
        self.assertTrue(tables, "expected at least one table hit")
        self.assertEqual(tables[0], "products")
        self.assertGreater(confidence, 0.0)
        self.assertIn("top_hits", debug)

    def test_reviews_question_ranks_reviews_first(self):
        tables, confidence, _ = self.lexical.relevant_tables(
            "نظرات و امتیاز محصولات را نشان بده", top_k_columns=5, top_k_tables=3
        )
        self.assertEqual(tables[0], "reviews")

    def test_inflected_word_still_matches_via_fuzzy(self):
        """"مشتریانی" (customers + suffix) doesn't exact-match "مشتری" in
        the metadata, so BM25 alone would miss it -- fuzzy matching should
        still surface the customers table first."""
        tables, _, _ = self.lexical.relevant_tables(
            "مشتریانی که ثبت‌نام کرده‌اند را نشان بده", top_k_columns=5, top_k_tables=3
        )
        self.assertEqual(tables[0], "customers")

    def test_offtopic_question_has_low_confidence(self):
        """A question with no real connection to the schema should score
        low enough to trip the fallback in SchemaRetriever (well under the
        default 0.55 threshold), not confidently claim an unrelated table."""
        _, confidence, _ = self.lexical.relevant_tables(
            "یک جمله نامفهوم که به هیچ جدولی مربوط نیست بلبل قناری",
            top_k_columns=5,
            top_k_tables=3,
        )
        self.assertLess(confidence, 0.55)

    def test_vocabulary_missing_from_schema_has_low_confidence(self):
        """"تهران" (a city name) never appears in the metadata -- only the
        generic column description "شهر" does -- so lexical matching alone
        should stay unconfident here; this is exactly the case
        SchemaRetriever should hand off to the embedding fallback."""
        _, confidence, _ = self.lexical.relevant_tables(
            "مشتریانی که در تهران هستند را نشان بده", top_k_columns=5, top_k_tables=3
        )
        self.assertLess(confidence, 0.55)

    def test_search_respects_level_filter(self):
        column_hits = self.lexical.search("قیمت محصولات", top_k=5, level="column")
        self.assertTrue(all(h["level"] == "column" for h in column_hits))

    def test_empty_question_yields_no_hits(self):
        tables, confidence, _ = self.lexical.relevant_tables("", top_k_columns=5, top_k_tables=3)
        self.assertEqual(tables, [])
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
