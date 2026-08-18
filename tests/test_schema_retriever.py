# -*- coding: utf-8 -*-
"""
Tests for the two-stage decision logic in app/embeddings/retriever.py:

    question -> BM25 + fuzzy matching
                  |-- confidence high enough -> use it, embedding never runs
                  `-- confidence too low     -> fall back to embedding + FAISS

Runs against this project's real metadata (data/metadata/) via a real
LexicalSchemaRetriever (it's stdlib-only, see test_lexical_retriever.py),
but the embedding side is swapped for lightweight fakes injected through
SchemaRetriever's existing `embedder=`/`store=` constructor parameters --
so these tests need neither a downloaded sentence-transformer model nor a
real FAISS index file, and can assert exactly how many times the fake
embedder was actually invoked.

Requires the project's real dependencies to be importable (faiss,
sentence-transformers, pydantic-settings) since app/embeddings/retriever.py
imports them at module level even though the fakes below mean their
heavy-weight code paths are never *executed* -- same constraint the
project already had before this change.

Run with:
    python -m pytest tests/test_schema_retriever.py -v
or:
    python -m unittest tests.test_schema_retriever -v
"""

from __future__ import annotations

import unittest

import numpy as np

from app.embeddings.retriever import SchemaRetriever


class FakeEmbedder:
    """Stands in for app.embeddings.embedder.Embedder: no model load, just
    counts calls so tests can assert the embedding path did/didn't run."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.zeros((len(texts), 4), dtype="float32")


class FakeStore:
    """Stands in for app.embeddings.vector_store.FaissVectorStore: returns
    a fixed, known hit instead of doing a real similarity search, so the
    fallback path's output is deterministic and doesn't depend on a real
    FAISS index file being present."""

    def __init__(self, table: str = "customers", score: float = 0.9):
        self.table = table
        self.score = score
        self.calls = 0

    def search(self, query_vector, top_k: int, level: str | None = None):
        self.calls += 1
        if level not in (None, "table"):
            return []
        return [
            {
                "id": f"table::{self.table}",
                "level": "table",
                "table": self.table,
                "column": None,
                "score": self.score,
            }
        ][:top_k]


def make_retriever(**overrides) -> tuple[SchemaRetriever, FakeEmbedder, FakeStore]:
    embedder = overrides.pop("embedder", FakeEmbedder())
    store = overrides.pop("store", FakeStore())
    retriever = SchemaRetriever(embedder=embedder, store=store, **overrides)
    return retriever, embedder, store


class TestSchemaRetrieverTwoStage(unittest.TestCase):
    def test_success_path_uses_bm25_fuzzy_and_never_calls_embedding(self):
        """A clear, on-topic question should be resolved entirely by the
        lexical stage -- the embedding fallback must not run at all."""
        retriever, embedder, store = make_retriever()

        tables = retriever.relevant_tables("قیمت محصولات چقدر است؟")

        self.assertEqual(tables[0], "products")
        self.assertEqual(retriever.last_retrieval_meta["method"], "bm25_fuzzy")
        self.assertGreaterEqual(
            retriever.last_retrieval_meta["confidence"], retriever.confidence_threshold
        )
        self.assertEqual(
            embedder.calls, 0, "embedding must not run when BM25/fuzzy is confident"
        )
        self.assertEqual(
            store.calls, 0, "FAISS search must not run when BM25/fuzzy is confident"
        )

    def test_fallback_path_used_when_lexical_confidence_is_low(self):
        """A question whose key term ("Tehran") never appears in the
        metadata should leave the lexical stage under-confident, and
        SchemaRetriever should fall back to the (fake) embedding stage --
        which here deterministically returns "customers"."""
        retriever, embedder, store = make_retriever(
            store=FakeStore(table="customers", score=0.9)
        )

        tables = retriever.relevant_tables("مشتریانی که در تهران هستند را نشان بده")

        self.assertEqual(tables, ["customers"])
        self.assertEqual(retriever.last_retrieval_meta["method"], "embedding_fallback")
        self.assertEqual(
            embedder.calls, 1, "embedding fallback should run exactly once"
        )
        self.assertEqual(
            store.calls, 2, "FAISS search runs for both column and table level"
        )
        # the lexical confidence that triggered the fallback is still
        # recorded, for benchmarking/debugging -- it should genuinely be
        # below threshold, not None or missing.
        self.assertIsNotNone(retriever.last_retrieval_meta["confidence"])
        self.assertLess(
            retriever.last_retrieval_meta["confidence"], retriever.confidence_threshold
        )

    def test_confidence_threshold_is_configurable(self):
        """Setting an unreachably high threshold should force every
        question to the embedding fallback, even a clear-cut one."""
        retriever, embedder, _ = make_retriever(confidence_threshold=1.01)

        retriever.relevant_tables("قیمت محصولات چقدر است؟")

        self.assertEqual(retriever.last_retrieval_meta["method"], "embedding_fallback")
        self.assertEqual(embedder.calls, 1)

    def test_disabling_lexical_stage_forces_legacy_embedding_only_behavior(self):
        """ENABLE_LEXICAL_SCHEMA_RETRIEVAL=False (or enable_lexical=False)
        should reproduce the pre-change behavior: embedding runs on every
        request, unconditionally."""
        retriever, embedder, _ = make_retriever(enable_lexical=False)

        retriever.relevant_tables("قیمت محصولات چقدر است؟")

        self.assertEqual(retriever.last_retrieval_meta["method"], "embedding_fallback")
        self.assertIsNone(retriever.last_retrieval_meta["confidence"])
        self.assertEqual(embedder.calls, 1)

    def test_get_relevant_schema_api_unchanged(self):
        """Public API contract (schema_context: str, allowed_schema: dict)
        must be unaffected by which stage answered the request."""
        retriever, _, _ = make_retriever()

        schema_context, allowed_schema = retriever.get_relevant_schema(
            "قیمت محصولات چقدر است؟"
        )

        self.assertIsInstance(schema_context, str)
        self.assertIsInstance(allowed_schema, dict)
        self.assertIn("products", allowed_schema)
        self.assertIn("product_name", allowed_schema["products"])


if __name__ == "__main__":
    unittest.main()
