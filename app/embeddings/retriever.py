# -*- coding: utf-8 -*-
"""
Two-stage schema linking: cheap lexical (BM25 + fuzzy) matching first,
embedding + FAISS only as a fallback.

The lexical stage (app/embeddings/lexical_retriever.py) is stdlib-only --
no model load, no GPU/CPU inference, no network -- so on the common case
(a question whose wording overlaps enough with the schema's Persian
table/column names and descriptions) it resolves retrieval entirely on
its own, without ever touching the embedding model or the FAISS index.
Only when it isn't confident (see LEXICAL_CONFIDENCE_THRESHOLD in
app/core/config.py) does this fall back to the slower embedding path,
which is better at paraphrase/synonym cases the lexical stage can't
match on tokens alone.

The embedding backend (Embedder + FaissVectorStore) is built lazily, on
first actual fallback -- not in __init__ -- so a process whose questions
are all resolved lexically never pays for loading the embedding model or
reading the FAISS index at all.
"""

import json
from pathlib import Path
from typing import Any

from app.core.config import setting
from app.embeddings.embedder import Embedder
from app.embeddings.lexical_retriever import LexicalSchemaRetriever
from app.embeddings.schema_documents import table_schema_text
from app.embeddings.vector_store import FaissVectorStore


class SchemaRetriever:
    def __init__(
        self,
        vector_db_dir: Path | None = None,
        meta_dir: Path | None = None,
        top_k_columns: int | None = None,
        top_k_tables: int = 3,
        embedder: Embedder | None = None,
        store: FaissVectorStore | None = None,
        lexical_confidence_threshold: float | None = None,
    ):
        self.vector_db_dir = vector_db_dir or setting.VECTOR_DB_DIR
        self.meta_dir = meta_dir or setting.META_DATA
        self.top_k_columns = top_k_columns or setting.TOP_K
        self.top_k_tables = top_k_tables
        self.lexical_confidence_threshold = (
            setting.LEXICAL_CONFIDENCE_THRESHOLD
            if lexical_confidence_threshold is None
            else lexical_confidence_threshold
        )

        # Cheap (stdlib tokenizing of metadata already on disk) -- built
        # eagerly, every instantiation. See LexicalSchemaRetriever's own
        # docstring.
        self.lexical = LexicalSchemaRetriever(self.meta_dir)

        # Expensive (SentenceTransformer model load + FAISS index read) --
        # NOT built here. Only materialized in _ensure_embedding_backend,
        # the first time relevant_tables() actually needs to fall back.
        # If the caller passed an already-constructed embedder and/or store,
        # honor them (skips the lazy load -- e.g. tests / shared instances);
        # whatever wasn't injected is still built lazily on first fallback.
        self._embedder = embedder
        self._store = store

        # Debug info from the most recent relevant_tables() /
        # get_relevant_schema() call. Deliberately NOT part of either
        # method's return value -- get_relevant_schema()'s
        # (schema_context, allowed_schema) shape has to stay exactly what
        # app/pipeline.py and app/cache/query_cache.py's retrieval cache
        # already expect. Callers that want to know which stage answered
        # a given question (e.g. benchmarks/instrumented_pipeline.py)
        # read these attributes right after calling.
        self.last_retrieval_method: str | None = None  # "lexical" | "embedding_fallback"
        self.last_retrieval_confidence: float | None = None  # lexical stage's top score
        self.last_retrieval_debug: dict[str, Any] | None = None

    def _ensure_embedding_backend(self) -> None:
        if self._store is None:
            self._store = FaissVectorStore.load(self.vector_db_dir)
        if self._embedder is None:
            self._embedder = Embedder()

    def _load_table(self, table_name: str) -> dict[str, Any]:
        with open(self.meta_dir / f"{table_name}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _relevant_tables_embedding(self, question: str) -> list[str]:
        self._ensure_embedding_backend()
        query_vector = self._embedder.encode([question])[0]
        col_hits = self._store.search(
            query_vector, top_k=self.top_k_columns, level="column"
        )
        table_hits = self._store.search(
            query_vector, top_k=self.top_k_tables, level="table"
        )

        ranked = sorted(col_hits + table_hits, key=lambda h: -h["score"])
        seen: list[str] = []
        for hit in ranked:
            if hit["table"] not in seen:
                seen.append(hit["table"])

        self.last_retrieval_debug = {
            "embedding_top_hits": [
                {"id": h["id"], "table": h["table"], "score": round(h["score"], 4)}
                for h in ranked[:5]
            ],
        }
        return seen

    def relevant_tables(self, question: str) -> list[str]:
        """Returns table names ranked by best matching score (column or
        table level hit), deduplicated, best first.

        Tries the lexical stage first. Only falls back to embedding +
        FAISS when the lexical stage came up empty or wasn't confident
        (top combined score below lexical_confidence_threshold)."""
        tables, confidence, lexical_debug = self.lexical.relevant_tables(
            question,
            top_k_columns=self.top_k_columns,
            top_k_tables=self.top_k_tables,
        )

        if tables and confidence >= self.lexical_confidence_threshold:
            self.last_retrieval_method = "lexical"
            self.last_retrieval_confidence = confidence
            self.last_retrieval_debug = lexical_debug
            return tables

        # Lexical stage didn't find a confident match -- fall back to the
        # slower embedding + FAISS path. Confidence is still recorded
        # (even at 0.0) so it's visible how far under threshold this was.
        fallback_tables = self._relevant_tables_embedding(question)
        self.last_retrieval_method = "embedding_fallback"
        self.last_retrieval_confidence = confidence
        self.last_retrieval_debug = {
            "lexical_confidence": confidence,
            "lexical_top_hits": lexical_debug.get("top_hits", []),
            **(self.last_retrieval_debug or {}),
        }
        return fallback_tables

    def get_schema_context(self, question: str) -> str:
        return self.get_relevant_schema(question)[0]

    def get_relevant_schema(self, question: str) -> tuple[str, dict[str, list[str]]]:
        """Returns (schema_context_text, allowed_schema) where allowed_schema
        is {table_name: [column_name, ...]} -- the ground truth used by
        sql_validator.py to catch hallucinated identifiers.

        Return shape is unchanged from the embedding-only version, so this
        is a drop-in replacement wherever app/pipeline.py and the
        retrieval cache in app/cache/query_cache.py use it -- neither
        knows or cares which stage (lexical vs. embedding) actually
        produced the tables. Which stage ran is available afterward via
        self.last_retrieval_method / self.last_retrieval_confidence for
        callers that want it (e.g. benchmarks/instrumented_pipeline.py)."""
        tables = self.relevant_tables(question)
        table_metas = [self._load_table(t) for t in tables]

        blocks = [table_schema_text(meta) for meta in table_metas]
        schema_context = "\n\n".join(blocks)

        allowed_schema = {
            meta["table"]: [col["name"] for col in meta["columns"]]
            for meta in table_metas
        }
        return schema_context, allowed_schema
