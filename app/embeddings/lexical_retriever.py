# -*- coding: utf-8 -*-
"""
Lexical (BM25 + fuzzy) schema retrieval -- the *primary* path for schema
linking. app/embeddings/retriever.py runs this first, on every request;
only when it can't find a confident match does it fall back to the
embedding + FAISS stage.

Deliberately dependency-free (stdlib only: math, re, difflib, collections)
so this stage is cheap compared to the embedding path it's gating -- no
model load, no GPU/CPU inference, no network.

Reuses app.embeddings.schema_documents.build_documents() so the lexical
corpus and the embedding corpus are built from the exact same table/column
metadata and text -- the two stages are answering the same question, just
with different scoring, so results should be directly interchangeable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.embeddings.schema_documents import build_documents, load_table_metadata

# Normalize the handful of Persian letters/marks that commonly vary between
# how a question is typed and how metadata was written (ي/ی, ك/ک, ZWNJ,
# diacritics) so tokens actually match instead of silently missing due to
# encoding variants.
_PERSIAN_NORMALIZE = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "‌": " ",  # ZWNJ -> space, so a half-joined word still splits into tokens
    }
)
_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # harakat + tatweel
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

# High-frequency Persian function words. Schema metadata text (table/column
# names and descriptions) barely contains any of these, so they add almost
# nothing to BM25 -- but left unfiltered they poison the *fuzzy* stage: a
# short stopword coincidentally has a high SequenceMatcher ratio against
# some unrelated short schema token, which used to be enough to (wrongly)
# inflate confidence on off-topic questions. Filtered out of the query only
# -- document tokens are left untouched.
_PERSIAN_STOPWORDS = {
    "و",
    "در",
    "به",
    "از",
    "که",
    "این",
    "آن",
    "را",
    "با",
    "برای",
    "تا",
    "هم",
    "یا",
    "اگر",
    "چون",
    "چه",
    "چند",
    "چقدر",
    "کدام",
    "کجا",
    "چطور",
    "چگونه",
    "آیا",
    "بده",
    "بدهید",
    "نشان",
    "لیست",
    "کن",
    "کنید",
    "است",
    "هست",
    "هستند",
    "بود",
    "بودند",
    "دارد",
    "دارند",
    "می",
    "شده",
    "شود",
    "شوند",
    "یک",
    "دو",
    "سه",
    "هر",
    "همه",
    "هیچ",
    "بر",
    "روی",
    "لطفا",
    "لطفاً",
}
# Below this ratio, a fuzzy match is treated as coincidental character
# overlap rather than a real spelling/inflection variant, and contributes
# nothing (see _fuzzy_score). Tokens shorter than this are skipped
# entirely on both sides of the comparison -- short strings match almost
# anything by chance.
_FUZZY_MIN_RATIO = 0.72
_FUZZY_MIN_TOKEN_LEN = 3


def normalize_text(text: str) -> str:
    text = text.translate(_PERSIAN_NORMALIZE)
    text = _DIACRITICS_RE.sub("", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def tokenize_query(text: str) -> list[str]:
    """Like tokenize(), plus stopword filtering -- only used on the
    incoming question, never on the indexed schema documents."""
    return [t for t in tokenize(text) if t not in _PERSIAN_STOPWORDS]


class _BM25:
    """Minimal Okapi BM25 over an in-memory corpus of tokenized documents.
    Pure stdlib on purpose -- see module docstring."""

    def __init__(
        self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75
    ):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = [Counter(toks) for toks in corpus_tokens]
        self.doc_lens = [len(toks) for toks in corpus_tokens]
        n_docs = len(corpus_tokens)
        self.avgdl = (sum(self.doc_lens) / n_docs) if n_docs else 0.0

        df: Counter = Counter()
        for tf in self.doc_freqs:
            for term in tf:
                df[term] += 1
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        if not query_tokens or not self.doc_freqs:
            return [0.0] * len(self.doc_freqs)
        out = []
        for tf, dl in zip(self.doc_freqs, self.doc_lens):
            s = 0.0
            denom_const = (
                self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                if self.avgdl
                else self.k1
            )
            for term in query_tokens:
                f = tf.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + denom_const)
            out.append(s)
        return out


def _fuzzy_score(query_tokens: list[str], doc_token_set: set[str]) -> float:
    """Average, over query tokens, of the best fuzzy ratio against any
    token in the document -- 0 for a token whose best match doesn't clear
    _FUZZY_MIN_RATIO. Catches typos and Persian inflection/plural suffixes
    (e.g. "مشتریانی" vs "مشتری") that BM25's exact-token matching misses,
    without letting short, coincidentally-similar words inflate the score
    (see _FUZZY_MIN_RATIO / _FUZZY_MIN_TOKEN_LEN). Denominator is always
    len(query_tokens), so a single lucky match among many unrelated tokens
    still yields a low overall score, not a high one.
    Exact matches short-circuit to 1.0."""
    tokens = [t for t in query_tokens if len(t) >= _FUZZY_MIN_TOKEN_LEN]
    if not tokens:
        return 0.0
    candidates = [dt for dt in doc_token_set if len(dt) >= _FUZZY_MIN_TOKEN_LEN]
    if not candidates:
        return 0.0
    total = 0.0
    for qt in tokens:
        if qt in doc_token_set:
            total += 1.0
            continue
        best = max(
            (SequenceMatcher(None, qt, dt).ratio() for dt in candidates), default=0.0
        )
        if best >= _FUZZY_MIN_RATIO:
            total += best
    return total / len(tokens)


def _bm25_normalize(raw: float, saturation: float = 3.0) -> float:
    """Squash an unbounded BM25 score into (0, 1) via score / (score + K).
    Deliberately NOT a relative (min-max over this query's results)
    normalization -- a genuinely weak match should stay low confidence
    regardless of how it ranks against other, equally weak, candidates."""
    if raw <= 0:
        return 0.0
    return raw / (raw + saturation)


class LexicalSchemaRetriever:
    """BM25 + fuzzy matching over table/column schema metadata.

    Built once (cheap -- just tokenizing the metadata JSON already on
    disk, no IO beyond that) and reused across requests, mirroring how
    SchemaRetriever holds one FaissVectorStore/Embedder for its lifetime.
    """

    def __init__(
        self,
        meta_dir: Path,
        bm25_weight: float = 0.6,
        fuzzy_weight: float = 0.4,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.meta_dir = meta_dir
        self.bm25_weight = bm25_weight
        self.fuzzy_weight = fuzzy_weight

        tables = load_table_metadata(meta_dir)
        self.documents: list[dict[str, Any]] = build_documents(tables)
        self.doc_tokens: list[list[str]] = [
            tokenize(doc["text"]) for doc in self.documents
        ]
        self.doc_token_sets: list[set[str]] = [set(toks) for toks in self.doc_tokens]
        self.bm25 = _BM25(self.doc_tokens, k1=bm25_k1, b=bm25_b)

    def _score_documents(self, question: str) -> list[dict[str, Any]]:
        query_tokens = tokenize_query(question)
        bm25_raw = self.bm25.scores(query_tokens)
        results = []
        for i, doc in enumerate(self.documents):
            bm25_norm = _bm25_normalize(bm25_raw[i])
            fuzzy = _fuzzy_score(query_tokens, self.doc_token_sets[i])
            combined = self.bm25_weight * bm25_norm + self.fuzzy_weight * fuzzy
            if combined <= 0:
                continue
            entry = dict(doc)
            entry["score"] = combined
            entry["bm25_score"] = bm25_norm
            entry["fuzzy_score"] = fuzzy
            results.append(entry)
        results.sort(key=lambda d: -d["score"])
        return results

    def search(
        self, question: str, top_k: int, level: str | None = None
    ) -> list[dict[str, Any]]:
        """Same shape/contract as FaissVectorStore.search: documents with
        an added "score" field, best first, optionally filtered by
        level ("table" | "column")."""
        scored = self._score_documents(question)
        if level is not None:
            scored = [d for d in scored if d["level"] == level]
        return scored[:top_k]

    def relevant_tables(
        self, question: str, top_k_columns: int, top_k_tables: int
    ) -> tuple[list[str], float, dict[str, Any]]:
        """Mirrors SchemaRetriever's embedding-path merge logic (column
        hits + table hits, ranked together, deduplicated by table, best
        first) so the two stages are interchangeable from the caller's
        point of view.

        Returns (tables, confidence, debug):
          tables     -- table names, ranked best first, deduplicated
          confidence -- the top-ranked hit's combined score; the caller
                        uses this to decide whether to trust this stage
                        or fall back to embeddings
          debug      -- top hits, for logging/benchmarking
        """
        col_hits = self.search(question, top_k=top_k_columns, level="column")
        table_hits = self.search(question, top_k=top_k_tables, level="table")
        ranked = sorted(col_hits + table_hits, key=lambda h: -h["score"])

        seen: list[str] = []
        for hit in ranked:
            if hit["table"] not in seen:
                seen.append(hit["table"])

        confidence = ranked[0]["score"] if ranked else 0.0
        debug = {
            "top_hits": [
                {"id": h["id"], "table": h["table"], "score": round(h["score"], 4)}
                for h in ranked[:5]
            ],
        }
        return seen, confidence, debug
