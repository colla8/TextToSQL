import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

INDEX_FILENAME = "schema_index.faiss"
DOCS_FILENAME = "schema_docs.json"


class FaissVectorStore:
    def __init__(self, index: faiss.Index, documents: list[dict[str, Any]]):
        self.index = index
        self.documents = documents

    @classmethod
    def build(
        cls, vectors: np.ndarray, documents: list[dict[str, Any]]
    ) -> "FaissVectorStore":
        if vectors.shape[0] != len(documents):
            raise ValueError("number of vectors must match number of documents")
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return cls(index, documents)

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / INDEX_FILENAME))
        with open(dir_path / DOCS_FILENAME, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: Path) -> "FaissVectorStore":
        index = faiss.read_index(str(dir_path / INDEX_FILENAME))
        with open(dir_path / DOCS_FILENAME, "r", encoding="utf-8") as f:
            documents = json.load(f)
        return cls(index, documents)

    def search(
        self, query_vector: np.ndarray, top_k: int, level: str | None = None
    ) -> list[dict[str, Any]]:
        """query_vector: shape (dim,) or (1, dim), already normalized.
        level: optionally restrict results to "table" or "column".
        Returns documents with an added "score" field, best first."""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # over-fetch when filtering by level so top_k is still met after filtering
        k = top_k if level is None else min(len(self.documents), top_k * 5)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = dict(self.documents[idx])
            if level is not None and doc["level"] != level:
                continue
            doc["score"] = float(score)
            results.append(doc)
            if len(results) >= top_k:
                break
        return results
