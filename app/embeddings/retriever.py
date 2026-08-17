import json
from pathlib import Path
from typing import Any

from app.core.config import setting
from app.embeddings.embedder import Embedder
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
    ):
        self.store = FaissVectorStore.load(vector_db_dir or setting.VECTOR_DB_DIR)
        self.embedder = embedder or Embedder()
        self.meta_dir = meta_dir or setting.META_DATA
        self.top_k_columns = top_k_columns or setting.TOP_K
        self.top_k_tables = top_k_tables

    def _load_table(self, table_name: str) -> dict[str, Any]:
        with open(self.meta_dir / f"{table_name}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def relevant_tables(self, question: str) -> list[str]:
        """Returns table names ranked by best matching score (column or
        table level hit), deduplicated, best first."""
        query_vector = self.embedder.encode([question])[0]
        col_hits = self.store.search(
            query_vector, top_k=self.top_k_columns, level="column"
        )
        table_hits = self.store.search(
            query_vector, top_k=self.top_k_tables, level="table"
        )

        ranked = sorted(col_hits + table_hits, key=lambda h: -h["score"])
        seen: list[str] = []
        for hit in ranked:
            if hit["table"] not in seen:
                seen.append(hit["table"])
        return seen

    def get_schema_context(self, question: str) -> str:
        return self.get_relevant_schema(question)[0]

    def get_relevant_schema(self, question: str) -> tuple[str, dict[str, list[str]]]:
        """Returns (schema_context_text, allowed_schema) where allowed_schema
        is {table_name: [column_name, ...]} -- the ground truth used by
        sql_validator.py to catch hallucinated identifiers."""
        tables = self.relevant_tables(question)
        table_metas = [self._load_table(t) for t in tables]

        blocks = [table_schema_text(meta) for meta in table_metas]
        schema_context = "\n\n".join(blocks)

        allowed_schema = {
            meta["table"]: [col["name"] for col in meta["columns"]]
            for meta in table_metas
        }
        return schema_context, allowed_schema
