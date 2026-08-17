import sys
import time
from pathlib import Path

# make `app` importable when running as `python scripts/build_embeddings.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import setting
from app.embeddings.schema_documents import load_table_metadata, build_documents
from app.embeddings.embedder import Embedder
from app.embeddings.vector_store import FaissVectorStore


def main():
    print(f"reading table metadata from {setting.META_DATA}")
    tables = load_table_metadata(setting.META_DATA)
    if not tables:
        sys.exit(
            f"no table metadata found in {setting.META_DATA}. Run scripts/build_metadata.py first."
        )
    print(f"loaded {len(tables)} table(s).")

    documents = build_documents(tables)
    n_table = sum(1 for d in documents if d["level"] == "table")
    n_column = sum(1 for d in documents if d["level"] == "column")
    print(
        f"built {len(documents)} document(s) -> {n_table} table-level, {n_column} column-level."
    )

    print(f"loading embedding model from {setting.EMBEDDING_MODEL_PATH} ...")
    t0 = time.time()
    embedder = Embedder()
    print(f"model loaded in {time.time() - t0:.1f}s.")

    texts = [d["text"] for d in documents]
    print(
        f"embedding {len(texts)} document(s), batch_size={setting.EMBEDDING_BATCH_SIZE} ..."
    )
    t0 = time.time()
    vectors = embedder.encode(texts)
    print(f"embedded in {time.time() - t0:.1f}s. vector shape: {vectors.shape}")

    store = FaissVectorStore.build(vectors, documents)
    store.save(setting.VECTOR_DB_DIR)
    print(f"saved FAISS index + docstore to {setting.VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
