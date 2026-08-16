from pathlib import Path
from dataclasses import dataclass


@dataclass
class Setting:

    PROJECT_NAME: str = "Text to SQL"

    # 📂 Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "data"
    CSV_DATA: Path = DATA_DIR / "cdv-data"
    META_DATA: Path = DATA_DIR / "metadata"
    DESCRIPTION: Path = META_DATA / "descriptions" / "descriptions.json"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"

    # 🤖 Embedding
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    # Local offline model path
    EMBEDDING_MODEL_PATH: str = (
        "/home/zeinab/Projects/ai/bank-rag/models/embeddings/bge-m3"
    )
    EMBEDDING_BATCH_SIZE: int = 4

    # 🧠 LLM
    OLLAMA_MODEL = "qwen3:8b"

    # ✂️ Text Splitter
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # 🔍 Retrieval
    TOP_K: int = 8

    # 🔁 Reranking
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MODEL_PATH: str = (
        "/home/zeinab/Projects/ai/bank-rag/models/reranker/bge-reranker-v2-m3"
    )
    RERANK_CANDIDATES: int = 20  
    RERANK_TOP_K: int = 5  
    RERANKER_DEVICE: str = "cpu"

    # 😃 Generation
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1563

    # 🦙 LLM
    OLLAMA_BASE_URL = "http://localhost:11434"


setting = Setting()
