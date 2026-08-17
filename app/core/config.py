# -*- coding: utf-8 -*-
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Text to SQL"
    ENV: str = "development"  # development | production

    # Paths 📁
    DATA_DIR: Path = BASE_DIR / "data"
    CSV_DATA: Path = DATA_DIR / "csv-data"
    META_DATA: Path = DATA_DIR / "metadata"
    DESCRIPTION: Path = META_DATA / "descriptions" / "descriptions.json"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"

    # PostgreSQL 🐘
    PG_HOST: str = "localhost"
    PG_PORT: str = "5432"
    PG_DATABASE: str = "online_store"
    PG_USER: str = "textsql"
    PG_PASSWORD: str = "123456"
    PG_POOL_MIN: int = 1
    PG_POOL_MAX: int = 10

    # API-side persistence 💾
    APP_DB_URL: str = (
        f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'state' / 'app_state.db'}"
    )

    # Embedding 🧠
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_PATH: str = (
        "/home/zeinab/Projects/ai/bank-rag/models/embeddings/bge-m3"
    )
    EMBEDDING_BATCH_SIZE: int = 4
    EMBEDDING_DEVICE: str = "cpu"

    # Ollama / LLM 🤖
    NUM_GPU: int = -1
    NUM_CTX: int = 2048
    OLLAMA_KEEP_ALIVE: str = "30m"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT_SECONDS: int = 0

    # Text splitter ✂️
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval 🔎
    TOP_K: int = 5
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MODEL_PATH: str = (
        "/home/zeinab/Projects/ai/bank-rag/models/reranker/bge-reranker-v2-m3"
    )
    RERANKER_CANDIDATES: int = 15
    RERANKER_DEVICE: str = "cpu"

    # Generation ✨
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 800

    # Telegram bot 📱
    TOKEN: str = Field(default="")
    FASTAPI_URL: str = "http://127.0.0.1:8000"

    # API server 🌐
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"
    MAX_QUESTION_LENGTH: int = 500

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


setting = Settings()
