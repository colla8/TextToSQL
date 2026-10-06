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

    # Paths
    DATA_DIR: Path = BASE_DIR / "data"
    CSV_DATA: Path = DATA_DIR / "csv-data"
    META_DATA: Path = DATA_DIR / "metadata"
    DESCRIPTION: Path = META_DATA / "descriptions" / "descriptions.json"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"

    # Caching (SQL / retrieval / result caches -- see app/cache/query_cache.py)
    CACHE_ENABLED: bool = True
    CACHE_DB_PATH: Path = DATA_DIR / "state" / "cache.db"
    # SQL and retrieval only depend on the schema + the question text, not
    # on row contents, so they can safely live for a while.
    SQL_CACHE_TTL_SECONDS: int = 60 * 60 * 24 * 7  # 7 days
    RETRIEVAL_CACHE_TTL_SECONDS: int = 60 * 60 * 24 * 7  # 7 days
    # Result rows come straight from Postgres and go stale the moment the
    # underlying data changes, so this TTL is deliberately much shorter
    # (db_setup/load_data.py also invalidates this namespace explicitly
    # after a reload -- the short TTL is just a backstop).
    RESULT_CACHE_TTL_SECONDS: int = 60 * 5  # 5 minutes

    # PostgreSQL
    PG_HOST: str = "localhost"
    PG_PORT: str = "5432"
    PG_DATABASE: str = "online_store"
    PG_USER: str = "textsql"
    PG_PASSWORD: str = "123456"
    PG_POOL_MIN: int = 1
    PG_POOL_MAX: int = 10

    # API-side persistence (request log / per-user state, see api/database.py)
    # Kept in its own subfolder (not the top-level data/ dir, which also
    # holds the baked-in vector_db/csv-data/metadata) so a Docker volume can
    # persist just this file across deploys without shadowing those.
    APP_DB_URL: str = (
        f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'state' / 'app_state.db'}"
    )

    # Embedding
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_PATH: str = (
        "models/embeddings/bge-m3"  # missing -> downloaded from HuggingFace
    )
    EMBEDDING_BATCH_SIZE: int = 4
    # Explicit CPU: SentenceTransformer will grab the GPU by default if one
    # is visible, which on a 4GB card means it competes with qwen3 for the
    # same VRAM. The reranker was already pinned to CPU for this reason;
    # the embedder needs the same treatment so all VRAM stays free for the
    # LLM (this call is small/infrequent -- one question at a time -- so
    # CPU cost here is negligible next to what it buys the LLM).
    EMBEDDING_DEVICE: str = "cpu"

    # NUM_GPU: forces Ollama to offload exactly this many layers instead
    # of relying on its own auto-detection. -1 = don't send this option at
    # all, let Ollama decide (its own detection failed open before at only
    # 2% GPU, but forcing 999 crashed the server with CUDA OOM -- there's
    # no single safe forced number without knowing the model's exact layer
    # count and free VRAM). Find your real max by testing upward from a
    # small number: set NUM_GPU=6 in .env, restart, run one question,
    # check `ollama ps` didn't crash and the CPU/GPU split improved; if
    # it's stable, try 10, then 15, etc. Stop at the last value that
    # didn't error.
    NUM_GPU: int = -1
    # NUM_CTX: default Ollama context (4096, or 32k for some qwen3 tags)
    # reserves far more KV-cache VRAM than this prompt actually needs,
    # which is what was pushing layers onto CPU. Lower this further if
    # schema_context is usually short; raise it if repair prompts get
    # truncated.
    NUM_CTX: int = 2048
    # Keeps the model resident between requests so it isn't reloaded
    # (which costs real seconds) after every idle gap.
    OLLAMA_KEEP_ALIVE: str = "30m"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # 0 -> no timeout (waits indefinitely, see ollama_client.py). On this
    # hardware (mostly-CPU inference) generation can legitimately take
    # several minutes; a hard cutoff here just turns "slow" into "error"
    # for no benefit.
    OLLAMA_TIMEOUT_SECONDS: int = 0

    # Text splitter
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval
    TOP_K: int = 5
    # SchemaRetriever tries the lexical (BM25 + fuzzy) stage first --
    # stdlib-only, no model load, cheap enough to run on every request --
    # and only falls back to embedding + FAISS when the lexical stage's
    # top combined score doesn't clear this bar (see
    # app/embeddings/lexical_retriever.py's _bm25_normalize/_fuzzy_score
    # for how that score is built; combined = 0.6*bm25_norm + 0.4*fuzzy,
    # so it lives in [0, 1)). 0.35 was picked to admit a solid single-term
    # or two-term overlap (BM25 alone comfortably clears this once a
    # query token exactly matches a table/column name) while still
    # sending vague/loosely-worded or off-schema questions to the
    # embedding path, which handles paraphrase/synonym cases lexical
    # matching can't. Tune by running benchmarks/run_benchmark.py and
    # checking the retrieval_method split in the report against how many
    # of those lexical hits were actually correct.
    LEXICAL_CONFIDENCE_THRESHOLD: float = 0.35
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MODEL_PATH: str = (
        "models/reranker/bge-reranker-v2-m3"
    )
    RERANK_CANDIDATES: int = 15
    RERANKER_DEVICE: str = "cpu"

    # Generation
    TEMPERATURE: float = 0.0
    # Was 1563: with think=True (see sql_generator.py), the model's full
    # chain-of-thought (schema linking -> join path -> filters -> fan-out
    # check -> projection -> optimization) runs inside this budget before
    # it ever writes the SQL, and on CPU-heavy inference that's a large
    # share of total latency. This dataset's queries don't need anywhere
    # near 1563 tokens of output; raise it back if you see truncated/cut-
    # off SQL in practice.
    MAX_TOKENS: int = 1563

    # Telegram bot
    TOKEN: str = Field(default="")
    FASTAPI_URL: str = "http://127.0.0.1:8000"

    # API server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "*"  # comma-separated list, or "*"
    LOG_LEVEL: str = "INFO"
    MAX_QUESTION_LENGTH: int = 500

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


setting = Settings()
