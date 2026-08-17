from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import setting


class Embedder:
    def __init__(
        self,
        model_path: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ):
        self.batch_size = batch_size or setting.EMBEDDING_BATCH_SIZE
        resolved = model_path or setting.EMBEDDING_MODEL_PATH
        if not Path(resolved).exists():
            print(
                f"warning: local model path '{resolved}' not found, "
                f"falling back to downloading '{setting.EMBEDDING_MODEL_NAME}' from HuggingFace."
            )
            resolved = setting.EMBEDDING_MODEL_NAME
        # Explicit device: left unset, SentenceTransformer grabs the GPU
        # automatically when one is visible, competing with qwen3 for the
        # same 4GB of VRAM. Pinning to CPU (setting.EMBEDDING_DEVICE)
        # keeps that VRAM free for the LLM, same reasoning as the
        # reranker's RERANKER_DEVICE="cpu".
        self.model = SentenceTransformer(
            resolved, device=device or setting.EMBEDDING_DEVICE
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns an (N, dim) float32 array, L2-normalized (so inner
        product == cosine similarity -- required for the FAISS index
        used in vector_store.py)."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")
