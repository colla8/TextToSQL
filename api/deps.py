# -*- coding: utf-8 -*-
"""
Holds the single, process-wide TextToSQLPipeline instance.

Building a TextToSQLPipeline loads the embedding model and the FAISS index
into memory - that's expensive (seconds, real RAM) and must NOT happen on
every request. It's created once in main.py's lifespan handler and stashed
here so request handlers can grab the same instance via get_pipeline().
"""

import logging

from app.pipeline import TextToSQLPipeline

logger = logging.getLogger("text2sql.api")

_pipeline: TextToSQLPipeline | None = None


def set_pipeline(pipeline: TextToSQLPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> TextToSQLPipeline:
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline requested before startup finished - "
            "this should only happen if it's called outside a request handler."
        )
    return _pipeline


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
