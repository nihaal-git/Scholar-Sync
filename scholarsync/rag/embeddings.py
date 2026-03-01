"""
Embedding utilities — wraps sentence-transformers for generating text embeddings.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Returns list of float vectors.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    return embed_texts([text])[0]
