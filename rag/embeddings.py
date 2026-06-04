"""Shared embedding model loader.

Uses a local sentence-transformers model (BAAI/bge-small-en-v1.5) so retrieval
costs nothing and runs entirely on the HuggingFace Space CPU. The model is
loaded once and cached for the process lifetime.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load (and cache) the embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents for storage."""
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query.

    bge models recommend prefixing queries with a short instruction to improve
    retrieval quality.
    """
    model = get_model()
    prompt = f"Represent this sentence for searching relevant passages: {text}"
    vector = model.encode(prompt, normalize_embeddings=True)
    return vector.tolist()
