"""Semantic retrieval over the persisted Chroma vector store.

Returns chunks together with citation metadata (source file + page) so the
assistant can ground and cite every answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, MIN_RELEVANCE, TOP_K
from rag.embeddings import embed_query


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float  # cosine similarity in [0, 1]; higher is better

    @property
    def citation(self) -> str:
        return f"{self.source}, p. {self.page}"


@lru_cache(maxsize=1)
def _collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. "
            "Run `python -m rag.ingest` to build it first."
        )
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


def search(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """Return the most relevant chunks for a query, filtered by relevance."""
    collection = _collection()
    res = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]

    hits: List[RetrievedChunk] = []
    for text, meta, dist in zip(docs, metas, distances):
        # Chroma cosine distance = 1 - cosine similarity.
        score = 1.0 - float(dist)
        if score < MIN_RELEVANCE:
            continue
        hits.append(
            RetrievedChunk(
                text=text,
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", 1)),
                score=score,
            )
        )
    return hits
