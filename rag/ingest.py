"""Ingestion pipeline: documents -> page-level chunks -> embeddings -> Chroma.

Run this ONCE locally to build the persisted vector store, then commit the
`chroma_db/` folder so the deployed Space loads it without re-embedding.

    python -m rag.ingest

Supported inputs (placed in ./data):
  * .pdf  -> parsed page by page (page numbers preserved for citations)
  * .md / .txt -> parsed as a single document (page = 1)
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
import tiktoken
from chromadb.config import Settings
from pypdf import PdfReader

from config import CHROMA_DIR, CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, COLLECTION_NAME, DATA_DIR
from rag.embeddings import embed_texts

_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source: str   # file name, e.g. "toyota-hilux-manual.pdf"
    page: int     # 1-based page number (1 for non-paginated docs)
    chunk_id: str


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_pdf(path: Path) -> List[tuple[int, str]]:
    """Return [(page_number, text), ...] for a PDF, skipping empty pages."""
    reader = PdfReader(str(path))
    pages: List[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def _load_text(path: Path) -> List[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(1, text)] if text else []


def _iter_documents(data_dir: Path) -> Iterable[tuple[Path, List[tuple[int, str]]]]:
    files = sorted(
        p for p in data_dir.iterdir()
        if p.suffix.lower() in (".pdf", ".md", ".txt")
    )
    if not files:
        raise FileNotFoundError(
            f"No .pdf/.md/.txt documents found in {data_dir}. "
            "Add your source documents before running ingestion."
        )
    for path in files:
        if path.suffix.lower() == ".pdf":
            yield path, _load_pdf(path)
        else:
            yield path, _load_text(path)


# --------------------------------------------------------------------------- #
# Chunking (token-aware, with overlap, page metadata preserved)
# --------------------------------------------------------------------------- #
def _split_page(text: str, source: str, page: int) -> List[Chunk]:
    tokens = _ENCODER.encode(text)
    if not tokens:
        return []

    chunks: List[Chunk] = []
    step = CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
    for start in range(0, len(tokens), step):
        window = tokens[start : start + CHUNK_TOKENS]
        chunk_text = _ENCODER.decode(window).strip()
        if not chunk_text:
            continue
        idx = len(chunks)
        chunks.append(
            Chunk(
                text=chunk_text,
                source=source,
                page=page,
                chunk_id=f"{source}::p{page}::c{idx}",
            )
        )
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def build_chunks(data_dir: Path = DATA_DIR) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for path, pages in _iter_documents(data_dir):
        doc_chunks: List[Chunk] = []
        for page_no, text in pages:
            doc_chunks.extend(_split_page(text, path.name, page_no))
        print(f"  • {path.name}: {len(pages)} page(s) -> {len(doc_chunks)} chunk(s)")
        all_chunks.extend(doc_chunks)
    return all_chunks


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def build_index(reset: bool = True) -> None:
    print(f"Reading documents from: {DATA_DIR}")
    chunks = build_chunks()
    if not chunks:
        raise RuntimeError("No chunks produced — are the documents empty?")
    print(f"Total chunks: {len(chunks)}")

    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Embedding chunks (this runs locally, may take a few minutes for large PDFs)...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c.source, "page": c.page} for c in chunks],
    )

    print(f"Done. Indexed {collection.count()} chunks into '{COLLECTION_NAME}'.")
    print(f"Vector store persisted at: {CHROMA_DIR}")
    print("Commit the chroma_db/ folder so the deployed app can load it.")


if __name__ == "__main__":
    build_index()
