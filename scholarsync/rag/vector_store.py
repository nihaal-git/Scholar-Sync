"""
ChromaDB vector store — index and retrieve document chunks by semantic similarity.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from scholarsync.config.settings import get_settings
from scholarsync.rag.embeddings import embed_texts, embed_single
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import DocumentChunk

logger = get_logger(__name__)

# Module-level client cache
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB persistent client."""
    global _client
    if _client is None:
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(persist_dir))
        logger.info("ChromaDB client initialised at %s", persist_dir)
    return _client


def get_collection(name: str | None = None) -> chromadb.Collection:
    """Get or create the default collection."""
    global _collection
    if _collection is None:
        settings = get_settings()
        coll_name = name or settings.chroma_collection_name
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Using ChromaDB collection: %s", coll_name)
    return _collection


def add_chunks(chunks: list[DocumentChunk]) -> int:
    """
    Add document chunks to the vector store.

    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    collection = get_collection()

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "paper_id": c.paper_id,
            "paper_title": c.paper_title,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    # Generate embeddings in batches
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_embeddings = embed_texts(batch_texts)

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
        )

    logger.info("Added %d chunks to vector store", len(chunks))
    return len(chunks)


def search(
    query: str,
    n_results: int = 10,
    paper_id: str | None = None,
) -> list[dict]:
    """
    Semantic similarity search over stored chunks.

    Parameters
    ----------
    query : str
        The search query.
    n_results : int
        Number of results to return.
    paper_id : str, optional
        Filter results to a specific paper.

    Returns
    -------
    list[dict]
        List of result dicts with keys: id, text, metadata, distance.
    """
    collection = get_collection()
    query_embedding = embed_single(query)

    where_filter = None
    if paper_id:
        where_filter = {"paper_id": paper_id}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results["ids"] and results["ids"][0]:
        for idx in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
            })

    return output


def reset_collection(name: str | None = None) -> None:
    """Delete and recreate the collection."""
    global _collection
    settings = get_settings()
    coll_name = name or settings.chroma_collection_name
    client = _get_client()
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    _collection = None
    logger.info("Reset collection: %s", coll_name)
