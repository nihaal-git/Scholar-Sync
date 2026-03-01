"""
Document chunker — splits extracted text into semantically meaningful chunks
using LlamaIndex's SentenceSplitter.
"""

from __future__ import annotations

import uuid

from llama_index.core.node_parser import SentenceSplitter

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import DocumentChunk, PaperMetadata

logger = get_logger(__name__)


def chunk_document(
    paper_meta: PaperMetadata,
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Split paper pages into overlapping chunks with metadata.

    Parameters
    ----------
    paper_meta : PaperMetadata
        Metadata for the paper.
    pages : list[dict]
        List of page dicts from ``pdf_loader.load_pdf``.
    chunk_size : int, optional
        Override chunk size from settings.
    chunk_overlap : int, optional
        Override chunk overlap from settings.

    Returns
    -------
    list[DocumentChunk]
    """
    settings = get_settings()
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap

    splitter = SentenceSplitter(chunk_size=cs, chunk_overlap=co)

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        page_number = page.get("page_number", 0)

        # Split page text into sentence-based chunks
        text_chunks = splitter.split_text(text)

        for text_chunk in text_chunks:
            if not text_chunk.strip():
                continue

            chunk_id = f"{paper_meta.paper_id}_c{chunk_index:04d}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_meta.paper_id,
                    paper_title=paper_meta.title,
                    text=text_chunk.strip(),
                    page_number=page_number,
                    chunk_index=chunk_index,
                    metadata={
                        "paper_id": paper_meta.paper_id,
                        "paper_title": paper_meta.title,
                        "filename": paper_meta.filename,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "authors": paper_meta.authors,
                    },
                )
            )
            chunk_index += 1

    logger.info(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d)",
        paper_meta.title,
        len(chunks),
        cs,
        co,
    )
    return chunks


def chunk_multiple_documents(
    papers: list[tuple[PaperMetadata, list[dict]]],
) -> dict[str, list[DocumentChunk]]:
    """
    Chunk multiple papers. Returns {paper_id: [DocumentChunk, ...]}.
    """
    all_chunks: dict[str, list[DocumentChunk]] = {}
    for paper_meta, pages in papers:
        chunks = chunk_document(paper_meta, pages)
        paper_meta.total_chunks = len(chunks)
        all_chunks[paper_meta.paper_id] = chunks
    return all_chunks
