"""
PDF document loader — extracts text and metadata from research papers using PyMuPDF.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF

from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import PaperMetadata

logger = get_logger(__name__)


def load_pdf(file_path: str | Path) -> tuple[PaperMetadata, list[dict]]:
    """
    Load a PDF file and extract text + metadata.

    Returns
    -------
    tuple[PaperMetadata, list[dict]]
        Paper metadata and a list of dicts with keys:
        ``page_number``, ``text``, ``metadata``.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("Loading PDF: %s", file_path.name)

    doc = fitz.open(str(file_path))
    paper_id = uuid.uuid4().hex[:12]

    # ── Extract metadata ────────────────────────────────────────────
    meta = doc.metadata or {}
    title = meta.get("title", "") or file_path.stem
    author_raw = meta.get("author", "")
    authors = [a.strip() for a in author_raw.split(",") if a.strip()] if author_raw else []

    # Try to parse year from the creation date
    year = None
    creation_date = meta.get("creationDate", "")
    if creation_date and len(creation_date) >= 6:
        try:
            year = int(creation_date[2:6])  # D:YYYYMMDD...
        except (ValueError, IndexError):
            pass

    # ── Extract pages ───────────────────────────────────────────────
    pages: list[dict] = []
    total_pages = len(doc)  # capture before closing
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "metadata": {
                    "paper_id": paper_id,
                    "paper_title": title,
                    "filename": file_path.name,
                    "page_number": page_num + 1,
                },
            })

    doc.close()

    paper_meta = PaperMetadata(
        paper_id=paper_id,
        filename=file_path.name,
        title=title,
        authors=authors,
        year=year,
        total_pages=total_pages,  # use pre-captured value, not closed doc
        total_chunks=0,  # updated after chunking
    )

    logger.info(
        "Loaded '%s' — %d pages, title='%s'",
        file_path.name,
        len(pages),
        title,
    )
    return paper_meta, pages


def load_multiple_pdfs(directory: str | Path) -> list[tuple[PaperMetadata, list[dict]]]:
    """Load all PDFs from a directory."""
    directory = Path(directory)
    results = []
    for pdf_file in sorted(directory.glob("*.pdf")):
        try:
            results.append(load_pdf(pdf_file))
        except Exception as e:
            logger.error("Failed to load %s: %s", pdf_file.name, e)
    return results
