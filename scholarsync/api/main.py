"""
FastAPI backend for ScholarSync — provides REST API endpoints for
paper upload, query execution, and report retrieval.

Serves the frontend static files and manages background pipeline tasks.
"""

from __future__ import annotations

import os
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger
from scholarsync.utils.schemas import (
    PipelineStatus,
    PaperMetadata,
    UploadResponse,
    QueryRequest,
    QueryResponse,
    ReportResponse,
    HealthResponse,
    LiteratureReview,
)
from scholarsync.ingestion.pdf_loader import load_pdf
from scholarsync.ingestion.chunker import chunk_document
from scholarsync.rag.vector_store import add_chunks, reset_collection
from scholarsync.workflow.langgraph_pipeline import run_pipeline

logger = get_logger(__name__)
settings = get_settings()

# ── In-memory session store ─────────────────────────────────────────
# In production, use Redis or a database.
sessions: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Ensure directories exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.reports_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    logger.info("ScholarSync API starting up")
    yield
    logger.info("ScholarSync API shutting down")


# ── FastAPI App ─────────────────────────────────────────────────────

app = FastAPI(
    title="ScholarSync API",
    description="Agentic AI Framework for Automated Literature Review",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {
        "api": "ok",
        "groq": "configured" if settings.groq_api_key else "not_configured",
    }

    # Check Neo4j
    try:
        from scholarsync.rag.graph_rag import get_driver
        driver = get_driver()
        driver.verify_connectivity()
        services["neo4j"] = "connected"
    except Exception:
        services["neo4j"] = "unavailable"

    # Check ChromaDB
    try:
        from scholarsync.rag.vector_store import get_collection
        get_collection()
        services["chromadb"] = "connected"
    except Exception:
        services["chromadb"] = "unavailable"

    return HealthResponse(
        status="ok",
        version="1.0.0",
        services=services,
    )


# ── Upload Papers ───────────────────────────────────────────────────

@app.post("/upload_papers", response_model=UploadResponse)
async def upload_papers(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(None),
):
    """
    Upload research papers (PDFs) for processing.

    Parses each PDF, chunks the text, and indexes into the vector store.
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:16]

    if len(files) > settings.max_papers:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_papers} papers allowed",
        )

    # Create session upload directory
    session_dir = Path(settings.upload_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    paper_metadata_list: list[PaperMetadata] = []
    total_chunks = 0

    for uploaded_file in files:
        if not uploaded_file.filename or not uploaded_file.filename.lower().endswith(".pdf"):
            logger.warning("Skipping non-PDF file: %s", uploaded_file.filename)
            continue

        # Save file to disk
        file_path = session_dir / uploaded_file.filename
        with open(file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)

        try:
            # Parse PDF
            paper_meta, pages = load_pdf(file_path)

            # Chunk document
            chunks = chunk_document(paper_meta, pages)
            paper_meta.total_chunks = len(chunks)

            # Add to vector store
            add_chunks(chunks)

            paper_metadata_list.append(paper_meta)
            total_chunks += len(chunks)

            logger.info(
                "Processed '%s': %d pages, %d chunks",
                uploaded_file.filename,
                len(pages),
                len(chunks),
            )
        except Exception as e:
            logger.error("Failed to process %s: %s", uploaded_file.filename, e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {uploaded_file.filename}: {str(e)}",
            )

    if not paper_metadata_list:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

    # Store session data
    sessions[session_id] = {
        "paper_metadata": paper_metadata_list,
        "status": PipelineStatus.PENDING,
        "pipeline_state": None,
    }

    return UploadResponse(
        session_id=session_id,
        papers=paper_metadata_list,
        total_chunks=total_chunks,
        message=f"Successfully processed {len(paper_metadata_list)} papers ({total_chunks} chunks)",
    )


# ── Query — Start Pipeline ─────────────────────────────────────────

def _run_pipeline_bg(session_id: str, query: str, paper_metadata: list[PaperMetadata]):
    """Background task to run the pipeline."""
    try:
        sessions[session_id]["status"] = PipelineStatus.PLANNING
        final_state = run_pipeline(session_id, query, paper_metadata)
        sessions[session_id]["pipeline_state"] = final_state
        sessions[session_id]["status"] = PipelineStatus(final_state.get("status", "completed"))
    except Exception as e:
        logger.error("Pipeline failed for session %s: %s", session_id, e)
        sessions[session_id]["status"] = PipelineStatus.FAILED
        sessions[session_id]["pipeline_state"] = {
            "errors": [str(e)],
            "progress_messages": [f"❌ Pipeline failed: {str(e)}"],
            "status": PipelineStatus.FAILED.value,
        }


@app.post("/query", response_model=QueryResponse)
async def start_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Start a literature review pipeline for the given query.

    The pipeline runs in the background. Poll GET /report/{session_id} for results.
    """
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload papers first.")

    paper_metadata = session["paper_metadata"]

    # Start pipeline in background
    background_tasks.add_task(
        _run_pipeline_bg,
        request.session_id,
        request.query,
        paper_metadata,
    )

    sessions[request.session_id]["status"] = PipelineStatus.INGESTING

    return QueryResponse(
        session_id=request.session_id,
        status=PipelineStatus.INGESTING,
        message="Pipeline started. Poll /report/{session_id} for results.",
    )


# ── Get Report ──────────────────────────────────────────────────────

@app.get("/report/{session_id}", response_model=ReportResponse)
async def get_report(session_id: str):
    """
    Get the current status and report for a pipeline session.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline_state = session.get("pipeline_state") or {}
    status = session.get("status", PipelineStatus.PENDING)

    # Parse report if available
    report = None
    report_md = pipeline_state.get("report_markdown", "")
    if pipeline_state.get("final_report"):
        try:
            report = LiteratureReview(**pipeline_state["final_report"])
        except Exception:
            pass

    return ReportResponse(
        session_id=session_id,
        status=status,
        report=report,
        report_markdown=report_md,
        progress_messages=pipeline_state.get("progress_messages", []),
        errors=pipeline_state.get("errors", []),
    )


# ── Serve Frontend ──────────────────────────────────────────────────

# Resolve the chatbot-ui directory relative to project root
_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_dir = _project_root / "chatbot-ui"


@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML."""
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        {"message": "ScholarSync API is running. Frontend not found at expected path."},
        status_code=200,
    )


# Mount static files if directory exists
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


# ── Run Server ──────────────────────────────────────────────────────

def start_server():
    """Start the uvicorn server (for use with `python -m scholarsync.api.main`)."""
    import uvicorn

    uvicorn.run(
        "scholarsync.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    start_server()
