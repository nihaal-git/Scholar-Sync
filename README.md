# ScholarSync — Agentic AI Literature Review System

An automated multi-agent system that generates trusted, citation-grounded literature reviews from research papers using collaborative AI agents, graph-based reasoning, and iterative validation.

## Architecture

```
User Input → Manager Agent → Worker Agents (parallel) → GraphRAG Engine
           → Checking Agent → Correction Loop → Final Synthesizer → Report
```

**Agents:**
- **Manager Agent** — Decomposes research queries into subtasks
- **Worker Agents** — Extract entities, methods, findings, risks, claims (parallel)
- **Checking Agent** — Validates grounding, detects hallucinations
- **Synthesizer Agent** — Merges validated content into a structured review

**Tech Stack:** Python 3.11+ · LangGraph · Groq (Llama-3) · LlamaIndex · ChromaDB · Neo4j · FastAPI

## Quick Start

### 1. Clone & Install

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

**Required:** A [Groq API key](https://console.groq.com) for the LLM agents.

**Optional:** Neo4j for GraphRAG (the system works without it, but cross-paper graph reasoning is disabled).

### 3. Start the Server

```bash
uvicorn scholarsync.api.main:app --reload
```

Open **http://localhost:8000** in your browser.

### 4. Use It

1. Upload PDFs (up to 10 research papers)
2. Ask a research question
3. Wait for the multi-agent pipeline to process
4. Receive a structured literature review with citations

## Project Structure

```
scholarsync/
├── agents/             # AI Agent implementations
│   ├── manager_agent.py      # Query decomposition & orchestration
│   ├── worker_agent.py       # Knowledge extraction (parallel)
│   ├── checking_agent.py     # Validation & hallucination detection
│   └── synthesizer_agent.py  # Report synthesis with citations
├── rag/                # Retrieval-Augmented Generation
│   ├── embeddings.py         # Sentence-transformer embeddings
│   ├── vector_store.py       # ChromaDB vector search
│   └── graph_rag.py          # Neo4j knowledge graph
├── ingestion/          # Document processing
│   ├── pdf_loader.py         # PDF parsing (PyMuPDF)
│   └── chunker.py            # Semantic chunking (LlamaIndex)
├── evaluation/         # Quality metrics
│   └── grounding_checker.py  # RAGAS-style faithfulness & relevancy
├── workflow/           # Pipeline orchestration
│   └── langgraph_pipeline.py # LangGraph state graph
├── api/                # REST API
│   └── main.py               # FastAPI endpoints
├── config/
│   └── settings.py           # Pydantic configuration
└── utils/
    ├── schemas.py             # Data models
    └── logger.py              # Structured logging
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/upload_papers` | POST | Upload PDF papers for processing |
| `/query` | POST | Start literature review pipeline |
| `/report/{session_id}` | GET | Get report and pipeline progress |

## Docker Deployment

```bash
docker-compose up --build
```

This starts:
- **ScholarSync API** on port 8000
- **Neo4j** on ports 7474 (browser) / 7687 (bolt)

## Command-Line Usage

```bash
# Place PDFs in data/uploads/, then:
python run_example.py
```

## Configuration

All settings are controlled via environment variables or `.env` file. See `.env.example` for all options including:
- `GROQ_API_KEY` — Groq API key (required)
- `GROQ_MODEL` — LLM model (default: `llama-3.3-70b-versatile`)
- `VALIDATION_THRESHOLD` — Minimum score to pass validation (default: 0.7)
- `MAX_CORRECTION_LOOPS` — Max retry iterations (default: 3)

## License

MIT
