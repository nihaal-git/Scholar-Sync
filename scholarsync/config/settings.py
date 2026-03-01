"""
Application settings — loaded from environment variables / .env file.
Uses Pydantic Settings for typed, validated config.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for ScholarSync."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "ScholarSync"
    debug: bool = False

    # ── Groq LLM ────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 4096

    # ── Neo4j (GraphRAG) ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── ChromaDB (Vector Store) ─────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "scholarsync_papers"

    # ── Embeddings ──────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Document Processing ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_papers: int = 10
    upload_dir: str = "./data/uploads"

    # ── Validation ──────────────────────────────────────────────────
    validation_threshold: float = 0.7
    max_correction_loops: int = 3

    # ── API ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Reports ─────────────────────────────────────────────────────
    reports_dir: str = "./data/reports"


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton to avoid re-reading .env on every call."""
    return Settings()
