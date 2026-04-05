"""
GraphRAG Retrieval Service — unified context retrieval combining
vector search (ChromaDB) and knowledge graph traversal (Neo4j).

Single interface for both Normal and Deep Research modes.
"""

from __future__ import annotations

from typing import Any

from scholarsync.rag.vector_store import search as vector_search
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


def _graph_search(query: str, depth: int, top_k: int) -> list[dict]:
    """
    Attempt graph-based entity retrieval via Neo4j.
    Returns empty list if Neo4j is unavailable — graceful degradation.
    """
    try:
        from scholarsync.rag.graph_rag import (
            query_related_entities,
            query_cross_paper_connections,
        )
        # Extract potential entity keywords from the query
        keywords = [w for w in query.split() if len(w) > 3]

        graph_results: list[dict] = []
        seen_names: set[str] = set()

        for keyword in keywords[:5]:  # Limit to top-5 keyword lookups
            try:
                related = query_related_entities(keyword, max_hops=depth)
                for r in related:
                    name = r.get("name", "")
                    if name and name not in seen_names:
                        seen_names.add(name)
                        graph_results.append(r)
            except Exception:
                continue

        # Also pull cross-paper connections for richer context
        if depth >= 2:
            try:
                cross_paper = query_cross_paper_connections()
                for cp in cross_paper[:top_k]:
                    graph_results.append({
                        "name": cp.get("entity", ""),
                        "entity_type": cp.get("entity_type", ""),
                        "description": f"Found across papers: {', '.join(cp.get('papers', []))}",
                        "source_paper": "cross-paper",
                        "hops": 0,
                    })
            except Exception:
                pass

        return graph_results[:top_k]

    except ImportError:
        logger.warning("Neo4j driver not available; graph search skipped")
        return []
    except Exception as e:
        logger.warning("Graph search failed (non-fatal): %s", e)
        return []


def _format_vector_results(results: list[dict]) -> str:
    """Format vector search results into a readable context block."""
    if not results:
        return ""

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        paper = meta.get("paper_title", "Unknown Paper")
        page = meta.get("page_number", "?")
        text = r.get("text", "")
        parts.append(
            f"[Source {i} — \"{paper}\", p.{page}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def _format_graph_results(results: list[dict]) -> str:
    """Format graph traversal results into a readable context block."""
    if not results:
        return ""

    parts: list[str] = []
    for r in results:
        name = r.get("name", "")
        etype = r.get("entity_type", "entity")
        desc = r.get("description", "")
        source = r.get("source_paper", "")
        hops = r.get("hops", "?")
        parts.append(
            f"• {name} ({etype}): {desc}"
            + (f" [from: {source}, hops: {hops}]" if source else "")
        )
    return "\n".join(parts)


def get_context(
    query: str,
    *,
    depth: int = 1,
    top_k: int = 5,
    paper_id: str | None = None,
) -> str:
    """
    Retrieve unified context from vector store + knowledge graph.

    Parameters
    ----------
    query : str
        The search query.
    depth : int
        Graph traversal depth (1 = shallow/normal, 2-3 = deep research).
    top_k : int
        Number of results to retrieve from each source.
    paper_id : str, optional
        Restrict vector search to a specific paper.

    Returns
    -------
    str
        Formatted context string ready for LLM consumption.
    """
    logger.info(
        "GraphRAG get_context: query=%r, depth=%d, top_k=%d",
        query[:80], depth, top_k,
    )

    # 1. Vector search (always available)
    vector_results = vector_search(
        query=query,
        n_results=top_k,
        paper_id=paper_id,
    )
    vector_context = _format_vector_results(vector_results)

    # 2. Graph search (optional, graceful degradation)
    graph_results = _graph_search(query, depth=depth, top_k=top_k)
    graph_context = _format_graph_results(graph_results)

    # 3. Combine
    sections: list[str] = []
    if vector_context:
        sections.append(
            "══ Retrieved Paper Excerpts ══\n\n" + vector_context
        )
    if graph_context:
        sections.append(
            "══ Knowledge Graph Entities & Relationships ══\n\n" + graph_context
        )

    if not sections:
        return "(No relevant context found in the uploaded papers.)"

    combined = "\n\n" + "\n\n".join(sections) + "\n"

    logger.info(
        "GraphRAG returned %d vector chunks + %d graph entities (%d chars)",
        len(vector_results), len(graph_results), len(combined),
    )
    return combined
