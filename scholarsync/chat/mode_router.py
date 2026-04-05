"""
Mode Router — routes chat messages through either Normal or Deep Research
pipeline based on the user's selected mode.

Features:
- Intent detection (greeting vs simple vs complex)
- Response length control based on query type
- Streaming support via generators
- Context bypass for greetings/small talk
"""

from __future__ import annotations

import json
import re
import asyncio
from typing import AsyncGenerator, Any

from scholarsync.chat.key_manager import get_key_manager
from scholarsync.chat.graphrag_service import get_context
from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


# ── Intent Classification ────────────────────────────────────────────

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|howdy|hiya|yo|sup|good\s*(morning|afternoon|evening|night)|"
    r"what'?s\s*up|how\s*are\s*you|how\s*do\s*you\s*do|greetings|namaste|hola|"
    r"thanks?(\s*you)?|thank\s*you|bye|goodbye|see\s*ya|nice\s*to\s*meet|"
    r"how'?s?\s*it\s*going|what'?s\s*good|hey\s*there)[\s!?.]*$",
    re.IGNORECASE,
)


def classify_intent(query: str) -> str:
    """
    Classify user query intent.
    Returns: 'greeting', 'simple', or 'complex'
    """
    q = query.strip()

    # Greeting / small talk
    if GREETING_PATTERNS.match(q):
        return "greeting"

    # Simple question heuristics: short, single clause, no technical depth
    word_count = len(q.split())
    has_complex_verb = bool(re.search(r'\b(compare|analyze|explain|describe|evaluate|discuss|summarize|review)\b', q, re.IGNORECASE))
    if word_count <= 6 and not has_complex_verb:
        return "simple"

    return "complex"


# ── System Prompts ───────────────────────────────────────────────────

GREETING_SYSTEM_PROMPT = """You are ScholarSync, a friendly AI research assistant.
The user is greeting you or making small talk. Respond warmly in 1-2 short sentences.
Do NOT reference papers, context, or research unless asked. Keep it natural and brief."""

NORMAL_SYSTEM_PROMPT = """You are ScholarSync, an expert AI research assistant for academic literature.
Be concise unless explicitly asked to explain in detail.

Response rules by query type:
- Simple factual questions → 2-4 sentences, direct answer
- Complex analytical questions → structured response with 2 paragraphs max
- Always reference source papers by name when citing claims
- Use markdown sparingly (bold key terms, bullet lists only when needed)
- NO paragraph dumping, NO unnecessary headers, NO walls of text
- Be conversational and helpful, not robotic"""


DEEP_DECOMPOSE_PROMPT = """You are a research planning agent. Given a research question, decompose it into exactly 4 focused sub-questions covering: definitions/background, methodologies, key findings, and limitations.

Output JSON: {"sub_questions": ["...", "...", "...", "..."]}"""


DEEP_SYNTHESIS_PROMPT = """You are ScholarSync in DEEP RESEARCH MODE. Produce a COMPREHENSIVE literature analysis.

MANDATORY FORMAT:

# [Descriptive Title]

## Summary
2+ detailed paragraphs covering the overall research landscape.

## Key Insights
5–8 bullet points, each with a bold heading and 2–3 sentence explanation.

## Detailed Analysis

### Background & Definitions
Foundational concepts, key terms — 2+ paragraphs.

### Methodologies & Approaches
How different methods work — 3+ paragraphs.

### Results & Findings
Experimental results, metrics — 3+ paragraphs with specific numbers.

### Comparative Analysis
Compare/contrast approaches. Include a comparison table if applicable.

### Limitations & Challenges
Limitations, risks, unresolved issues — 2+ paragraphs.

## Future Directions
Open questions, promising directions — 1+ paragraph.

## Conclusion
Thorough synthesis — 2+ paragraphs.

## References
List source papers referenced.

CRITICAL: Every section must have substantial content. Use specific data and paper names. No vague summaries."""


# ── Greeting Handler ─────────────────────────────────────────────────

def _handle_greeting(message: str) -> str:
    """Handle greetings with a short, friendly response — no context needed."""
    km = get_key_manager()
    return km.call_llm(
        messages=[
            {"role": "system", "content": GREETING_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        max_tokens=100,
    )


# ── Normal Mode Handler ─────────────────────────────────────────────

def _handle_normal(
    chat_id: str,
    message: str,
    history: list[dict],
    intent: str,
) -> str:
    """Normal mode: retrieve context + single LLM call."""
    settings = get_settings()
    km = get_key_manager()

    history_text = _format_history(history, max_messages=4)

    context = get_context(
        query=message,
        depth=settings.normal_mode_graph_depth,
        top_k=settings.normal_mode_top_k,
    )
    if len(context) > 4000:
        context = context[:4000] + "\n\n[context truncated]"

    # Adjust max_tokens based on intent
    if intent == "simple":
        length_hint = "Answer concisely in 2-4 sentences."
        max_tok = 512
    else:
        length_hint = "Give a clear, structured answer. Be thorough but avoid unnecessary padding."
        max_tok = 2048

    user_prompt = f"""Conversation history:
{history_text}

Research context from papers:
{context}

Question: {message}

{length_hint}"""

    return km.call_llm(
        messages=[
            {"role": "system", "content": NORMAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tok,
    )


# ── Streaming Handlers ───────────────────────────────────────────────

def _stream_greeting(message: str):
    """Stream a greeting response."""
    km = get_key_manager()
    return km.call_llm_stream(
        messages=[
            {"role": "system", "content": GREETING_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        max_tokens=100,
    )


def _stream_normal(
    chat_id: str,
    message: str,
    history: list[dict],
    intent: str,
):
    """Stream a normal mode response."""
    settings = get_settings()
    km = get_key_manager()

    history_text = _format_history(history, max_messages=4)
    context = get_context(
        query=message,
        depth=settings.normal_mode_graph_depth,
        top_k=settings.normal_mode_top_k,
    )
    if len(context) > 4000:
        context = context[:4000] + "\n\n[context truncated]"

    if intent == "simple":
        length_hint = "Answer concisely in 2-4 sentences."
        max_tok = 512
    else:
        length_hint = "Give a clear, structured answer. Be thorough but avoid unnecessary padding."
        max_tok = 2048

    user_prompt = f"""Conversation history:
{history_text}

Research context from papers:
{context}

Question: {message}

{length_hint}"""

    return km.call_llm_stream(
        messages=[
            {"role": "system", "content": NORMAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tok,
    )


def _stream_deep_research(
    chat_id: str,
    message: str,
    history: list[dict],
):
    """
    Generator for deep research streaming.
    Yields progress events first, then the synthesis stream.
    """
    settings = get_settings()
    km = get_key_manager()

    # Step 1: Decompose (non-streamed, fast)
    yield {"event": "progress", "data": "🔍 Decomposing research question..."}

    history_text = _format_history(history, max_messages=3)
    result = km.call_llm(
        messages=[
            {"role": "system", "content": DEEP_DECOMPOSE_PROMPT},
            {"role": "user", "content": f"Conversation context:\n{history_text}\n\nResearch question: {message}\n\nDecompose into 4 focused sub-questions. Output JSON only."},
        ],
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(result)
        sub_questions = parsed.get("sub_questions", [])[:4]
        if not sub_questions:
            raise ValueError("empty")
    except Exception:
        sub_questions = [
            f"What are the key concepts related to: {message}",
            f"What methodologies are used for: {message}",
            f"What are the main findings regarding: {message}",
            f"What are the limitations of: {message}",
        ]

    # Step 2: Retrieve context per sub-question
    all_contexts = []
    for i, sq in enumerate(sub_questions, 1):
        yield {"event": "progress", "data": f"📚 Researching sub-question {i}/{len(sub_questions)}: {sq[:60]}..."}
        ctx = get_context(query=sq, depth=settings.deep_research_graph_depth, top_k=settings.deep_research_top_k)
        if len(ctx) > 1500:
            ctx = ctx[:1500] + "..."
        all_contexts.append(f"Sub-question {i}: {sq}\n\n{ctx}")

    aggregated = "\n\n---\n\n".join(all_contexts)
    if len(aggregated) > 6000:
        aggregated = aggregated[:6000] + "\n\n[truncated]"

    yield {"event": "progress", "data": "✨ Synthesizing comprehensive analysis..."}

    # Step 3: Stream synthesis
    synthesis_prompt = f"""Original research question: {message}

Conversation context:
{history_text}

Research context from {len(sub_questions)} sub-queries:

{aggregated}

Produce a comprehensive deep research analysis following the MANDATORY FORMAT."""

    stream = km.call_llm_stream(
        messages=[
            {"role": "system", "content": DEEP_SYNTHESIS_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ],
        max_tokens=4096,
    )

    for chunk in stream:
        yield {"event": "token", "data": chunk}

    yield {"event": "done", "data": ""}


# ── Shared Helpers ───────────────────────────────────────────────────

def _format_history(history: list[dict], max_messages: int = 4) -> str:
    """Format recent conversation history (trimmed to save tokens)."""
    if not history:
        return "(No prior conversation)"
    recent = history[-max_messages:]
    parts = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if len(content) > 300:
            content = content[:300] + "…"
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ── Public Entry Points ──────────────────────────────────────────────

async def route_message(
    chat_id: str,
    message: str,
    history: list[dict],
    deep_research: bool = False,
) -> str:
    """Non-streaming route (backward compat)."""
    intent = classify_intent(message)
    logger.info("ModeRouter: chat=%s mode=%s intent=%s", chat_id, "DEEP" if deep_research else "NORMAL", intent)

    loop = asyncio.get_event_loop()

    if intent == "greeting" and not deep_research:
        return await loop.run_in_executor(None, _handle_greeting, message)

    if deep_research:
        # For non-streaming deep, collect all tokens
        parts = []
        for event in _handle_deep_research_sync(chat_id, message, history):
            if event.get("event") == "token":
                parts.append(event["data"])
        return "".join(parts)

    return await loop.run_in_executor(
        None, _handle_normal, chat_id, message, history, intent,
    )


def _handle_deep_research_sync(chat_id, message, history):
    """Non-streaming deep research for backward compat."""
    settings = get_settings()
    km = get_key_manager()
    history_text = _format_history(history, max_messages=3)

    # Decompose
    result = km.call_llm(
        messages=[
            {"role": "system", "content": DEEP_DECOMPOSE_PROMPT},
            {"role": "user", "content": f"Research question: {message}\n\nDecompose into 4 sub-questions. JSON only."},
        ],
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    try:
        sub_questions = json.loads(result).get("sub_questions", [])[:4]
        if not sub_questions: raise ValueError()
    except Exception:
        sub_questions = [
            f"Key concepts of: {message}",
            f"Methodologies for: {message}",
            f"Main findings of: {message}",
            f"Limitations of: {message}",
        ]

    all_contexts = []
    for i, sq in enumerate(sub_questions, 1):
        ctx = get_context(query=sq, depth=settings.deep_research_graph_depth, top_k=settings.deep_research_top_k)
        if len(ctx) > 1500:
            ctx = ctx[:1500] + "..."
        all_contexts.append(f"Sub-question {i}: {sq}\n\n{ctx}")

    aggregated = "\n\n---\n\n".join(all_contexts)
    if len(aggregated) > 6000:
        aggregated = aggregated[:6000] + "\n\n[truncated]"

    synthesis_prompt = f"""Original research question: {message}

Context:
{history_text}

Research from {len(sub_questions)} sub-queries:

{aggregated}

Produce a comprehensive analysis following the MANDATORY FORMAT."""

    text = km.call_llm(
        messages=[
            {"role": "system", "content": DEEP_SYNTHESIS_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ],
        max_tokens=4096,
    )
    yield {"event": "token", "data": text}


async def route_message_stream(
    chat_id: str,
    message: str,
    history: list[dict],
    deep_research: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Streaming route — yields dicts with 'event' and 'data' keys.
    Events: 'progress', 'token', 'done'
    """
    intent = classify_intent(message)
    logger.info("ModeRouter[stream]: chat=%s mode=%s intent=%s", chat_id, "DEEP" if deep_research else "NORMAL", intent)

    loop = asyncio.get_event_loop()

    if intent == "greeting" and not deep_research:
        stream = await loop.run_in_executor(None, _stream_greeting, message)
        for chunk in stream:
            yield {"event": "token", "data": chunk}
        yield {"event": "done", "data": ""}
        return

    if deep_research:
        gen = _stream_deep_research(chat_id, message, history)
        for event in gen:
            yield event
        return

    stream = await loop.run_in_executor(
        None, _stream_normal, chat_id, message, history, intent,
    )
    for chunk in stream:
        yield {"event": "token", "data": chunk}
    yield {"event": "done", "data": ""}
