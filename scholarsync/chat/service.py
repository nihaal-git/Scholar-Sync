"""
Chat service — business logic for multi-user chat with agent integration.

Handles CRUD for chats/messages and calls the existing pipeline as a
black-box function. Ensures strict user isolation on every operation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from scholarsync.chat.database import get_db
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _new_id() -> str:
    """Generate a short unique ID for documents."""
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _auto_title(message: str) -> str:
    """Generate a chat title from the first user message."""
    title = message.strip()[:60]
    if len(message.strip()) > 60:
        title += "…"
    return title


# ── Create Chat ──────────────────────────────────────────────────────

async def create_chat(user_id: str, title: str | None = None) -> dict:
    """
    Create a new chat for a user.

    Args:
        user_id: The authenticated user's _id.
        title: Optional title. If None, will be set later from first message.

    Returns:
        dict with chat_id, title, created_at.
    """
    db = get_db()
    chat_doc = {
        "chat_id": _new_id(),
        "user_id": user_id,
        "title": title or "New Chat",
        "created_at": _now(),
    }
    await db.chats.insert_one(chat_doc)
    logger.info("Created chat %s for user %s", chat_doc["chat_id"], user_id)

    return {
        "chat_id": chat_doc["chat_id"],
        "title": chat_doc["title"],
        "created_at": chat_doc["created_at"],
    }


# ── List User Chats ─────────────────────────────────────────────────

async def list_user_chats(
    user_id: str,
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """
    List all chats for a user, ordered by most recent first.
    Includes message count and last message preview.

    Returns:
        dict with chats list, total, page, per_page.
    """
    db = get_db()
    skip = (page - 1) * per_page

    # Total count
    total = await db.chats.count_documents({"user_id": user_id})

    # Fetch chats
    cursor = (
        db.chats.find({"user_id": user_id})
        .sort("created_at", -1)
        .skip(skip)
        .limit(per_page)
    )
    chats = await cursor.to_list(length=per_page)

    result = []
    for chat in chats:
        # Count messages
        msg_count = await db.messages.count_documents({"chat_id": chat["chat_id"]})

        # Get last message preview
        last_msg = await db.messages.find_one(
            {"chat_id": chat["chat_id"]},
            sort=[("timestamp", -1)],
        )
        preview = None
        if last_msg:
            preview = last_msg["content"][:80]
            if len(last_msg["content"]) > 80:
                preview += "…"

        result.append({
            "id": chat["chat_id"],
            "title": chat["title"],
            "created_at": chat["created_at"],
            "message_count": msg_count,
            "last_message_preview": preview,
        })

    return {
        "chats": result,
        "total": total,
        "page": page,
        "per_page": per_page,
    }


# ── Get Chat History ─────────────────────────────────────────────────

async def get_chat_history(user_id: str, chat_id: str) -> dict:
    """
    Load full message history for a chat. Verifies user ownership.

    Returns:
        dict with chat_id, title, messages.

    Raises:
        ValueError: If chat not found or doesn't belong to user.
    """
    db = get_db()

    # Verify ownership
    chat = await db.chats.find_one({"chat_id": chat_id, "user_id": user_id})
    if not chat:
        raise ValueError("Chat not found or access denied.")

    # Fetch messages ordered by timestamp
    cursor = db.messages.find({"chat_id": chat_id}).sort("timestamp", 1)
    messages = await cursor.to_list(length=10000)

    return {
        "chat_id": chat_id,
        "title": chat["title"],
        "messages": [
            {
                "id": msg["message_id"],
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            }
            for msg in messages
        ],
    }


# ── Send Message ─────────────────────────────────────────────────────

async def send_message(user_id: str, chat_id: str, message: str, deep_research: bool = False) -> dict:
    """
    Process a user message:
    1. Verify chat ownership
    2. Save user message
    3. Fetch conversation history
    4. Call the agent pipeline (normal or deep research)
    5. Save assistant response
    6. Return the response

    Args:
        user_id: Authenticated user's ID.
        chat_id: Target chat ID.
        message: The user's message text.
        deep_research: Whether to use deep research mode.

    Returns:
        dict with chat_id, message_id, response, timestamp.
    """
    db = get_db()

    # 1. Verify chat belongs to user
    chat = await db.chats.find_one({"chat_id": chat_id, "user_id": user_id})
    if not chat:
        raise ValueError("Chat not found or access denied.")

    now = _now()

    # 2. Save user message
    user_msg_id = _new_id()
    await db.messages.insert_one({
        "message_id": user_msg_id,
        "chat_id": chat_id,
        "role": "user",
        "content": message,
        "timestamp": now,
    })

    # Auto-generate title from first message if still default
    if chat.get("title") == "New Chat":
        await db.chats.update_one(
            {"chat_id": chat_id},
            {"$set": {"title": _auto_title(message)}},
        )

    # 3. Fetch previous messages for context
    cursor = db.messages.find({"chat_id": chat_id}).sort("timestamp", 1)
    all_messages = await cursor.to_list(length=10000)

    # 4. Build context and call the agent (mode-aware)
    response_text = await _call_agent(chat_id, message, all_messages, deep_research=deep_research)

    # 5. Save assistant response
    assistant_msg_id = _new_id()
    response_time = _now()
    await db.messages.insert_one({
        "message_id": assistant_msg_id,
        "chat_id": chat_id,
        "role": "assistant",
        "content": response_text,
        "timestamp": response_time,
    })

    logger.info("Processed message in chat %s: user msg %s -> assistant msg %s",
                chat_id, user_msg_id, assistant_msg_id)

    return {
        "chat_id": chat_id,
        "message_id": assistant_msg_id,
        "response": response_text,
        "timestamp": response_time,
    }


async def _call_agent(
    chat_id: str,
    current_message: str,
    history: list[dict],
    *,
    deep_research: bool = False,
) -> str:
    """
    Route the message through normal or deep research mode.

    Uses the new mode_router which handles GraphRAG retrieval and
    API key rotation internally. Falls back gracefully on errors.
    """
    from scholarsync.chat.mode_router import route_message

    try:
        response = await route_message(
            chat_id=chat_id,
            message=current_message,
            history=history,
            deep_research=deep_research,
        )
        return response

    except Exception as e:
        logger.error("Agent error for chat %s: %s", chat_id, e)
        return (
            "I'm sorry, I encountered an error while processing your request. "
            "Please ensure that research papers have been uploaded and try again.\n\n"
            f"Error details: {str(e)}"
        )


# ── Delete Chat ──────────────────────────────────────────────────────

async def delete_chat(user_id: str, chat_id: str) -> bool:
    """
    Delete a chat and all its messages. Verifies user ownership.

    Returns:
        True if deleted, False if not found.
    """
    db = get_db()

    # Verify ownership
    result = await db.chats.delete_one({"chat_id": chat_id, "user_id": user_id})
    if result.deleted_count == 0:
        return False

    # Delete all messages
    del_result = await db.messages.delete_many({"chat_id": chat_id})
    logger.info(
        "Deleted chat %s and %d messages for user %s",
        chat_id, del_result.deleted_count, user_id,
    )
    return True
