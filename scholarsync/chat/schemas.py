"""
Pydantic schemas for the chat API — request/response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────

class CreateChatRequest(BaseModel):
    """Request body for creating a new chat."""
    title: Optional[str] = Field(
        None,
        description="Chat title. Auto-generated from first message if omitted.",
    )


class SendMessageRequest(BaseModel):
    """Request body for sending a message in a chat."""
    chat_id: str
    message: str = Field(..., min_length=1, description="The user's message text.")
    deep_research: bool = Field(False, description="Enable deep research mode")


# ── Responses ────────────────────────────────────────────────────────

class CreateChatResponse(BaseModel):
    """Response after creating a chat."""
    chat_id: str
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    """A single message in chat history."""
    id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime


class SendMessageResponse(BaseModel):
    """Response after sending a message."""
    chat_id: str
    message_id: str
    response: str
    timestamp: datetime


class ChatOut(BaseModel):
    """Summary of a chat for listing."""
    id: str
    title: str
    created_at: datetime
    message_count: int = 0
    last_message_preview: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Full chat history response."""
    chat_id: str
    title: str
    messages: list[MessageOut]


class ChatListResponse(BaseModel):
    """Paginated list of user chats."""
    chats: list[ChatOut]
    total: int
    page: int
    per_page: int
