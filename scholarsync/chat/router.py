"""
Chat API router — FastAPI endpoints for multi-user chat.

All endpoints require Firebase authentication and enforce
strict user-level data isolation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from scholarsync.chat.firebase_auth import get_current_user
from scholarsync.chat import service
from scholarsync.chat.schemas import (
    CreateChatRequest,
    CreateChatResponse,
    SendMessageRequest,
    SendMessageResponse,
    ChatHistoryResponse,
    ChatListResponse,
    MessageOut,
    ChatOut,
)
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── POST /chat/create ────────────────────────────────────────────────

@router.post("/create", response_model=CreateChatResponse)
async def create_chat(
    request: CreateChatRequest,
    user: dict = Depends(get_current_user),
):
    """
    Create a new chat session for the authenticated user.

    If no title is provided, it will be auto-generated from the first message.
    """
    result = await service.create_chat(
        user_id=user["_id"],
        title=request.title,
    )
    return CreateChatResponse(**result)


# ── POST /chat/message ───────────────────────────────────────────────

@router.post("/message", response_model=SendMessageResponse)
async def send_message(
    request: SendMessageRequest,
    user: dict = Depends(get_current_user),
):
    """
    Send a message in a chat and receive the agent's response.

    The agent processes the message with full conversation history context.
    """
    try:
        result = await service.send_message(
            user_id=user["_id"],
            chat_id=request.chat_id,
            message=request.message,
            deep_research=request.deep_research,
        )
        return SendMessageResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error in send_message: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process message.")


# ── GET /chat/list ───────────────────────────────────────────────────

@router.get("/list", response_model=ChatListResponse)
async def list_chats(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    user: dict = Depends(get_current_user),
):
    """
    List all chats for the authenticated user, ordered by most recent.
    Supports pagination.
    """
    result = await service.list_user_chats(
        user_id=user["_id"],
        page=page,
        per_page=per_page,
    )
    return ChatListResponse(
        chats=[ChatOut(**c) for c in result["chats"]],
        total=result["total"],
        page=result["page"],
        per_page=result["per_page"],
    )


# ── GET /chat/{chat_id} ─────────────────────────────────────────────

@router.get("/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Get the full message history for a specific chat.
    Only accessible by the chat owner.
    """
    try:
        result = await service.get_chat_history(
            user_id=user["_id"],
            chat_id=chat_id,
        )
        return ChatHistoryResponse(
            chat_id=result["chat_id"],
            title=result["title"],
            messages=[MessageOut(**m) for m in result["messages"]],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── DELETE /chat/{chat_id} ───────────────────────────────────────────

@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Delete a chat and all its messages.
    Only the chat owner can delete it.
    """
    deleted = await service.delete_chat(
        user_id=user["_id"],
        chat_id=chat_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")
    return {"message": "Chat deleted successfully.", "chat_id": chat_id}
