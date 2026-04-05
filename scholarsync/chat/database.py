"""
MongoDB connection and database setup for the chat module.

Uses motor (async MongoDB driver) for non-blocking database operations.
Collections: users, chats, messages — all created automatically.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

# ── Module-level client (singleton) ─────────────────────────────────
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_client() -> AsyncIOMotorClient:
    """Return the shared MongoDB client, creating it if needed."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.mongodb_url)
        logger.info("MongoDB client created for %s", settings.mongodb_url)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    """Return the application database handle."""
    global _db
    if _db is None:
        settings = get_settings()
        _db = get_client()[settings.mongodb_db_name]
    return _db


async def init_db() -> None:
    """
    Initialize database indexes. Called once at application startup.

    Creates indexes for efficient querying and data integrity:
    - users.firebase_uid: unique index for fast auth lookups
    - chats.user_id: index for listing user's chats
    - messages.chat_id + timestamp: compound index for ordered history
    """
    db = get_db()

    # Users collection
    await db.users.create_index("firebase_uid", unique=True)
    await db.users.create_index("email")

    # Chats collection
    await db.chats.create_index("user_id")
    await db.chats.create_index([("user_id", 1), ("created_at", -1)])

    # Messages collection
    await db.messages.create_index("chat_id")
    await db.messages.create_index([("chat_id", 1), ("timestamp", 1)])

    logger.info("MongoDB indexes created/verified for database '%s'", db.name)


async def close_db() -> None:
    """Close the MongoDB client connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB client closed")


async def check_connection() -> bool:
    """Check if MongoDB is reachable. Returns True if connected."""
    try:
        client = get_client()
        await client.admin.command("ping")
        return True
    except Exception:
        return False
