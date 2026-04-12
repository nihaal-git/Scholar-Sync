"""
Auth service — business logic for user registration, login, and management.

All database operations use the SQLite auth database via aiosqlite.
"""

from __future__ import annotations

from scholarsync.auth.database import get_auth_db
from scholarsync.auth.security import hash_password, verify_password
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


async def register_user(username: str, password: str) -> dict | None:
    """
    Register a new user.

    Args:
        username: Unique username (already validated and lowercased).
        password: Plain-text password (already validated for strength).

    Returns:
        User dict on success, None if username already exists.
    """
    pw_hash = hash_password(password)

    db = await get_auth_db()
    try:
        # Check if username already exists
        cursor = await db.execute(
            "SELECT id FROM auth_users WHERE username = ?",
            (username,),
        )
        existing = await cursor.fetchone()
        if existing:
            return None  # Username taken

        # Insert new user
        cursor = await db.execute(
            "INSERT INTO auth_users (username, password_hash) VALUES (?, ?)",
            (username, pw_hash),
        )
        await db.commit()
        user_id = cursor.lastrowid

        logger.info("Registered new user: %s (id=%d)", username, user_id)

        # Fetch the created user
        return await get_user_by_id(user_id)
    except Exception as e:
        logger.error("Registration error for '%s': %s", username, e)
        raise
    finally:
        await db.close()


async def authenticate_user(username: str, password: str) -> dict | None:
    """
    Authenticate a user by username and password.

    Args:
        username: The username to look up.
        password: Plain-text password to verify.

    Returns:
        User dict on success, None if credentials are invalid.
    """
    db = await get_auth_db()
    try:
        cursor = await db.execute(
            "SELECT id, username, password_hash, role, created_at FROM auth_users WHERE username = ?",
            (username.lower(),),
        )
        row = await cursor.fetchone()

        if not row:
            logger.debug("Login failed: user '%s' not found", username)
            return None

        if not verify_password(password, row["password_hash"]):
            logger.debug("Login failed: invalid password for '%s'", username)
            return None

        logger.info("User '%s' authenticated successfully", username)
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "created_at": str(row["created_at"]),
        }
    finally:
        await db.close()


async def get_user_by_id(user_id: int) -> dict | None:
    """
    Fetch a user by their ID.

    Returns:
        User dict or None if not found.
    """
    db = await get_auth_db()
    try:
        cursor = await db.execute(
            "SELECT id, username, role, created_at FROM auth_users WHERE id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "created_at": str(row["created_at"]),
        }
    finally:
        await db.close()


async def change_password(
    user_id: int,
    current_password: str,
    new_password: str,
) -> bool:
    """
    Change a user's password.

    Args:
        user_id: The user's ID.
        current_password: Current password for verification.
        new_password: New password to set.

    Returns:
        True if password was changed, False if current password is wrong.
    """
    db = await get_auth_db()
    try:
        cursor = await db.execute(
            "SELECT password_hash FROM auth_users WHERE id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return False

        if not verify_password(current_password, row["password_hash"]):
            return False

        new_hash = hash_password(new_password)
        await db.execute(
            "UPDATE auth_users SET password_hash = ? WHERE id = ?",
            (new_hash, user_id),
        )
        await db.commit()
        logger.info("Password changed for user id=%d", user_id)
        return True
    finally:
        await db.close()
