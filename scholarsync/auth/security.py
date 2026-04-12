"""
Security utilities — password hashing (bcrypt) and JWT token management.

All sensitive configuration is read from environment variables via settings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt, JWTError, ExpiredSignatureError

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


# ── Bcrypt Password Hashing ─────────────────────────────────────────
# Using bcrypt directly (passlib has compatibility issues with bcrypt 5.x)

def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt with automatic salt."""
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# ── JWT Token Management ────────────────────────────────────────────

def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload to encode (typically {"sub": user_id, "username": ...}).
        expires_delta: Token lifetime. Defaults to settings.jwt_expiry_minutes.

    Returns:
        Encoded JWT string.
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_expiry_minutes)

    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict | None:
    """
    Decode and verify a JWT access token.

    Returns:
        Decoded payload dict, or None if token is invalid/expired.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except ExpiredSignatureError:
        logger.debug("JWT token expired")
        return None
    except JWTError as e:
        logger.debug("JWT decode error: %s", e)
        return None
